"""Plugin wrapping :class:`lgq.trainer.LGQTrainer`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from lgq.config import LGQConfig
from lgq.metrics import psnr as _psnr
from lgq.trainer import LGQTrainer

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)


def _tiny_config() -> LGQConfig:
    """A small LGQConfig used for smoke tests and dry-run.

    Resolution 32 keeps the encoder cheap; ``downsample_factor=4`` gives
    8×8 latent maps. Discriminator starts at step 5 so the first call to
    :meth:`step` exercises the generator-only path.
    """
    return LGQConfig(
        n_codebooks=4,
        vocab_size=32,
        codebook_dim=4,
        vq_dim=16,
        in_channels=3,
        hidden_dim=32,
        n_res_blocks=1,
        downsample_factor=4,
        resolution=32,
        tau_init=1.0,
        tau_final=0.05,
        tau_warmup_steps=10,
        tau_anneal_steps=100,
        disc_hidden_dim=16,
        disc_n_layers=2,
        disc_start_step=5,
        batch_size=2,
        total_steps=1000,
        warmup_steps=0,           # LambdaLR(step=0) = 0 with positive warmup; smoke
                                  # tests need a non-zero first step.
        precision="fp32",
    )


class LGQPlugin(ComponentPlugin):
    name = "lgq"

    def __init__(
        self,
        config: LGQConfig | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config if config is not None else _tiny_config()
        self._trainer = LGQTrainer(self.config, device=device)

    # -- required surface --------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    @property
    def device(self) -> torch.device:
        return self._trainer.device

    def make_synthetic_batch(self, batch_size: int = 2) -> torch.Tensor:
        cfg = self.config
        return torch.rand(batch_size, cfg.in_channels, cfg.resolution, cfg.resolution)

    def _generator_forward(
        self, x: torch.Tensor, *, use_adv: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Single generator forward + criterion.

        Mirrors the generator-side of :meth:`LGQTrainer.step` but exposes
        ``model_out`` and the losses dict so the plugin can fuse replay
        + DER consistency + EWC penalty into one backward instead of
        delegating to the trainer's self-contained step.

        Returns ``(model_out, gen_losses)``.
        """
        trainer = self._trainer
        with torch.amp.autocast(
            "cuda", dtype=trainer.amp_dtype, enabled=trainer.use_amp,
        ):
            model_out = trainer.model(x)
            adv_loss = None
            if use_adv and trainer.step_count >= self.config.disc_start_step:
                adv_loss = trainer.model.generator_adversarial_loss(
                    model_out["recon"],
                )
            gen_losses = trainer.loss_fn.generator_loss(
                x, model_out, adv_loss, trainer.step_count,
            )
        return model_out, gen_losses

    def train_step(self, batch: torch.Tensor) -> StepReport:
        """One LGQ training step with fused DER++ + optional EWC + EMA.

        Restructures the dual-optimizer GAN trainer step so the
        generator-side forward + backward is owned by the plugin
        (giving EWC's Fisher accumulation access to the task
        gradient and letting DER++ replay fold into the same
        backward). The discriminator-side step is otherwise
        unchanged — EWC anchors only the generator's parameters
        (the conventional choice for VQ-GAN-style anchoring).

        Sequence (per step):
          1. Generator forward on the fresh batch → ``gen_losses``.
          2. If replay attached + non-empty: forward on the replay
             batch; add ``replay_weight·replay_task`` to total;
             add ``der_weight·MSE(replay_recon, stored_recon)``
             where the stored signal is the recon captured at
             insertion time.
          3. ``total.backward()`` (with AMP scaler if configured).
          4. If EWC attached: ``consolidate(model)`` → ``penalty``
             → ``penalty.backward()``.
          5. Clip + ``opt_gen.step()`` + ``sched_gen.step()``.
          6. If EWC attached: ``post_step(model)``.
          7. Discriminator step (unchanged, uses detached recon).
          8. Insert ``batch`` into the buffer with the current
             recon as the DER teacher signal.
          9. EMA update if attached.
        """
        trainer = self._trainer
        cfg = self.config
        model = trainer.model
        x = batch.to(trainer.device, non_blocking=True)
        model.train()

        # 1. Fresh-batch generator forward.
        model_out, gen_losses = self._generator_forward(x, use_adv=True)
        total = gen_losses["total"]

        losses: dict[str, float] = {}
        for k, v in gen_losses.items():
            if isinstance(v, torch.Tensor):
                losses[k] = float(v.detach().cpu().item())
            elif isinstance(v, (int, float)):
                losses[k] = float(v)

        # 2. Fused DER++ replay. One forward over the replay batch
        # serves both the task-loss term and the DER consistency
        # term (the buffer always carries stored recons for LGQ —
        # buffer insertion below sets them on every step).
        if self.replay_bank is not None and self.replay_bank.size(self.name) > 0:
            items = self.replay_bank.sample(self.name, k=self.replay_batch_size)
            if items:
                rx = torch.stack([it.sample for it in items]).to(trainer.device)
                # ``use_adv=False`` on the replay forward to keep the
                # discriminator loss out of the replay-side gradient —
                # adversarial training on stale clips would push the
                # generator toward fitting whatever the *current*
                # discriminator wants on the buffer, which isn't the
                # forgetting-defense goal.
                r_out, r_losses = self._generator_forward(rx, use_adv=False)
                total = total + self.replay_weight * r_losses["total"]
                losses["replay/total"] = float(r_losses["total"].detach().cpu().item())

                der_items = [it for it in items if it.stored_logits is not None]
                if der_items:
                    # We can reuse ``r_out`` only when *every* item
                    # carries stored logits (the common case after a
                    # warm-up step). Otherwise we'd need a separate
                    # forward over the gated subset.
                    if len(der_items) == len(items):
                        rd_recon = r_out["recon"]
                        stored = torch.stack(
                            [it.stored_logits for it in der_items],
                        ).to(trainer.device)
                        der = nn.functional.mse_loss(rd_recon, stored)
                    else:
                        rd_x = torch.stack(
                            [it.sample for it in der_items],
                        ).to(trainer.device)
                        rd_out, _ = self._generator_forward(rd_x, use_adv=False)
                        stored = torch.stack(
                            [it.stored_logits for it in der_items],
                        ).to(trainer.device)
                        der = nn.functional.mse_loss(rd_out["recon"], stored)
                    total = total + self.der_weight * der
                    losses["der_consistency"] = float(der.detach().cpu().item())

        # 3. Backward + 4. EWC + 5. Clip + opt_gen.step + sched_gen.step.
        trainer.opt_gen.zero_grad(set_to_none=True)
        if trainer.scaler is not None:
            trainer.scaler.scale(total).backward()
            # Unscale before EWC's grad-based Fisher accumulation so the
            # scale doesn't pollute the empirical Fisher.
            trainer.scaler.unscale_(trainer.opt_gen)
        else:
            total.backward()

        if self.ewc is not None:
            self.ewc.consolidate(model)
            penalty = self.ewc.penalty(model)
            penalty.backward()
            losses["ewc/penalty"] = float(penalty.detach().cpu().item())

        nn.utils.clip_grad_norm_(
            trainer._param_groups["generator"], cfg.max_grad_norm,
        )
        if trainer.scaler is not None:
            trainer.scaler.step(trainer.opt_gen)
            trainer.scaler.update()
        else:
            trainer.opt_gen.step()
        trainer.sched_gen.step()

        if self.ewc is not None:
            self.ewc.post_step(model)

        # 7. Discriminator step (after warmup). Operates on detached
        # recon from the *fresh* batch — we don't run the discriminator
        # on replayed images.
        disc_loss_val: float | None = None
        if trainer.step_count >= cfg.disc_start_step:
            with torch.amp.autocast(
                "cuda", dtype=trainer.amp_dtype, enabled=trainer.use_amp,
            ):
                disc_out = model.forward_discriminator(
                    x, model_out["recon"].detach(),
                )
            trainer.opt_disc.zero_grad(set_to_none=True)
            if trainer.scaler is not None:
                trainer.scaler.scale(disc_out["d_loss"]).backward()
                trainer.scaler.step(trainer.opt_disc)
                trainer.scaler.update()
            else:
                disc_out["d_loss"].backward()
                trainer.opt_disc.step()
            trainer.sched_disc.step()
            disc_loss_val = float(disc_out["d_loss"].detach().cpu().item())

        if disc_loss_val is not None:
            losses["d_loss"] = disc_loss_val

        trainer.step_count += 1

        # 8. Buffer insertion with DER teacher signal.
        if self.replay_bank is not None:
            with torch.no_grad():
                stored_recon = model(x)["recon"].detach().cpu()
            for i in range(batch.shape[0]):
                self.replay_bank.insert(
                    self.name,
                    batch[i].detach().cpu(),
                    stored_logits=stored_recon[i],
                    step=trainer.step_count,
                )

        # 9. EMA update.
        if self.ema_teacher is not None:
            self.ema_teacher.update(model)

        losses["total"] = float(total.detach().cpu().item())
        return StepReport(loss=losses["total"], losses=losses)

    @torch.no_grad()
    def evaluate(self, probe_set: Any | None = None) -> EvalReport:
        """Score on a probe set: PSNR (primary) + reconstruction MSE.

        PSNR is bounded above and behaves smoothly; MSE catches the
        degenerate-mean reconstruction case PSNR averaging can mask.
        """
        if probe_set is None:
            from scripts.training.self_improve.eval_registry import build_lgq_probe
            probe_set = build_lgq_probe(
                resolution=self.config.resolution,
                in_channels=self.config.in_channels,
            )

        model = self._trainer.model
        was_training = model.training
        model.eval()
        try:
            psnr_sum = 0.0
            mse_sum = 0.0
            n = 0
            for batch in probe_set:
                x = batch.to(self.device, non_blocking=True)
                out = model(x)
                recon = out["recon"].clamp(0, 1)
                psnr_sum += float(_psnr(recon, x).item()) * x.shape[0]
                mse_sum += float(torch.nn.functional.mse_loss(recon, x).item()) * x.shape[0]
                n += x.shape[0]
        finally:
            model.train(was_training)

        psnr_val = psnr_sum / max(n, 1)
        mse = mse_sum / max(n, 1)
        return EvalReport(
            score=psnr_val,
            metrics={"psnr": psnr_val, "mse": mse},
            higher_is_better=True,
        )

    # -- state -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        out = dict(self._trainer.state_dict())
        ema = self._ema_state()
        if ema is not None:
            out["_ema"] = ema
        ewc = self._ewc_state()
        if ewc is not None:
            out["_ewc"] = ewc
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        trainer_state = {
            k: v for k, v in state.items() if k not in ("_ema", "_ewc")
        }
        self._trainer.load_state_dict(trainer_state)
        self._load_ema_state(state.get("_ema"))
        self._load_ewc_state(state.get("_ewc"))

    # -- helpers -----------------------------------------------------------

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "LGQPlugin":
        """Build with a tiny config plus per-field overrides."""
        cfg = replace(_tiny_config(), **overrides)
        return cls(cfg)


__all__ = ["LGQPlugin"]
