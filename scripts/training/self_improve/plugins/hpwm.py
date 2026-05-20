"""Plugin wrapping the HPWM model.

We bypass :class:`hpwm.train.Trainer` because that class also constructs
dataloaders, an evaluator, and a TensorBoard writer in ``__init__`` —
machinery the orchestrator does not need and that complicates state
management. Instead we instantiate the model directly, configure the
same optimizer + scheduler the upstream trainer would, and run the same
forward/backward path one step at a time.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from hpwm.config import HPWMConfig
from hpwm.model import HPWM

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)


def _tiny_config() -> HPWMConfig:
    """Smaller HPWMConfig for smoke tests.

    Keeps the architectural choices (Mamba, slots, MoD) but shrinks
    everything spatially and reduces temporal extent. DINO loads via the
    random fallback in offline test environments.
    """
    return HPWMConfig(
        resolution=64,
        fps=1,
        clip_length_s=2,
        n_frames=2,
        n_patches=16,
        patch_grid=4,
        fwm_channels=64,
        d_heavy=32,
        n_heavy_layers=1,
        fwm_layers=1,
        n_slots=4,
        d_slot=32,
        slot_iters=2,
        slot_mlp_hidden=64,
        n_temporal_tiers=1,
        d_mamba=64,
        mamba_n_layers=1,
        n_scales=1,
        n_layers_fast=1,
        n_layers_slow=1,
        d_fast=32,
        d_slow=32,
        token_budget=16,
        n_heads=2,
        vqvae_codebooks=2,
        vqvae_vocab_size=16,
        vqvae_dim=16,
        vqvae_hidden=32,
        vqvae_n_layers=1,
        batch_size=1,
        grad_accum_steps=1,
        grad_checkpointing=False,
        precision="fp32",
        total_steps=1000,
        warmup_steps=0,           # avoid lambda(0)=0 in cosine warmup
        vqvae_warmup_steps=0,
        pred_warmup_steps=0,
    )


class HPWMPlugin(ComponentPlugin):
    name = "hpwm"

    def __init__(
        self,
        config: HPWMConfig | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config if config is not None else _tiny_config()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._model = HPWM(self.config).to(self._device)
        # Force the lazy DINO backbone to materialize *now* so it
        # contributes keys to both saved and loaded state dicts. Without
        # this, an HPWM saved after one forward pass holds ``dino._model.*``
        # keys that a freshly-constructed plugin silently drops on
        # ``load_state_dict(..., strict=False)``, breaking faithful
        # snapshot/rollback (matters for offline fallback DINO and for
        # any future unfrozen / fine-tuned DINO).
        self._model.dino._load_model(self._device)

        param_groups = self._model.get_param_groups()
        self.optimizer = torch.optim.AdamW(
            [
                {"params": g["params"], "lr": self.config.lr * g["lr_scale"]}
                for g in param_groups
                if g["params"]
            ],
            weight_decay=self.config.weight_decay,
        )

        def _lr(step: int) -> float:
            cfg = self.config
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(
                1, cfg.total_steps - cfg.warmup_steps,
            )
            cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return max(cfg.min_lr_ratio, cosine)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr)
        self.step_count = 0
        self._temporal_states: list[torch.Tensor | None] | None = None

    # -- required surface --------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def make_synthetic_batch(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        cfg = self.config
        frames = torch.rand(
            batch_size, cfg.n_frames, 3, cfg.resolution, cfg.resolution,
        )
        return {"frames": frames}

    def train_step(self, batch: dict[str, torch.Tensor]) -> StepReport:
        """One HPWM training step with fused DER++ replay + optional EWC.

        Section-B followup restructure: the previous "extra-step
        replay" path took a second ``optimizer.step()`` on the replay
        batch. This version folds the replay task loss and the
        DER++ logits-consistency term into a single backward pass
        alongside the main task loss, then takes one ``optimizer.step``.
        That matches the design-doc default (§4.1 DER++) and halves
        the per-step optimizer call count on replay-active steps.

        HPWM owns a temporal state across steps; we *don't* replay
        through it (replayed clips have no temporal continuity with
        the fresh batch), so the replay forward uses ``None``.
        """
        cfg = self.config
        frames = batch["frames"].to(self._device, non_blocking=True)
        self._model.train()

        outputs = self._model(frames, self._temporal_states)
        loss = outputs["loss"]
        total_loss = loss

        losses: dict[str, float] = {}
        for key in (
            "loss", "prediction_loss", "vqvae_recon_loss", "fwm_loss",
            "commitment_loss", "entropy_loss",
            "slot_consistency_loss", "slot_specialization_loss",
            "slot_diversity_loss",
        ):
            v = outputs.get(key)
            if isinstance(v, torch.Tensor):
                losses[key] = float(v.detach().cpu().item())

        # --- Fused DER++ replay --------------------------------------------
        # Sample replayed clips, run them through the model (within the
        # autograd graph), and add (a) the replay task loss and (b) the
        # DER++ MSE between current ``pred_logits`` and the stored
        # insertion-time logits. Everything contributes to the *same*
        # backward below.
        if self.replay_bank is not None and self.replay_bank.size(self.name) > 0:
            items = self.replay_bank.sample(self.name, k=self.replay_batch_size)
            if items:
                r_frames = torch.stack([it.sample for it in items]).to(self._device)
                r_outputs = self._model(r_frames, None)
                r_loss = r_outputs["loss"]
                total_loss = total_loss + self.replay_weight * r_loss
                losses["replay/loss"] = float(r_loss.detach().cpu().item())

                der_items = [it for it in items if it.stored_logits is not None]
                if der_items:
                    # Re-forward through only the DER-tagged subset and
                    # pull the (still-tracked) ``pred_logits`` from that
                    # forward to MSE against the stored detached logits.
                    rd_frames = torch.stack(
                        [it.sample for it in der_items],
                    ).to(self._device)
                    rd_outputs = self._model(rd_frames, None)
                    rd_logits = rd_outputs["pred_logits"]
                    stored = torch.stack(
                        [it.stored_logits for it in der_items],
                    ).to(self._device)
                    der = nn.functional.mse_loss(rd_logits, stored)
                    total_loss = total_loss + self.der_weight * der
                    losses["der_consistency"] = float(der.detach().cpu().item())

        # --- Backward / Fisher / clip / step / SI -------------------------
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.ewc is not None:
            self.ewc.consolidate(self._model)
            penalty = self.ewc.penalty(self._model)
            penalty.backward()
            losses["ewc/penalty"] = float(penalty.detach().cpu().item())
        grad_norm = nn.utils.clip_grad_norm_(
            self._model.parameters(), cfg.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        if self.ewc is not None:
            self.ewc.post_step(self._model)

        # Detach temporal states for next step (truncated BPTT).
        self._temporal_states = [
            s.detach() if s is not None else None
            for s in outputs.get("temporal_states", [])
        ]

        self.step_count += 1

        # --- buffer insertion --------------------------------------------
        # Store ``pred_logits`` (already detached by the model's forward)
        # as the DER++ teacher signal for each newly-buffered sample.
        if self.replay_bank is not None:
            pred_logits = outputs.get("pred_logits")
            for i in range(frames.shape[0]):
                stored = (
                    pred_logits[i].detach().cpu()
                    if isinstance(pred_logits, torch.Tensor)
                    else None
                )
                self.replay_bank.insert(
                    self.name,
                    frames[i].detach().cpu(),
                    stored_logits=stored,
                    step=self.step_count,
                )

        # --- EMA update --------------------------------------------------
        if self.ema_teacher is not None:
            self.ema_teacher.update(self._model)

        return StepReport(
            loss=losses.get("loss", float(loss.detach().cpu().item())),
            losses=losses,
            grad_norm=float(grad_norm.detach().cpu().item())
            if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        )

    @torch.no_grad()
    def evaluate(self, probe_set: Any | None = None) -> EvalReport:
        """Score on a probe set: prediction loss + VQ-VAE reconstruction.

        These are the two losses ``hpwm.evaluate`` treats as primary
        training signals. The three validation signals from
        ``hpwm/evaluate.py`` (MoD entropy, slot Jaccard, Mamba retention)
        are deferred to a later phase that loads them via the upstream
        ``Evaluator`` class.
        """
        if probe_set is None:
            from scripts.training.self_improve.eval_registry import build_hpwm_probe
            probe_set = build_hpwm_probe(
                n_frames=self.config.n_frames,
                resolution=self.config.resolution,
            )

        was_training = self._model.training
        self._model.eval()
        try:
            pred_sum = 0.0
            recon_sum = 0.0
            n = 0
            for batch in probe_set:
                frames = batch["frames"].to(self._device, non_blocking=True)
                out = self._model(frames, None)  # fresh temporal state per probe batch
                bs = frames.shape[0]
                pred_sum += float(out["prediction_loss"].item()) * bs
                recon_sum += float(out["vqvae_recon_loss"].item()) * bs
                n += bs
        finally:
            self._model.train(was_training)

        pred = pred_sum / max(n, 1)
        recon = recon_sum / max(n, 1)
        return EvalReport(
            score=pred,
            metrics={"prediction_loss": pred, "vqvae_recon_loss": recon},
            higher_is_better=False,
        )

    # -- state -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        # Include the full model state (frozen DINO backbone included) so
        # that snapshot/rollback is bit-faithful. The __init__ above forces
        # DINO to materialize, so these keys are always present on both
        # save and (post-construction) load.
        out: dict[str, Any] = {
            "model": self._model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step_count": self.step_count,
        }
        ema = self._ema_state()
        if ema is not None:
            out["_ema"] = ema
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # Ensure DINO is materialized before load — defensive in case a
        # caller bypassed __init__ (e.g. unpickled a plugin); the call is
        # cheap and idempotent after the first invocation.
        self._model.dino._load_model(self._device)
        self._model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        self.step_count = int(state.get("step_count", 0))
        self._load_ema_state(state.get("_ema"))

    # -- helpers -----------------------------------------------------------

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "HPWMPlugin":
        return cls(replace(_tiny_config(), **overrides))


__all__ = ["HPWMPlugin"]
