"""Plugin wrapping :class:`gaussian_encoder.trainer.GaussianTrainer`."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gaussian_encoder.trainer import GaussianTrainer

from scripts.training.self_improve.edges import (
    PluginPseudoLabelConsumer,
    apply_pending_labels,
)
from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)


class GaussianEncoderPlugin(ComponentPlugin, PluginPseudoLabelConsumer):
    name = "gaussian_encoder"

    def __init__(
        self,
        *,
        in_ch: int = 1,
        latent_dim: int = 16,
        channels: tuple[int, ...] = (8, 16),
        lr: float = 1e-3,
        image_size: int = 28,
        device: torch.device | None = None,
    ) -> None:
        self._trainer = GaussianTrainer(
            in_ch=in_ch,
            latent_dim=latent_dim,
            channels=channels,
            lr=lr,
            device=device,
        )
        self.in_ch = in_ch
        self.image_size = image_size

    # -- required surface --------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    @property
    def device(self) -> torch.device:
        return self._trainer.device

    def make_synthetic_batch(self, batch_size: int = 2) -> torch.Tensor:
        return torch.rand(batch_size, self.in_ch, self.image_size, self.image_size)

    def train_step(self, batch: torch.Tensor) -> StepReport:
        """One optimizer step with optional DER++ replay and EMA distillation.

        Without attached replay/EMA this is bit-identical to
        :meth:`GaussianTrainer.step`. With them, we:

        1. Run the standard forward + task loss on ``batch``.
        2. If ``replay_bank`` is attached and non-empty: sample ``k``
           items, recompute task loss on them, and add a DER++ MSE term
           between the current recon and the stored insertion-time recon.
        3. If ``ema_teacher`` is attached: add MSE between the student's
           recon and the EMA teacher's recon on the same ``batch``.
        4. One ``backward``/``optimizer.step``.
        5. Insert ``batch`` back into the replay bank (storing the recon
           as the DER++ teacher signal) and update the EMA.
        """
        model = self._trainer.model
        optimizer = self._trainer.optimizer
        device = self._trainer.device

        x = batch.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        x_hat, _ = model(x)
        loss_recon = nn.functional.mse_loss(x_hat, x)

        losses: dict[str, float] = {"recon": float(loss_recon.detach().item())}
        total_loss = loss_recon

        # --- DER++ replay ---------------------------------------------------
        if self.replay_bank is not None and self.replay_bank.size(self.name) > 0:
            items = self.replay_bank.sample(self.name, k=self.replay_batch_size)
            if items:
                rx = torch.stack([it.sample for it in items]).to(device)
                rx_hat, _ = model(rx)
                replay_task = nn.functional.mse_loss(rx_hat, rx)
                total_loss = total_loss + self.replay_weight * replay_task
                losses["replay_task"] = float(replay_task.detach().item())

                # DER++ consistency: only over items that carry stored
                # logits (insertion-time recons). Skip otherwise.
                der_items = [it for it in items if it.stored_logits is not None]
                if der_items:
                    rx_d = torch.stack([it.sample for it in der_items]).to(device)
                    stored = torch.stack(
                        [it.stored_logits for it in der_items]
                    ).to(device)
                    rx_d_hat, _ = model(rx_d)
                    der = nn.functional.mse_loss(rx_d_hat, stored)
                    total_loss = total_loss + self.der_weight * der
                    losses["der_consistency"] = float(der.detach().item())

        # --- EMA distillation ----------------------------------------------
        # The student forward (``x_hat`` above) was computed before any
        # swap, so its autograd graph references the original student
        # parameters. ``swap_into`` mutates ``.data`` in-place but
        # restores it on exit; the captured graph then backprops against
        # the restored values. We only need the teacher's *forward*.
        if self.ema_teacher is not None:
            with self.ema_teacher.swap_into(model):
                with torch.no_grad():
                    teacher_hat, _ = model(x)
            distill = nn.functional.mse_loss(x_hat, teacher_hat)
            total_loss = total_loss + self.distill_weight * distill
            losses["distill"] = float(distill.detach().item())

        # --- Cross-component pseudo-labels (§5) ----------------------------
        # The orchestrator drops a list of :class:`PseudoLabelBatch` onto
        # the plugin's ``_pending_pseudo_labels`` slot before this call.
        # ``apply_pending_labels`` consumes them and returns a scalar
        # consistency loss (zero if none queued) plus per-producer log
        # entries.
        pseudo_loss, pseudo_log = apply_pending_labels(self, x_hat, x)
        if pseudo_log:
            total_loss = total_loss + self.pseudo_weight * pseudo_loss
            losses.update(pseudo_log)
            losses["pseudo/total"] = float(pseudo_loss.detach().item())

        # --- Backward / Fisher / step / SI --------------------------------
        # EWC's Fisher accumulation needs the *task* gradient only, so
        # we backward the task-side loss first and call ``consolidate``
        # before adding the regularizer's gradient on top via a second
        # ``backward`` on the penalty term. The optimizer.step then sees
        # the full ∇(task + λ·penalty).
        total_loss.backward()
        if self.ewc is not None:
            self.ewc.consolidate(model)
            penalty = self.ewc.penalty(model)
            penalty.backward()
            losses["ewc/penalty"] = float(penalty.detach().item())
            total_loss = total_loss + penalty  # only used for the .total report

        optimizer.step()
        if self.ewc is not None:
            self.ewc.post_step(model)
        self._trainer.step_count += 1

        losses["total"] = float(total_loss.detach().item())

        # --- post-step bookkeeping -----------------------------------------
        if self.replay_bank is not None:
            with torch.no_grad():
                stored_hat, _ = model(x)
            for i in range(x.shape[0]):
                self.replay_bank.insert(
                    self.name,
                    x[i].detach().cpu(),
                    stored_logits=stored_hat[i].detach().cpu(),
                    step=self._trainer.step_count,
                )

        if self.ema_teacher is not None:
            self.ema_teacher.update(model)

        return StepReport(loss=losses["total"], losses=losses)

    @torch.no_grad()
    def evaluate(self, probe_set: Any | None = None) -> EvalReport:
        """Score on a probe set: reconstruction MSE + L1 (uncorrelated enough
        to detect mode-collapse style reward hacking).
        """
        if probe_set is None:
            from scripts.training.self_improve.eval_registry import (
                build_gaussian_encoder_probe,
            )
            probe_set = build_gaussian_encoder_probe(
                in_ch=self.in_ch,
                image_size=self.image_size,
            )

        model = self._trainer.model
        was_training = model.training
        model.eval()
        try:
            mse_sum = 0.0
            l1_sum = 0.0
            n = 0
            for batch in probe_set:
                x = batch.to(self.device, non_blocking=True)
                x_hat, _ = model(x)
                mse_sum += float(torch.nn.functional.mse_loss(x_hat, x).item()) * x.shape[0]
                l1_sum += float(torch.nn.functional.l1_loss(x_hat, x).item()) * x.shape[0]
                n += x.shape[0]
        finally:
            model.train(was_training)

        mse = mse_sum / max(n, 1)
        l1 = l1_sum / max(n, 1)
        return EvalReport(
            score=mse,
            metrics={"mse": mse, "l1": l1},
            higher_is_better=False,
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
        # Strip EMA/EWC before delegating; the underlying GaussianTrainer
        # doesn't know those keys and would mishandle them.
        trainer_state = {
            k: v for k, v in state.items() if k not in ("_ema", "_ewc")
        }
        self._trainer.load_state_dict(trainer_state)
        self._load_ema_state(state.get("_ema"))
        self._load_ewc_state(state.get("_ewc"))


__all__ = ["GaussianEncoderPlugin"]
