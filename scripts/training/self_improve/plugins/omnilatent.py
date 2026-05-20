"""Plugin wrapping :class:`omnilatent.training.trainer.Trainer`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.trainer import Trainer as _OmniTrainer

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)


def _tiny_config() -> OmniLatentConfig:
    """Small OmniLatentConfig for smoke tests and dry-run.

    Keeps the backbone tiny (64-dim, 2 layers, 2 heads), shrinks every
    modality, and disables the reasoning bottleneck.
    """
    cfg = OmniLatentConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        max_seq_len=128,
        gradient_checkpointing=False,
        vocab_size=256,
        text_max_len=16,
        audio_n_mels=16,
        audio_max_frames=8,
        audio_patch_frames=4,     # must match AudioEncoder conv stride (=4)
        image_size=32,
        image_patch_size=8,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        batch_size=2,
        learning_rate=1e-4,
        warmup_steps=10,
        max_steps=1000,
        mixed_precision=False,
        contrastive_weight=0.0,
    )
    return cfg


class OmniLatentPlugin(ComponentPlugin):
    name = "omnilatent"

    def __init__(
        self,
        config: OmniLatentConfig | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config if config is not None else _tiny_config()
        model = OmniLatentModel(self.config)

        # Trainer requires a DataLoader but we drive batches externally.
        # Pass an empty in-memory loader; .train() is never called.
        from torch.utils.data import DataLoader, TensorDataset

        dummy = DataLoader(TensorDataset(torch.empty(0)), batch_size=1)
        self._trainer = _OmniTrainer(model, self.config, dummy)
        if device is not None:
            self._trainer.device = device
            self._trainer.model.to(device)

    # -- required surface --------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    @property
    def device(self) -> torch.device:
        return self._trainer.device

    def make_synthetic_batch(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        cfg = self.config
        # Image-only batch: avoids the cost of building text+audio+video
        # for a smoke test. The TaskSampler will pick image→image.
        return {
            "image": torch.rand(batch_size, cfg.image_channels, cfg.image_size, cfg.image_size),
        }

    # ------------------------------------------------------------------
    # Phase 4 helpers
    # ------------------------------------------------------------------

    def _task_forward(
        self, batch: dict[str, torch.Tensor], *, src: str | None = None, tgt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Forward + criterion. Returns (loss, target_output, target_modality).

        ``target_output`` is autograd-tracked and is what gets compared
        against stored DER++ teacher logits on replay. When ``src`` /
        ``tgt`` are not specified, the task sampler picks them (this is
        what fresh-batch training uses); replay forwards pass them
        explicitly so the buffer's stored ``image`` outputs match the
        target modality every time.

        Builds the per-modality contrastive latent bundle when the
        batch has ≥ 2 modalities and ``contrastive_weight > 0`` (the
        same gate the upstream trainer uses).
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        available = list(batch.keys())
        if src is None or tgt is None:
            src, tgt = self._trainer.task_sampler.sample(available)

        model = self._trainer.model
        latents: dict[str, torch.Tensor] | None = None
        if len(available) >= 2 and self.config.contrastive_weight > 0:
            latents = {}
            for mod in available:
                enc = model.encode(mod, batch[mod])
                # Mean-pool content tokens, skipping the modality
                # indicator at position 0 (matches upstream).
                latents[mod] = enc[:, 1:].mean(dim=1)

        result = model(
            source_modality=src,
            source_data=batch[src],
            target_modality=tgt,
            target_data=batch[tgt],
        )
        loss_dict = self._trainer.criterion(
            {tgt: result["output"]},
            {tgt: batch[tgt]},
            latents,
            reasoning_bottleneck=result.get("reasoning_bottleneck"),
            source_summary=result.get("source_summary"),
        )
        return loss_dict["total"], result["output"], tgt

    def _task_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Backwards-compat wrapper for callers that only want the loss."""
        loss, _, _ = self._task_forward(batch)
        return loss

    def train_step(self, batch: dict[str, torch.Tensor]) -> StepReport:
        """One training step with fused DER++ replay + optional EMA + EWC/SI.

        Section-B followup restructure: the previous "extra-step
        replay" path took a second ``optimizer.step()`` on the replay
        batch. This version folds the replay task loss and the
        DER++ MSE-consistency term into a single backward pass
        alongside the main task loss. Total loss layout::

            total = task + replay_weight·replay_task
                       + der_weight·MSE(current_recon, stored_recon)
                       + (EWC penalty added by a second backward)

        Buffer storage: the model's image-target output from the
        current step is stored as the DER teacher signal — the replay
        forward picks ``image→image`` explicitly so the comparison is
        well-defined modality-by-modality.
        """
        model = self._trainer.model
        optimizer = self._trainer.optimizer

        model.train()
        optimizer.zero_grad(set_to_none=True)

        task_loss, task_output, task_tgt = self._task_forward(batch)
        losses: dict[str, float] = {"task": float(task_loss.detach().item())}
        total_loss = task_loss

        # --- Fused DER++ replay -------------------------------------------
        if self.replay_bank is not None and self.replay_bank.size(self.name) > 0:
            items = self.replay_bank.sample(self.name, k=self.replay_batch_size)
            if items:
                rx = {"image": torch.stack([it.sample for it in items]).to(self.device)}
                r_loss, r_output, _ = self._task_forward(
                    rx, src="image", tgt="image",
                )
                total_loss = total_loss + self.replay_weight * r_loss
                losses["replay/task"] = float(r_loss.detach().item())

                der_items = [it for it in items if it.stored_logits is not None]
                if der_items:
                    # Re-forward through only the DER-tagged subset so
                    # the MSE has the right batch dim.
                    rd_inputs = torch.stack(
                        [it.sample for it in der_items],
                    ).to(self.device)
                    _, rd_output, _ = self._task_forward(
                        {"image": rd_inputs}, src="image", tgt="image",
                    )
                    stored = torch.stack(
                        [it.stored_logits for it in der_items],
                    ).to(self.device)
                    der = nn.functional.mse_loss(rd_output, stored)
                    total_loss = total_loss + self.der_weight * der
                    losses["der_consistency"] = float(der.detach().item())

        losses["total"] = float(total_loss.detach().item())
        total_loss.backward()
        if self.ewc is not None:
            # Fisher snapshot uses task grads (well: task + replay) only —
            # call before penalty so the regularizer's gradient doesn't
            # contaminate Fisher.
            self.ewc.consolidate(model)
            penalty = self.ewc.penalty(model)
            penalty.backward()
            losses["ewc/penalty"] = float(penalty.detach().item())

        optimizer.step()
        if self.ewc is not None:
            self.ewc.post_step(model)

        self._trainer.global_step += 1

        # --- buffer insertion (image-only; later phases handle all modalities)
        if self.replay_bank is not None and "image" in batch:
            img = batch["image"]
            # If the task forward happened to pick image→image, the
            # produced output is already on hand; otherwise we run an
            # extra image→image forward purely for the teacher signal.
            if task_tgt == "image":
                stored_recon = task_output.detach()
            else:
                with torch.no_grad():
                    _, stored_recon, _ = self._task_forward(
                        {"image": img.to(self.device)},
                        src="image", tgt="image",
                    )
                stored_recon = stored_recon.detach()
            for i in range(img.shape[0]):
                self.replay_bank.insert(
                    self.name,
                    img[i].detach().cpu(),
                    stored_logits=stored_recon[i].cpu(),
                    step=self._trainer.global_step,
                )

        # --- EMA update ----------------------------------------------------
        if self.ema_teacher is not None:
            self.ema_teacher.update(model)

        return StepReport(loss=losses["total"], losses=losses)

    @torch.no_grad()
    def evaluate(self, probe_set: Any | None = None) -> EvalReport:
        """Score on a probe set: image-reconstruction MSE + L1.

        Phase 2 only exercises the image→image autoencoder path; later
        phases expand to all 16 modality pairs per the design doc §6.1.
        """
        if probe_set is None:
            from scripts.training.self_improve.eval_registry import (
                build_omnilatent_probe,
            )
            probe_set = build_omnilatent_probe(
                image_size=self.config.image_size,
                image_channels=self.config.image_channels,
            )

        model = self._trainer.model
        was_training = model.training
        model.eval()
        try:
            mse_sum = 0.0
            l1_sum = 0.0
            n = 0
            for batch in probe_set:
                image = batch["image"].to(self.device, non_blocking=True)
                result = model.reconstruct("image", image)
                recon = result["output"]
                mse_sum += float(torch.nn.functional.mse_loss(recon, image).item()) * image.shape[0]
                l1_sum += float(torch.nn.functional.l1_loss(recon, image).item()) * image.shape[0]
                n += image.shape[0]
        finally:
            model.train(was_training)

        mse = mse_sum / max(n, 1)
        l1 = l1_sum / max(n, 1)
        return EvalReport(
            score=mse,
            metrics={"image_mse": mse, "image_l1": l1},
            higher_is_better=False,
        )

    # -- state -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self._trainer.model.state_dict(),
            "optimizer": self._trainer.optimizer.state_dict(),
            "scaler": self._trainer.scaler.state_dict(),
            "criterion": self._trainer.criterion.state_dict(),
            "global_step": self._trainer.global_step,
        }
        ema = self._ema_state()
        if ema is not None:
            out["_ema"] = ema
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._trainer.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self._trainer.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            self._trainer.scaler.load_state_dict(state["scaler"])
        if "criterion" in state:
            self._trainer.criterion.load_state_dict(state["criterion"])
        self._trainer.global_step = int(state.get("global_step", 0))
        self._load_ema_state(state.get("_ema"))

    # -- helpers -----------------------------------------------------------

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "OmniLatentPlugin":
        return cls(replace(_tiny_config(), **overrides))


__all__ = ["OmniLatentPlugin"]
