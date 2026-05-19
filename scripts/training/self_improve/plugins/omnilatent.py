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

    def train_step(self, batch: dict[str, torch.Tensor]) -> StepReport:
        losses = dict(self._trainer._train_step(batch))

        # --- extra-step replay --------------------------------------------
        if self.replay_bank is not None and self.replay_bank.size(self.name) > 0:
            items = self.replay_bank.sample(self.name, k=self.replay_batch_size)
            if items:
                rx = {"image": torch.stack([it.sample for it in items])}
                replay_losses = self._trainer._train_step(rx)
                for k, v in replay_losses.items():
                    losses[f"replay/{k}"] = v

        # --- buffer insertion (image-only; later phases handle all modalities)
        if self.replay_bank is not None and "image" in batch:
            img = batch["image"]
            for i in range(img.shape[0]):
                self.replay_bank.insert(
                    self.name,
                    img[i].detach().cpu(),
                    stored_logits=None,
                    step=self._trainer.global_step,
                )

        # --- EMA update ----------------------------------------------------
        if self.ema_teacher is not None:
            self.ema_teacher.update(self._trainer.model)

        total = float(losses.get("total", 0.0))
        return StepReport(loss=total, losses={k: float(v) for k, v in losses.items()})

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
        return {
            "model": self._trainer.model.state_dict(),
            "optimizer": self._trainer.optimizer.state_dict(),
            "scaler": self._trainer.scaler.state_dict(),
            "criterion": self._trainer.criterion.state_dict(),
            "global_step": self._trainer.global_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._trainer.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self._trainer.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            self._trainer.scaler.load_state_dict(state["scaler"])
        if "criterion" in state:
            self._trainer.criterion.load_state_dict(state["criterion"])
        self._trainer.global_step = int(state.get("global_step", 0))

    # -- helpers -----------------------------------------------------------

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "OmniLatentPlugin":
        return cls(replace(_tiny_config(), **overrides))


__all__ = ["OmniLatentPlugin"]
