"""Plugin wrapping :class:`gaussian_encoder.trainer.GaussianTrainer`."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gaussian_encoder.trainer import GaussianTrainer

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    StepReport,
)


class GaussianEncoderPlugin(ComponentPlugin):
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
        losses = self._trainer.step(batch)
        total = losses["total"]
        return StepReport(loss=total, losses=dict(losses))

    # -- state -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._trainer.load_state_dict(state)


__all__ = ["GaussianEncoderPlugin"]
