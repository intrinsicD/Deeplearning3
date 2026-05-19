"""Stateful single-step trainer for the Gaussian autoencoder.

``gaussian_encoder/train.py`` runs a full MNIST training script. This
module exposes the per-step logic as a ``step(batch)`` method so it can
be driven externally (e.g. by the self-improvement orchestrator).

The existing CLI in ``train.py`` keeps working unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .model import GaussianAutoencoder


class GaussianTrainer:
    """One-step trainer wrapping :class:`GaussianAutoencoder`.

    Parameters
    ----------
    in_ch, latent_dim, channels:
        Forwarded to :class:`GaussianAutoencoder`.
    lr:
        Adam learning rate.
    device:
        Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        in_ch: int = 1,
        latent_dim: int = 32,
        channels: tuple[int, ...] = (16, 32),
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = GaussianAutoencoder(
            in_ch=in_ch, latent_dim=latent_dim, channels=channels,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0

    def step(self, batch: torch.Tensor) -> dict[str, float]:
        """One reconstruction step.

        Args:
            batch: ``[B, in_ch, H, W]`` images. Auto-moved to device.

        Returns:
            ``{"total": <mse>, "recon": <mse>}``.
        """
        x = batch.to(self.device, non_blocking=True)
        self.model.train()

        x_hat, _ = self.model(x)
        loss = self.loss_fn(x_hat, x)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        value = float(loss.detach().cpu().item())
        return {"total": value, "recon": value}

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = int(state.get("step_count", 0))
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


__all__ = ["GaussianTrainer"]
