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

    def train_step(self, batch: torch.Tensor) -> StepReport:
        losses = self._trainer.step(batch)
        total = losses["total"]
        return StepReport(loss=total, losses=dict(losses))

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
        return self._trainer.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._trainer.load_state_dict(state)

    # -- helpers -----------------------------------------------------------

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "LGQPlugin":
        """Build with a tiny config plus per-field overrides."""
        cfg = replace(_tiny_config(), **overrides)
        return cls(cfg)


__all__ = ["LGQPlugin"]
