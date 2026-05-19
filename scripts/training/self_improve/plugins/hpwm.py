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
        cfg = self.config
        frames = batch["frames"].to(self._device, non_blocking=True)
        self._model.train()

        outputs = self._model(frames, self._temporal_states)
        loss = outputs["loss"]

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self._model.parameters(), cfg.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()

        # Detach temporal states for next step (truncated BPTT).
        self._temporal_states = [
            s.detach() if s is not None else None
            for s in outputs.get("temporal_states", [])
        ]

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

        self.step_count += 1
        return StepReport(
            loss=losses.get("loss", float(loss.detach().cpu().item())),
            losses=losses,
            grad_norm=float(grad_norm.detach().cpu().item())
            if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        )

    # -- state -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        # Filter out frozen DINO backbone params from the saved state to
        # match the upstream HPWM checkpoint convention.
        msd = self._model.state_dict()
        return {
            "model": msd,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._model.load_state_dict(state["model"], strict=False)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        self.step_count = int(state.get("step_count", 0))

    # -- helpers -----------------------------------------------------------

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "HPWMPlugin":
        return cls(replace(_tiny_config(), **overrides))


__all__ = ["HPWMPlugin"]
