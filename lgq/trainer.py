"""Stateful single-step trainer for LGQ.

``lgq/train.py`` runs an end-to-end training loop. This module exposes
the same per-step logic — generator forward/backward, discriminator
forward/backward, schedulers, mixed precision — as a ``step(batch)``
method so it can be driven externally (e.g. by the self-improvement
orchestrator).

The existing CLI in ``lgq/train.py`` keeps working unchanged; this file
is a parallel entry point, not a replacement.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from lgq.config import LGQConfig
from lgq.losses import LGQLoss
from lgq.model import LGQVAE


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class LGQTrainer:
    """One-step LGQ trainer suitable for external orchestration.

    Differs from ``lgq.train.train`` only in that it does not own a
    dataloader or loop — callers supply each batch via :meth:`step`.

    Parameters
    ----------
    config:
        ``LGQConfig`` controlling model size, loss weights, and scheduler.
    device:
        Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        config: LGQConfig,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = LGQVAE(config).to(self.device)
        self.loss_fn = LGQLoss(config)

        param_groups = self.model.get_param_groups()
        self._param_groups = param_groups
        self.opt_gen = torch.optim.AdamW(
            param_groups["generator"],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.5, 0.9),
        )
        self.opt_disc = torch.optim.AdamW(
            param_groups["discriminator"],
            lr=config.disc_learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.5, 0.9),
        )

        def _lr(step: int) -> float:
            return _cosine_warmup(step, config.warmup_steps, config.total_steps)

        self.sched_gen = torch.optim.lr_scheduler.LambdaLR(self.opt_gen, _lr)
        self.sched_disc = torch.optim.lr_scheduler.LambdaLR(self.opt_disc, _lr)

        self.use_amp = config.precision in ("bf16", "fp16") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        # bf16 does not need a GradScaler; fp16 does.
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler("cuda")
            if (self.use_amp and config.precision == "fp16")
            else None
        )

        self.step_count = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, batch: torch.Tensor) -> dict[str, float]:
        """One generator + (optional) discriminator step on a batch of images.

        Args:
            batch: ``[B, C, H, W]`` images on any device; moved to
                ``self.device`` automatically.

        Returns:
            Flat dict of named loss values; always contains ``"total"``.
        """
        cfg = self.config
        x = batch.to(self.device, non_blocking=True)
        self.model.train()

        # --- Generator step ---
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            model_out = self.model(x)
            adv_loss = None
            if self.step_count >= cfg.disc_start_step:
                adv_loss = self.model.generator_adversarial_loss(model_out["recon"])
            gen_losses = self.loss_fn.generator_loss(x, model_out, adv_loss, self.step_count)

        total = gen_losses["total"]

        self.opt_gen.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.opt_gen)
            nn.utils.clip_grad_norm_(self._param_groups["generator"], cfg.max_grad_norm)
            self.scaler.step(self.opt_gen)
            self.scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(self._param_groups["generator"], cfg.max_grad_norm)
            self.opt_gen.step()
        self.sched_gen.step()

        # --- Discriminator step (after warmup) ---
        disc_loss_val: float | None = None
        if self.step_count >= cfg.disc_start_step:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                disc_out = self.model.forward_discriminator(x, model_out["recon"].detach())
            self.opt_disc.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(disc_out["d_loss"]).backward()
                self.scaler.step(self.opt_disc)
                self.scaler.update()
            else:
                disc_out["d_loss"].backward()
                self.opt_disc.step()
            self.sched_disc.step()
            disc_loss_val = float(disc_out["d_loss"].detach().cpu().item())

        # --- Flatten losses ---
        out: dict[str, float] = {}
        for k, v in gen_losses.items():
            if isinstance(v, torch.Tensor):
                out[k] = float(v.detach().cpu().item())
            elif isinstance(v, (int, float)):
                out[k] = float(v)
        if disc_loss_val is not None:
            out["d_loss"] = disc_loss_val

        self.step_count += 1
        return out

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        state = {
            "step_count": self.step_count,
            "model": self.model.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_disc": self.opt_disc.state_dict(),
            "sched_gen": self.sched_gen.state_dict(),
            "sched_disc": self.sched_disc.state_dict(),
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_count = int(state.get("step_count", 0))
        self.model.load_state_dict(state["model"])
        self.opt_gen.load_state_dict(state["opt_gen"])
        self.opt_disc.load_state_dict(state["opt_disc"])
        if "sched_gen" in state:
            self.sched_gen.load_state_dict(state["sched_gen"])
        if "sched_disc" in state:
            self.sched_disc.load_state_dict(state["sched_disc"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


__all__ = ["LGQTrainer"]
