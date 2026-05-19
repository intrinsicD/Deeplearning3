"""Exponential-moving-average teacher for self-distillation.

Design (``docs/self_improvement.md`` §4.2):

- Maintain a slow-moving copy of every parameter:
  ``θ_ema ← d · θ_ema + (1-d) · θ_student`` with default decay ``d=0.999``.
- The teacher serves three purposes:
  1. **Distillation target** during training: add
     ``λ_kd · L(student_out, teacher_out)`` where ``L`` is MSE or KL
     depending on the output type.
  2. **Rollback target** (§4.6): EMA weights are typically more robust
     than student weights at any given step.
  3. **Export checkpoint** (§4.6): we publish EMA weights, not raw
     weights — empirically better generalization across BYOL/MoCo/DINO.

This module owns only the EMA *state*; the loss formula is the plugin's
job (different components have different output types).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMATeacher:
    """Slow-moving copy of a model's parameters and buffers.

    Parameter math:
        ``θ_ema ← d · θ_ema + (1-d) · θ_student``

    Buffer math: copied exactly. EMA-averaging BatchNorm running stats
    has been shown to *hurt* in self-supervised regimes (MoCo paper,
    appendix B), so we mirror PyTorch's official ``swa_utils`` choice
    and treat buffers as direct copies.

    Initialization is bit-exact to the student at construction time.
    """

    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0,1); got {decay}")
        self.decay = decay
        self.num_updates = 0
        # CPU shadow would save GPU memory but blow up update cost; keep
        # on the model's device. Callers can move with ``.to(device)``.
        self._params: dict[str, torch.Tensor] = {
            k: p.detach().clone() for k, p in model.named_parameters()
        }
        self._buffers: dict[str, torch.Tensor] = {
            k: b.detach().clone() for k, b in model.named_buffers()
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """One EMA step against ``model``'s current parameters.

        Parameter shapes are assumed unchanged since construction; if the
        student grows new parameters (phase 8 hook expansion) the EMA
        must be reinitialized. We raise rather than silently dropping
        keys so that bug is loud.
        """
        d = self.decay
        for k, p in model.named_parameters():
            ema = self._params.get(k)
            if ema is None:
                raise KeyError(
                    f"EMA has no entry for parameter {k!r}; reinitialize "
                    "after architectural changes"
                )
            if ema.shape != p.shape:
                raise ValueError(
                    f"EMA shape mismatch for {k!r}: ema={tuple(ema.shape)} "
                    f"student={tuple(p.shape)}"
                )
            ema.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for k, b in model.named_buffers():
            buf = self._buffers.get(k)
            if buf is None:
                # Buffers can come and go (lazy module materialization,
                # e.g. HPWM's DINO backbone); track new ones silently.
                self._buffers[k] = b.detach().clone()
            else:
                buf.copy_(b.detach())
        self.num_updates += 1

    # ------------------------------------------------------------------
    # Inference with EMA weights
    # ------------------------------------------------------------------

    @contextmanager
    def swap_into(self, model: nn.Module) -> Iterator[None]:
        """Temporarily replace ``model``'s weights with EMA weights.

        Restores the originals on exit, even if the wrapped block
        raises. Use this around the teacher's forward pass in a distill
        loss; the alternative — keeping a full duplicate ``nn.Module`` —
        doubles peak memory.
        """
        backup_params: dict[str, torch.Tensor] = {}
        backup_buffers: dict[str, torch.Tensor] = {}
        try:
            for k, p in model.named_parameters():
                backup_params[k] = p.detach().clone()
                p.data.copy_(self._params[k])
            for k, b in model.named_buffers():
                if k in self._buffers:
                    backup_buffers[k] = b.detach().clone()
                    b.data.copy_(self._buffers[k])
            yield
        finally:
            for k, p in model.named_parameters():
                if k in backup_params:
                    p.data.copy_(backup_params[k])
            for k, b in model.named_buffers():
                if k in backup_buffers:
                    b.data.copy_(backup_buffers[k])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "params": {k: v.detach().clone() for k, v in self._params.items()},
            "buffers": {k: v.detach().clone() for k, v in self._buffers.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self._params = {k: v.clone() for k, v in state["params"].items()}
        self._buffers = {k: v.clone() for k, v in state["buffers"].items()}

    def to(self, device: torch.device | str) -> "EMATeacher":
        device = torch.device(device)
        self._params = {k: v.to(device) for k, v in self._params.items()}
        self._buffers = {k: v.to(device) for k, v in self._buffers.items()}
        return self


__all__ = ["EMATeacher"]
