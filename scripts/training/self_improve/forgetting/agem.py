"""Averaged Gradient Episodic Memory (A-GEM).

Design (``docs/self_improvement.md`` §4.5). When the current-batch
gradient ``g`` would *increase* loss on the replay buffer (``g·g_ref < 0``),
project it onto the orthogonal complement of ``g_ref``:

    g' = g − (g·g_ref / ‖g_ref‖²) · g_ref     if g·g_ref < 0

This guarantees that the projected step cannot harm replay-buffer loss.
A-GEM (Chaudhry et al. 2019) is the cheap variant of GEM: one extra
forward+backward on a small replay batch per step, no quadratic
program.

Off by default. The DER++ replay + EWC + EMA stack from earlier phases
is usually sufficient; A-GEM is the per-component escape hatch when we
see persistent regression on a specific plugin.

This module ships only the gradient-projection math; plugins wire it in
themselves (the replay-loss computation is plugin-specific).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn


class AGEMProjector:
    """Project task grads onto the orthogonal complement of replay grads.

    Usage inside a plugin's ``train_step``::

        loss = self._task_loss(batch)
        loss.backward()                               # populates task grads
        projector.project(self.model, replay_loss_fn) # projects in-place
        optimizer.step()

    ``replay_loss_fn`` is a no-argument callable that returns a scalar
    replay loss with autograd attached. The projector handles its own
    backward pass for the replay loss; the *caller* should not call
    ``backward()`` on it again afterward.
    """

    def __init__(self) -> None:
        self.last_dot_product: float | None = None
        self.last_was_projected: bool = False
        self.num_projections: int = 0
        self.num_calls: int = 0

    @torch.no_grad()
    def _grad_dot(
        self, gs1: Iterable[torch.Tensor], gs2: Iterable[torch.Tensor],
    ) -> float:
        total = 0.0
        for a, b in zip(gs1, gs2):
            total += float((a * b).sum().item())
        return total

    @torch.no_grad()
    def _grad_sq_norm(self, gs: Iterable[torch.Tensor]) -> float:
        total = 0.0
        for g in gs:
            total += float((g * g).sum().item())
        return total

    def project(
        self,
        model: nn.Module,
        replay_loss_fn: Callable[[], torch.Tensor],
    ) -> None:
        """In-place A-GEM projection of the task grads currently on ``model``.

        Procedure:

        1. Capture task grads ``g`` (assumed already populated by an
           earlier ``loss.backward()`` on the task loss).
        2. Zero grads, call ``replay_loss_fn().backward()``, capture
           reference grads ``g_ref``.
        3. If ``g·g_ref < 0``, project ``g`` onto the orthogonal
           complement of ``g_ref`` and write the projected grads back
           onto the parameters. Otherwise restore the original task
           grads.

        After this returns, the params hold the (possibly projected)
        task grads — ready for ``optimizer.step()``.
        """
        self.num_calls += 1

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return

        # 1. Snapshot task grads.
        task_grads: list[torch.Tensor | None] = []
        for p in params:
            task_grads.append(p.grad.detach().clone() if p.grad is not None else None)

        # 2. Replay grads. The replay loss owns its own forward+backward.
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        replay_loss = replay_loss_fn()
        replay_loss.backward()
        replay_grads: list[torch.Tensor | None] = []
        for p in params:
            replay_grads.append(p.grad.detach().clone() if p.grad is not None else None)

        # Only parameters that have grads from BOTH passes participate
        # in the projection; mismatched ones get the original task grad
        # restored (no-op).
        both = [(t, r) for t, r in zip(task_grads, replay_grads) if t is not None and r is not None]
        if not both:
            # Fall back: restore task grads if any.
            for p, t in zip(params, task_grads):
                p.grad = t
            return

        t_seq = [t for t, _ in both]
        r_seq = [r for _, r in both]
        dot = self._grad_dot(t_seq, r_seq)
        self.last_dot_product = dot

        # 3. Project if and only if the dot product is negative.
        if dot < 0:
            ref_sq = self._grad_sq_norm(r_seq)
            if ref_sq > 0:
                coef = dot / ref_sq
                projected: list[torch.Tensor] = []
                for t, r in both:
                    projected.append(t - coef * r)
                # Write back, matching the params order.
                it = iter(projected)
                for p, t, r in zip(params, task_grads, replay_grads):
                    if t is None or r is None:
                        p.grad = t
                    else:
                        p.grad = next(it)
                self.num_projections += 1
                self.last_was_projected = True
                return

        # No projection needed: restore the task grads (we overwrote
        # them with replay grads while computing g_ref).
        for p, t in zip(params, task_grads):
            p.grad = t
        self.last_was_projected = False


__all__ = ["AGEMProjector"]
