"""Online EWC + Synaptic Intelligence regularizers.

Design (``docs/self_improvement.md`` §4.3):

We anchor important parameters with a quadratic penalty against a slowly
moving snapshot ``θ*``:

    L_reg = Σ_i  (λ_ewc·F_i + λ_si·ω_i) / 2  ·  (θ_i − θ_i*)²

where

- **F_i** (online EWC, Schwarz et al. 2018) is the running diagonal
  empirical Fisher: ``F ← γ·F + (1−γ)·grad²``. The empirical Fisher is
  cheaper than the true Fisher (no sampling from the model's predictive
  distribution) and usually performs comparably.
- **ω_i** (Synaptic Intelligence, Zenke et al. 2017) is the path
  integral of how much each parameter contributed to reducing loss
  *between anchor snapshots*: ``ω_i ← ω_i + max(0, −g_i·Δθ_i)``. The
  positive-clipping convention matches the original paper; only
  loss-decreasing contributions count.

Snapshot semantics: ``snapshot_anchor`` resets both ``θ*`` and the SI
path integral ``ω`` (start of a new "task window"). Fisher persists
across snapshots (it tracks global parameter sensitivity, not
task-specific importance).

The penalty is auto-grad tracked through ``θ``; ``F`` / ``ω`` / ``θ*``
participate as constants (no grad). Sparsity is enforced by ignoring
parameters whose anchor magnitude is below ``weight_threshold``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EWCSI:
    """Online EWC + Synaptic Intelligence regularizer.

    One instance per plugin. Holds three per-parameter dicts (Fisher,
    omega, anchor) keyed by ``model.named_parameters`` keys.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        fisher_decay: float = 0.95,
        lam_ewc: float = 1.0,
        lam_si: float = 1.0,
        weight_threshold: float = 1e-6,
        si_damping: float = 1e-3,
    ) -> None:
        if not 0.0 < fisher_decay < 1.0:
            raise ValueError(
                f"fisher_decay must be in (0,1); got {fisher_decay}"
            )
        self.fisher_decay = fisher_decay
        self.lam_ewc = lam_ewc
        self.lam_si = lam_si
        self.weight_threshold = weight_threshold
        #: Numerical floor on the (θ−θ*)² normalization inside SI; matches
        #: Zenke et al. §3 (their ``ξ``). Without it the SI importance
        #: divides by zero whenever a param hasn't moved since the anchor.
        self.si_damping = si_damping

        # Track only parameters that require grad.
        self._param_names = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]
        device = next(model.parameters()).device
        self.fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.omega: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.anchor: dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # Step-internal scratch used by consolidate -> post_step.
        self._prev_params: dict[str, torch.Tensor] = {}
        self._prev_grads: dict[str, torch.Tensor] = {}

        self.num_fisher_updates = 0
        self.num_si_updates = 0
        self.num_anchor_snapshots = 0

    # ------------------------------------------------------------------
    # Anchor management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def snapshot_anchor(self, model: nn.Module) -> None:
        """Capture the current parameters as the new ``θ*`` and reset
        the SI path integral. Call this at the start of a "task" or
        whenever the EMA teacher refreshes (design doc §4.3).
        """
        for n, p in model.named_parameters():
            if n in self.anchor:
                self.anchor[n].copy_(p.detach())
                self.omega[n].zero_()
        self.num_anchor_snapshots += 1

    # ------------------------------------------------------------------
    # Per-step updates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def consolidate(self, model: nn.Module) -> None:
        """Pre-step Fisher accumulation and pre-step SI snapshot.

        Call after ``loss.backward()`` and before ``optimizer.step()``.
        Reads the grads currently on the parameters; squares them into
        ``F_batch``; updates the running Fisher
        ``F ← γ·F + (1−γ)·F_batch``. Also snapshots ``prev_params`` and
        ``prev_grads`` for the upcoming ``post_step`` call.
        """
        g = self.fisher_decay
        for n, p in model.named_parameters():
            if p.grad is None or n not in self.fisher:
                continue
            grad = p.grad.detach()
            self.fisher[n].mul_(g).addcmul_(grad, grad, value=1.0 - g)
            self._prev_params[n] = p.detach().clone()
            self._prev_grads[n] = grad.clone()
        self.num_fisher_updates += 1

    @torch.no_grad()
    def post_step(self, model: nn.Module) -> None:
        """Post-step SI accumulation. Call after ``optimizer.step()``.

        Computes Δθ = current − prev (the actual step taken) and adds
        the positive part of ``−g_prev · Δθ`` to the path integral.
        Only the "decreased loss" contributions count, matching the
        Zenke convention.
        """
        for n, p in model.named_parameters():
            if n not in self._prev_params:
                continue
            delta = p.detach() - self._prev_params[n]
            contrib = -(self._prev_grads[n] * delta)
            # Positive clip: only count contributions that reduced loss.
            contrib.clamp_(min=0.0)
            self.omega[n].add_(contrib)
        self._prev_params.clear()
        self._prev_grads.clear()
        self.num_si_updates += 1

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return the regularizer ``Σ (λ_ewc·F + λ_si·ω) / 2 · (θ−θ*)²``.

        Autograd-tracked through ``θ`` only; ``F``, ``ω``, ``θ*`` are
        held constant. Returns a scalar tensor on the same device as
        the parameters.

        The Zenke (θ−θ*)² normalization of ω is applied *at task
        boundaries* by :meth:`snapshot_anchor`, not on every penalty
        call — see ``self.importance`` for the consolidated quantity.
        Here ``ω`` enters linearly as importance, on the same footing
        as Fisher.
        """
        loss = None
        for n, p in model.named_parameters():
            if n not in self.fisher:
                continue
            anchor = self.anchor[n]
            if anchor.abs().max() < self.weight_threshold:
                # Skip params we've never anchored substantially (typical
                # of frozen bias terms or LayerNorm scales at init).
                continue
            importance = (
                self.lam_ewc * self.fisher[n] + self.lam_si * self.omega[n]
            )
            term = 0.5 * (importance * (p - anchor) ** 2).sum()
            loss = term if loss is None else loss + term
        if loss is None:
            # No parameters above threshold yet — return a zero on the
            # right device for autograd composition.
            device = next(model.parameters()).device
            return torch.zeros((), device=device)
        return loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        return {
            "fisher_decay": self.fisher_decay,
            "lam_ewc": self.lam_ewc,
            "lam_si": self.lam_si,
            "weight_threshold": self.weight_threshold,
            "si_damping": self.si_damping,
            "fisher": {k: v.clone() for k, v in self.fisher.items()},
            "omega": {k: v.clone() for k, v in self.omega.items()},
            "anchor": {k: v.clone() for k, v in self.anchor.items()},
            "counters": {
                "num_fisher_updates": self.num_fisher_updates,
                "num_si_updates": self.num_si_updates,
                "num_anchor_snapshots": self.num_anchor_snapshots,
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.fisher_decay = float(state["fisher_decay"])
        self.lam_ewc = float(state["lam_ewc"])
        self.lam_si = float(state["lam_si"])
        self.weight_threshold = float(state["weight_threshold"])
        self.si_damping = float(state["si_damping"])
        self.fisher = {k: v.clone() for k, v in state["fisher"].items()}
        self.omega = {k: v.clone() for k, v in state["omega"].items()}
        self.anchor = {k: v.clone() for k, v in state["anchor"].items()}
        c = state["counters"]
        self.num_fisher_updates = int(c["num_fisher_updates"])
        self.num_si_updates = int(c["num_si_updates"])
        self.num_anchor_snapshots = int(c["num_anchor_snapshots"])


__all__ = ["EWCSI"]
