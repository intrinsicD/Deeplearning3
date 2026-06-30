"""Routed model forward: apply router selection to hooks (work plan W3.2).

This is the execution side of wish 3 — *using* the selected pattern. Given an
input, the :class:`~omnilatent.agent.router.LearnedLatentRouter` produces
per-input weights over the expert registry; the hook experts' weights are
applied to the model's :class:`NeuralPortManager` as content-conditioned gates
(W3.1), so the top-k selected hooks **co-activate and compose in attention**
during a single forward. The effective gates are recorded into an
:class:`AgentTraceStep` for observability.

The controller is duck-typed (it needs ``encode``, ``hook_manager`` and to be
callable as ``model(src, data, tgt, tgt_data)``) to avoid an agent→model import
cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.protocol import AgentTraceStep


@dataclass
class RoutedForward:
    """Drive a model forward with router-selected, content-conditioned hooks."""

    model: Any
    router: LearnedLatentRouter
    hook_prefix: str = "hook:"

    @property
    def registry(self) -> ExpertRegistry:
        return self.router.registry

    def _summarize(self, source_modality: str, source_data: torch.Tensor) -> torch.Tensor:
        """Mean-pool the encoded source (skip the modality indicator token)."""
        enc = self.model.encode(source_modality, source_data)
        return enc[:, 1:].mean(dim=1)  # (B, D)

    def _hook_route_weights(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map per-expert routing weights → {hook_name: (B,) weight} for live hooks."""
        ids = self.registry.ids()
        rw: dict[str, torch.Tensor] = {}
        live = self.model.hook_manager.hooks
        for i, eid in enumerate(ids):
            if self.registry.kind(eid) != "hook" or not eid.startswith(self.hook_prefix):
                continue
            hook_name = eid[len(self.hook_prefix):]
            if hook_name in live:
                rw[hook_name] = weights[:, i]
        return rw

    def route_and_forward(
        self,
        source_modality: str,
        source_data: torch.Tensor,
        target_modality: str,
        target_data: torch.Tensor | None = None,
    ) -> tuple[dict, AgentTraceStep]:
        """Route the input, activate the selected hooks, and run the forward.

        Returns the model result dict and an :class:`AgentTraceStep` recording
        the effective hook gates and the routing weights.
        """
        summary = self._summarize(source_modality, source_data)
        routed = self.router.forward(summary)
        weights = routed["weights"]                      # (B, E)
        rw = self._hook_route_weights(weights)

        manager = self.model.hook_manager
        manager.set_route_weights(rw)
        try:
            result = self.model(source_modality, source_data, target_modality, target_data)
            gate_log = dict(manager.gate_log())
        finally:
            manager.set_route_weights(None)              # never leak weights to the next call

        ids = self.registry.ids()
        expert_weights = {ids[i]: float(weights[:, i].mean().item()) for i in range(len(ids))}
        # ``rw`` carries every hook column (zero-weight ones must be present so
        # they are skipped rather than defaulting to weight 1.0); report only
        # the hooks that actually fired.
        active = sorted(h for h, w in rw.items() if bool(torch.any(w != 0)))
        step = AgentTraceStep(
            node_type="DECODE",
            selected_action="DECODE",
            hook_gates=gate_log,
            metadata={
                "expert_weights": expert_weights,
                "confidence": float(routed["confidence"].mean().item()),
                "active_hooks": active,
            },
        )
        return result, step


__all__ = ["RoutedForward"]
