"""Router interfaces for explicit agent graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import torch
import torch.nn as nn

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.protocol import AgentTrace, LatentPacket, ToolContext


@dataclass(frozen=True)
class RouteDecision:
    """A deterministic route decision emitted by a router."""

    action: str
    stop: bool = False
    confidence: float = 1.0
    stop_prob: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("RouteDecision.action must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("RouteDecision.confidence must be in [0, 1]")
        if self.stop_prob is not None and not 0.0 <= self.stop_prob <= 1.0:
            raise ValueError("RouteDecision.stop_prob must be in [0, 1]")


class BaseRouter(Protocol):
    """Protocol for routers consumed by :class:`AgentRuntime`."""

    def route(self, packet: LatentPacket, context: ToolContext, trace: AgentTrace) -> RouteDecision:
        raise NotImplementedError


class StaticRouter:
    """Deterministic router backed by a finite sequence of decisions.

    This is primarily useful for tests, synthetic traces, and scripted agent
    flows. When decisions are exhausted, the router emits a stop decision unless
    ``repeat_last`` is enabled.
    """

    def __init__(self, decisions: Sequence[RouteDecision | str], *, repeat_last: bool = False) -> None:
        if not decisions:
            decisions = [RouteDecision("STOP", stop=True, stop_prob=1.0)]
        self.decisions = [d if isinstance(d, RouteDecision) else RouteDecision(d) for d in decisions]
        self.repeat_last = repeat_last
        self.index = 0

    def route(self, packet: LatentPacket, context: ToolContext, trace: AgentTrace) -> RouteDecision:
        if self.index < len(self.decisions):
            decision = self.decisions[self.index]
            self.index += 1
            return decision
        if self.repeat_last:
            return self.decisions[-1]
        return RouteDecision("STOP", stop=True, stop_prob=1.0, metadata={"reason": "router_exhausted"})

    def reset(self) -> None:
        self.index = 0


#: How each expert kind maps to an explicit agent-graph action. Hook experts
#: are applied *inside* the backbone forward (content-conditioned gating, W3.1)
#: rather than as a graph node, so selecting a hook routes to DECODE with the
#: hook weights carried in metadata.
DEFAULT_ACTION_MAP: dict[str, str] = {
    "tool": "TOOL_CALL",
    "kb": "KB_READ",
    "hook": "DECODE",
}


class LearnedLatentRouter(nn.Module):
    """Input-conditioned router over an :class:`ExpertRegistry` (work plan W2.2).

    Given a pooled summary of a packet's latent state, score it against every
    registered expert key and keep the top-k as a sparse, renormalized weight
    distribution (Switch/MoE-style). This is the mechanism for wish 2 —
    "identify the right pattern" — selecting *which* learned structure (hook,
    tool, or KB query) is relevant to the current input.

    Two entry points:

    * :meth:`forward` returns differentiable per-batch routing weights
      ``(B, num_experts)`` — consumed by content-conditioned hook gating
      (W3.1) and trained end-to-end (W3.3).
    * :meth:`route` pools to a single query and returns a :class:`RouteDecision`
      so the router is a drop-in for :class:`StaticRouter` in the agent runtime.

    Args:
        registry: the experts to route among.
        input_dim: feature dim of the pooled latent summary (model hidden dim).
        top_k: number of experts kept active per input.
        temperature: softmax temperature over the top-k logits.
        action_map: expert-kind → graph-action mapping.
        fallback_action: action emitted when the registry is empty.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        input_dim: int,
        top_k: int = 2,
        temperature: float = 1.0,
        action_map: Mapping[str, str] | None = None,
        fallback_action: str = "DECODE",
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.registry = registry
        self.top_k = top_k
        self.temperature = float(temperature)
        self.action_map = dict(action_map) if action_map is not None else dict(DEFAULT_ACTION_MAP)
        self.fallback_action = fallback_action
        self.query_proj = nn.Linear(input_dim, registry.key_dim)

    # -- differentiable routing -----------------------------------------
    def forward(self, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        """Route a batch of pooled summaries.

        Args:
            summary: ``(B, input_dim)`` pooled latent summaries.

        Returns a dict with:
            ``weights``: ``(B, E)`` sparse top-k routing weights (rows sum to 1
                over the active experts; 0 elsewhere). Empty registry → ``(B, 0)``.
            ``logits``:  ``(B, E)`` raw scores.
            ``confidence``: ``(B,)`` peakiness of the full softmax (max prob);
                low ⇒ the router has no clearly-relevant expert (abstain, W2.3).
        """
        if summary.dim() != 2:
            raise ValueError(f"summary must be (B, input_dim); got {tuple(summary.shape)}")
        b = summary.shape[0]
        n = len(self.registry)
        if n == 0:
            empty = summary.new_zeros(b, 0)
            return {"weights": empty, "logits": empty, "confidence": summary.new_zeros(b)}

        keys = self.registry.keys().to(summary.dtype).to(summary.device)  # (E, key_dim)
        query = self.query_proj(summary)                                  # (B, key_dim)
        logits = query @ keys.t() / self.temperature                      # (B, E)

        k = min(self.top_k, n)
        topk_vals, topk_idx = logits.topk(k, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, topk_idx, topk_w)

        # Calibration signal: how peaked is the *full* distribution.
        confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
        return {"weights": weights, "logits": logits, "confidence": confidence}

    @staticmethod
    def _summarize(packet: LatentPacket) -> torch.Tensor:
        """Pool a packet's latent state into a ``(B, D)`` summary."""
        primary = packet.state.primary()
        if primary.dim() == 1:
            primary = primary.unsqueeze(0)
        return primary

    def _pooled_query(self, packet: LatentPacket) -> torch.Tensor:
        """Pool a packet's (possibly batched) latent into a single ``(1, D)``
        summary — one decision per packet."""
        return self._summarize(packet).mean(dim=0, keepdim=True)

    def expert_weights(self, packet: LatentPacket) -> dict[str, torch.Tensor]:
        """One differentiable routing weight per expert id (packet-pooled).

        For *per-input* (per batch element) weights — the form W3.1 uses to
        scale each hook's gate — call :meth:`forward` directly and map columns
        with :meth:`ExpertRegistry.ids`.
        """
        out = self.forward(self._pooled_query(packet))
        weights = out["weights"].squeeze(0)  # (E,)
        return {eid: weights[i] for i, eid in enumerate(self.registry.ids())}

    # -- BaseRouter API --------------------------------------------------
    def route(self, packet: LatentPacket, context: ToolContext, trace: AgentTrace) -> RouteDecision:
        ids = self.registry.ids()
        if not ids:
            return RouteDecision(self.fallback_action, confidence=0.0, metadata={"expert_weights": {}})

        out = self.forward(self._pooled_query(packet))
        weights = out["weights"].squeeze(0)                # (E,) — exactly top_k active
        confidence = float(out["confidence"].squeeze(0).item())

        # Top expert decides the graph action; its kind maps to a node.
        top_idx = int(torch.argmax(weights).item())
        top_id = ids[top_idx]
        action = self.action_map.get(self.registry.kind(top_id), self.fallback_action)

        active = {ids[i]: float(weights[i].item()) for i in range(len(ids)) if weights[i] > 0}
        return RouteDecision(
            action,
            confidence=confidence,
            metadata={"expert_weights": active, "top_expert": top_id},
        )


__all__ = [
    "BaseRouter",
    "RouteDecision",
    "StaticRouter",
    "LearnedLatentRouter",
    "DEFAULT_ACTION_MAP",
]

