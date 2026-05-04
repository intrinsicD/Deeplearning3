"""Router interfaces for explicit agent graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

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


__all__ = ["BaseRouter", "RouteDecision", "StaticRouter"]

