"""Deterministic runtime for explicit OmniLatent agent graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from omnilatent.agent.graph import AgentGraph, NodeType, default_agent_graph
from omnilatent.agent.router import BaseRouter, RouteDecision
from omnilatent.protocol import AgentTrace, AgentTraceStep, LatentPacket, ToolContext, ToolResult

ToolHandler = Callable[[LatentPacket, ToolContext], ToolResult]
NodeHandler = Callable[[LatentPacket, ToolContext], LatentPacket]


class AgentRuntimeError(RuntimeError):
    """Base error for explicit agent runtime failures."""


class SideEffectViolation(AgentRuntimeError):
    """Raised when a non-side-effect node returns side effects."""


@dataclass
class AgentRuntime:
    """Run a canonical packet through an explicit agent graph.

    The runtime is deliberately small: OBSERVE/ENCODE are no-op traceable steps
    unless a handler is supplied; the router chooses subsequent graph nodes;
    tools/handlers are invoked only for explicit node types. Side effects are
    rejected unless they originate from side-effect-capable nodes.
    """

    router: BaseRouter
    graph: AgentGraph = field(default_factory=default_agent_graph)
    tools: Mapping[str, ToolHandler] = field(default_factory=dict)
    node_handlers: Mapping[NodeType, NodeHandler] = field(default_factory=dict)
    side_effect_nodes: frozenset[NodeType] = frozenset({NodeType.TOOL_CALL, NodeType.MEMORY_WRITE, NodeType.KB_WRITE})

    def run(
        self,
        packet: LatentPacket,
        context: ToolContext | None = None,
        *,
        max_steps: int = 8,
    ) -> tuple[LatentPacket, AgentTrace]:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        context = ToolContext() if context is None else context
        trace = AgentTrace(metadata={"max_steps": max_steps, "stopped": False, "budget_exhausted": False})
        current = packet
        packet_id = 0

        current = self._execute_passive_node(NodeType.OBSERVE, current, context, trace, packet_id)
        packet_id += 1
        current = self._execute_passive_node(NodeType.ENCODE, current, context, trace, packet_id)
        packet_id += 1

        route_steps = 0
        while route_steps < max_steps:
            route_steps += 1
            if self.graph.allows(NodeType.THINK) and NodeType.THINK in self.node_handlers:
                current = self._execute_passive_node(NodeType.THINK, current, context, trace, packet_id)
                packet_id += 1

            decision = self.router.route(current, context, trace)
            trace.append(
                AgentTraceStep(
                    node_type=NodeType.ROUTE.value,
                    input_packet_id=f"p{packet_id}",
                    output_packet_id=f"p{packet_id}",
                    selected_action=decision.action,
                    stop_prob=decision.stop_prob,
                    metadata={"confidence": decision.confidence, **decision.metadata},
                )
            )
            if decision.stop or decision.action == "STOP":
                trace.metadata["stopped"] = True
                break

            node_type = self.graph.node_for_action(decision.action)
            if node_type == NodeType.ROUTE:
                continue
            result = self._execute_action_node(node_type, decision, current, context)
            self._validate_side_effects(node_type, result)
            next_packet = result.packet if result.packet is not None else current
            trace.append(
                AgentTraceStep(
                    node_type=str(node_type.value),
                    input_packet_id=f"p{packet_id}",
                    output_packet_id=f"p{packet_id + 1}",
                    selected_action=decision.action,
                    tool_id=decision.action if decision.action in self.tools else None,
                    side_effects=dict(result.side_effects),
                    metadata={"metrics": dict(result.metrics)},
                )
            )
            current = next_packet
            packet_id += 1
        else:
            trace.metadata["budget_exhausted"] = True

        trace.metadata["steps_used"] = route_steps
        return current, trace

    def _execute_passive_node(
        self,
        node_type: NodeType,
        packet: LatentPacket,
        context: ToolContext,
        trace: AgentTrace,
        packet_id: int,
    ) -> LatentPacket:
        if not self.graph.allows(node_type):
            return packet
        handler = self.node_handlers.get(node_type)
        next_packet = handler(packet, context) if handler is not None else packet
        trace.append(
            AgentTraceStep(
                node_type=node_type.value,
                input_packet_id=f"p{packet_id}",
                output_packet_id=f"p{packet_id + (0 if next_packet is packet else 1)}",
            )
        )
        return next_packet

    def _execute_action_node(
        self,
        node_type: NodeType,
        decision: RouteDecision,
        packet: LatentPacket,
        context: ToolContext,
    ) -> ToolResult:
        if decision.action in self.tools:
            return self.tools[decision.action](packet, context)
        if node_type in self.node_handlers:
            next_packet = self.node_handlers[node_type](packet, context)
            return ToolResult(packet=next_packet)
        if node_type in {NodeType.DECODE, NodeType.THINK, NodeType.MEMORY_READ, NodeType.KB_READ}:
            return ToolResult(packet=packet)
        raise AgentRuntimeError(f"No handler/tool registered for action {decision.action!r} ({node_type.value})")

    def _validate_side_effects(self, node_type: NodeType, result: ToolResult) -> None:
        if result.side_effects and node_type not in self.side_effect_nodes:
            raise SideEffectViolation(
                f"Node {node_type.value} returned side effects outside allowed side-effect nodes: "
                f"{sorted(result.side_effects)}"
            )


__all__ = ["AgentRuntime", "AgentRuntimeError", "SideEffectViolation", "ToolHandler", "NodeHandler"]

