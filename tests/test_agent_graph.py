from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from omnilatent.agent import (
    AgentGraph,
    AgentRuntime,
    NodeType,
    RouteDecision,
    SideEffectViolation,
    StaticRouter,
    default_agent_graph,
)
from omnilatent.protocol import LatentPacket, LatentState, ToolContext, ToolResult


def _packet(value: float = 0.0) -> LatentPacket:
    return LatentPacket(state=LatentState(z_sem=torch.full((1, 4), value)))


def _trace_signature(trace) -> list[tuple[str, str | None, str | None, str | None]]:
    return [
        (step.node_type, step.input_packet_id, step.output_packet_id, step.selected_action)
        for step in trace.steps
    ]


def test_agent_import_does_not_import_mmwm_in_fresh_process() -> None:
    code = """
import sys
import omnilatent.agent
print(any(name == 'MMWM' or name.startswith('MMWM.') for name in sys.modules))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "False"


def test_default_graph_contains_required_nodes() -> None:
    graph = default_agent_graph()

    for node_type in NodeType:
        assert graph.allows(node_type)
    assert graph.node_for_action("MEMORY_READ") == NodeType.MEMORY_READ
    assert graph.node_for_action("STOP") == NodeType.ROUTE


def test_static_router_stop_action_produces_stop_trace() -> None:
    runtime = AgentRuntime(router=StaticRouter([RouteDecision("STOP", stop=True, stop_prob=0.99)]))

    out, trace = runtime.run(_packet(), max_steps=4)

    assert out.state.z_sem.shape == (1, 4)
    assert trace.metadata["stopped"] is True
    assert trace.metadata["budget_exhausted"] is False
    assert [step.node_type for step in trace.steps] == ["OBSERVE", "ENCODE", "ROUTE"]
    assert trace.steps[-1].selected_action == "STOP"
    assert trace.steps[-1].stop_prob == pytest.approx(0.99)


def test_agent_runtime_trace_is_deterministic_for_fixed_router() -> None:
    def memory_read(packet: LatentPacket, context: ToolContext) -> ToolResult:
        next_state = LatentState(z_sem=packet.state.z_sem + 1.0, z_ctx=torch.ones_like(packet.state.z_sem))
        return ToolResult(packet=LatentPacket(state=next_state, trace=packet.trace + ["memory.read"]))

    decisions = ["memory.read", RouteDecision("STOP", stop=True, stop_prob=1.0)]
    graph = AgentGraph(action_nodes={"memory.read": NodeType.MEMORY_READ, "STOP": NodeType.ROUTE})

    runtime_a = AgentRuntime(router=StaticRouter(decisions), graph=graph, tools={"memory.read": memory_read})
    runtime_b = AgentRuntime(router=StaticRouter(decisions), graph=graph, tools={"memory.read": memory_read})

    out_a, trace_a = runtime_a.run(_packet(), max_steps=4)
    out_b, trace_b = runtime_b.run(_packet(), max_steps=4)

    assert torch.equal(out_a.state.z_sem, out_b.state.z_sem)
    assert torch.equal(out_a.state.z_sem, torch.ones(1, 4))
    assert _trace_signature(trace_a) == _trace_signature(trace_b)
    assert trace_a.steps[3].node_type == "MEMORY_READ"
    assert trace_a.steps[3].tool_id == "memory.read"


def test_max_step_budget_is_enforced() -> None:
    runtime = AgentRuntime(router=StaticRouter(["MEMORY_READ"], repeat_last=True))

    out, trace = runtime.run(_packet(), max_steps=2)

    assert out is not None
    assert trace.metadata["stopped"] is False
    assert trace.metadata["budget_exhausted"] is True
    assert trace.metadata["steps_used"] == 2
    assert [s.node_type for s in trace.steps].count("ROUTE") == 2
    assert [s.node_type for s in trace.steps].count("MEMORY_READ") == 2


def test_side_effects_allowed_only_for_explicit_write_or_tool_nodes() -> None:
    def bad_read(packet: LatentPacket, context: ToolContext) -> ToolResult:
        return ToolResult(packet=packet, side_effects={"wrote": "unexpected"})

    graph = AgentGraph(action_nodes={"bad_read": NodeType.KB_READ})
    runtime = AgentRuntime(router=StaticRouter(["bad_read"]), graph=graph, tools={"bad_read": bad_read})

    with pytest.raises(SideEffectViolation, match="KB_READ"):
        runtime.run(_packet(), max_steps=1)


def test_side_effects_are_recorded_for_memory_write_node() -> None:
    memory_bank = {}

    def memory_write(packet: LatentPacket, context: ToolContext) -> ToolResult:
        memory_bank["latest"] = packet.state.z_sem.detach().clone()
        return ToolResult(packet=packet, side_effects={"memory_written": "latest"})

    graph = AgentGraph(action_nodes={"write": NodeType.MEMORY_WRITE, "STOP": NodeType.ROUTE})
    runtime = AgentRuntime(
        router=StaticRouter(["write", RouteDecision("STOP", stop=True)]),
        graph=graph,
        tools={"write": memory_write},
    )

    _, trace = runtime.run(_packet(3.0), max_steps=4)

    assert torch.equal(memory_bank["latest"], torch.full((1, 4), 3.0))
    write_step = next(step for step in trace.steps if step.node_type == "MEMORY_WRITE")
    assert write_step.side_effects == {"memory_written": "latest"}


def test_node_handlers_can_implement_think_and_decode_without_side_effects() -> None:
    def think(packet: LatentPacket, context: ToolContext) -> LatentPacket:
        return LatentPacket(
            state=LatentState(z_sem=packet.state.z_sem + 2.0),
            trace=packet.trace + ["think"],
            metadata=dict(packet.metadata),
        )

    def decode(packet: LatentPacket, context: ToolContext) -> LatentPacket:
        packet.metadata["decoded"] = True
        return packet

    runtime = AgentRuntime(
        router=StaticRouter(["DECODE", RouteDecision("STOP", stop=True)]),
        node_handlers={NodeType.THINK: think, NodeType.DECODE: decode},
    )

    out, trace = runtime.run(_packet(1.0), max_steps=3)

    # THINK runs before each route step; this trace routes DECODE then STOP.
    assert torch.equal(out.state.z_sem, torch.full((1, 4), 5.0))
    assert out.metadata["decoded"] is True
    assert [step.node_type for step in trace.steps].count("THINK") == 2
    assert "DECODE" in [step.node_type for step in trace.steps]

