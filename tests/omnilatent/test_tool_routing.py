"""Bug 3 — a selected tool expert routes to and executes the real tool."""

from __future__ import annotations

import torch

from omnilatent.agent.graph import NodeType, default_agent_graph
from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.runtime import AgentRuntime
from omnilatent.protocol import (
    AgentTrace,
    LatentPacket,
    LatentState,
    ToolContext,
    ToolResult,
)


def _packet(d: int = 8) -> LatentPacket:
    return LatentPacket(state=LatentState(z_sem=torch.randn(1, d)))


def test_tool_expert_emits_dispatch_id_not_generic_tool_call() -> None:
    reg = ExpertRegistry(key_dim=8)
    reg.register("tool:search", "tool")  # action defaults to "search"
    router = LearnedLatentRouter(reg, input_dim=8, top_k=1)
    decision = router.route(_packet(), ToolContext(), AgentTrace())
    assert decision.action == "search"          # not "TOOL_CALL"
    assert decision.metadata["top_expert"] == "tool:search"


def test_registry_tool_actions_map() -> None:
    reg = ExpertRegistry(key_dim=8)
    reg.register("tool:search", "tool")
    reg.register("calc", "tool", action="calculate")
    reg.register("hook:x", "hook")
    assert reg.tool_actions() == {"search": "tool:search", "calculate": "calc"}


def test_runtime_executes_selected_tool() -> None:
    reg = ExpertRegistry(key_dim=8)
    reg.register("tool:search", "tool")
    router = LearnedLatentRouter(reg, input_dim=8, top_k=1)

    calls = {"n": 0}

    def search_tool(packet: LatentPacket, context: ToolContext) -> ToolResult:
        calls["n"] += 1
        return ToolResult(packet=packet, side_effects={"queried": "kb"})

    graph = default_agent_graph(tool_actions=reg.tool_actions())
    runtime = AgentRuntime(router=router, graph=graph, tools={"search": search_tool})

    runtime.run(_packet(), ToolContext(), max_steps=2)
    # The selected tool actually ran (and TOOL_CALL permits its side effects).
    assert calls["n"] >= 1


def test_graph_resolves_tool_action_to_tool_call_node() -> None:
    reg = ExpertRegistry(key_dim=8)
    reg.register("tool:search", "tool")
    graph = default_agent_graph(tool_actions=reg.tool_actions())
    assert graph.node_for_action("search") == NodeType.TOOL_CALL
