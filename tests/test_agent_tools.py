from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from omnilatent.agent.tools import (
    BaseTool,
    BinaryToolLoader,
    LatentRouter,
    MemoryReadTool,
    MemoryWriteTool,
    ToolExecutionEngine,
    ToolRegistrySystem,
)
from omnilatent.protocol import LatentPacket, LatentState, ToolContext, ToolDescriptor, ToolResult


def _packet(batch_size: int = 1, value: float = 0.0) -> LatentPacket:
    return LatentPacket(state=LatentState(z_sem=torch.full((batch_size, 4), value)))


class AddOneTool(BaseTool):
    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        next_packet = LatentPacket(
            state=LatentState(z_sem=packet.state.z_sem + 1.0),
            source_tool=self.descriptor.tool_id,
            trace=packet.trace + [self.descriptor.tool_id],
        )
        return ToolResult(packet=next_packet, confidence=torch.tensor([0.75]), metrics={"quality": 1.0})


class FixedRouter(nn.Module):
    def __init__(self, action_logits: torch.Tensor, stop_logit: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("action_logits", action_logits)
        self.register_buffer("stop_logit", stop_logit)

    def forward(self, packet: LatentPacket) -> dict[str, torch.Tensor]:
        batch = packet.state.z_sem.shape[0]
        return {
            "action_logits": self.action_logits[:batch],
            "stop_logit": self.stop_logit[:batch],
        }


def _registry_with_loaded_tool(tool: BaseTool) -> ToolRegistrySystem:
    loader = BinaryToolLoader(device=torch.device("cpu"))
    loader._loaded[tool.descriptor.tool_id] = tool
    registry = ToolRegistrySystem(loader)
    registry.register(tool.descriptor, alias="add")
    return registry


def test_agent_tools_import_does_not_import_mmwm_in_fresh_process() -> None:
    code = """
import sys
import omnilatent.agent.tools
print(any(name == 'MMWM' or name.startswith('MMWM.') for name in sys.modules))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "False"


def test_latent_router_consumes_canonical_latent_packet() -> None:
    packet = LatentPacket(
        state=LatentState(
            z_sem=torch.randn(2, 4),
            z_dyn=torch.randn(2, 2),
            z_ctrl=torch.randn(2, 1),
        )
    )
    router = LatentRouter(latent_dim=7, num_actions=3, hidden_dim=8)

    route = router(packet)

    assert route["action_logits"].shape == (2, 3)
    assert route["stop_logit"].shape == (2, 1)


def test_tool_registry_calls_return_canonical_tool_result() -> None:
    descriptor = ToolDescriptor(tool_id="add_one.v1", kind="test", version="1", entry_point="loaded")
    tool = AddOneTool(descriptor)
    registry = _registry_with_loaded_tool(tool)

    result = registry.call("add", _packet(), ToolContext())

    assert isinstance(result, ToolResult)
    assert result.packet is not None
    assert torch.equal(result.packet.state.z_sem, torch.ones(1, 4))
    assert result.confidence is not None


def test_tool_execution_trace_includes_selected_tool_and_confidence() -> None:
    descriptor = ToolDescriptor(tool_id="add_one.v1", kind="test", version="1", entry_point="loaded")
    registry = _registry_with_loaded_tool(AddOneTool(descriptor))
    router = FixedRouter(
        action_logits=torch.tensor([[10.0, -10.0]]),
        stop_logit=torch.tensor([[-10.0]]),
    )
    engine = ToolExecutionEngine(
        registry=registry,
        router=router,  # type: ignore[arg-type]
        action_id_to_tool={0: "add"},
        device=torch.device("cpu"),
    )

    out, trace = engine.iterate(_packet(), ToolContext(), max_steps=1)

    assert torch.equal(out.state.z_sem, torch.ones(1, 4))
    assert trace[0]["tool"] == "add"
    assert trace[0]["action_id"] == 0
    assert trace[0]["action_ids"] == [0]
    assert trace[0]["confidence"] == pytest.approx(0.75)


def test_tool_execution_batch_limitation_is_explicit() -> None:
    descriptor = ToolDescriptor(tool_id="add_one.v1", kind="test", version="1", entry_point="loaded")
    registry = _registry_with_loaded_tool(AddOneTool(descriptor))
    router = FixedRouter(
        action_logits=torch.tensor([[10.0, -10.0], [-10.0, 10.0]]),
        stop_logit=torch.tensor([[-10.0], [-10.0]]),
    )
    engine = ToolExecutionEngine(
        registry=registry,
        router=router,  # type: ignore[arg-type]
        action_id_to_tool={0: "add", 1: "other"},
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="batched route with different tool actions"):
        engine.iterate(_packet(batch_size=2), ToolContext(), max_steps=1)


def test_memory_read_and_write_tools_use_canonical_packets() -> None:
    bank = {}
    write_descriptor = ToolDescriptor(tool_id="memory.write.v1", kind="memory.write", version="1")
    read_descriptor = ToolDescriptor(tool_id="memory.read.v1", kind="memory.read", version="1")
    writer = MemoryWriteTool(write_descriptor, bank)
    reader = MemoryReadTool(read_descriptor, bank)
    packet = LatentPacket(
        state=LatentState(z_sem=torch.zeros(1, 4), z_ctx=torch.full((1, 4), 3.0)),
        confidence=torch.tensor([0.9]),
    )
    context = ToolContext(raw_inputs={"memory_key": "slot"})

    write_result = writer.run(packet, context)
    read_result = reader.run(LatentPacket(state=LatentState(z_sem=torch.zeros(1, 4))), context)

    assert write_result.side_effects == {"memory_written": "slot"}
    assert "slot" in bank
    assert read_result.packet is not None
    assert read_result.packet.source_tool == "memory.read.v1"
    assert read_result.packet.state.z_ctx is not None
    assert torch.equal(read_result.packet.state.z_ctx, torch.full((1, 4), 3.0))


def test_mmwm_tools_still_use_canonical_packet_classes() -> None:
    from MMWM.containers import LatentPacket as MMWMLatentPacket
    from MMWM.containers import ToolResult as MMWMToolResult

    assert MMWMLatentPacket is LatentPacket
    assert MMWMToolResult is ToolResult

