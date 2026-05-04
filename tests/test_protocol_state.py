from __future__ import annotations

import subprocess
import sys

import torch

from omnilatent.protocol import (
    AgentTrace,
    AgentTraceStep,
    LatentPacket,
    LatentState,
    MemoryState,
    ObservationPacket,
    TargetSpec,
    TokenState,
    ToolContext,
    ToolDescriptor,
    ToolResult,
)


def test_protocol_import_does_not_import_mmwm_in_fresh_process() -> None:
    code = """
import sys
import omnilatent.protocol
print(any(name == 'MMWM' or name.startswith('MMWM.') for name in sys.modules))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "False"


def test_mmwm_containers_reexport_canonical_class_objects() -> None:
    from MMWM import containers as mmwm_containers

    assert mmwm_containers.ObservationPacket is ObservationPacket
    assert mmwm_containers.TargetSpec is TargetSpec
    assert mmwm_containers.TokenState is TokenState
    assert mmwm_containers.LatentState is LatentState
    assert mmwm_containers.MemoryState is MemoryState
    assert mmwm_containers.LatentPacket is LatentPacket
    assert mmwm_containers.ToolContext is ToolContext
    assert mmwm_containers.ToolDescriptor is ToolDescriptor
    assert mmwm_containers.ToolResult is ToolResult
    assert mmwm_containers.AgentTrace is AgentTrace
    assert mmwm_containers.AgentTraceStep is AgentTraceStep


def test_latent_state_serialization_preserves_z_ctx() -> None:
    latent = LatentState(
        z_sem=torch.randn(2, 4),
        z_dyn=torch.randn(2, 3),
        z_ctrl=torch.randn(2, 2),
        z_ctx=torch.randn(2, 5),
        extras={"aux": torch.ones(2, 1)},
    )

    restored = LatentState.from_dict(latent.to_dict())

    assert torch.equal(restored.z_sem, latent.z_sem)
    assert restored.z_ctx is not None
    assert latent.z_ctx is not None
    assert torch.equal(restored.z_ctx, latent.z_ctx)
    assert torch.equal(restored.extras["aux"], latent.extras["aux"])


def test_latent_state_backward_compat_accepts_legacy_z_mem() -> None:
    legacy_z_mem = torch.randn(2, 5)
    restored = LatentState.from_dict(
        {
            "z_sem": torch.randn(2, 4),
            "z_dyn": None,
            "z_ctrl": None,
            "z_mem": legacy_z_mem,
        }
    )

    assert restored.z_ctx is not None
    assert torch.equal(restored.z_ctx, legacy_z_mem)
    assert not hasattr(restored, "z_mem")


def test_memory_state_detach_handles_nested_tuple_list_and_extras() -> None:
    context = torch.randn(2, 4, requires_grad=True)
    hidden_a = torch.randn(2, 3, requires_grad=True)
    hidden_b = torch.randn(2, 2, requires_grad=True)
    extra = torch.randn(1, requires_grad=True)
    memory = MemoryState(
        context=context,
        hidden=(hidden_a, [hidden_b, "untouched"]),
        extras={"extra": extra, "plain": 3},
    )

    detached = memory.detach()

    assert detached.context is not None
    assert not detached.context.requires_grad
    assert isinstance(detached.hidden, tuple)
    assert not detached.hidden[0].requires_grad
    assert isinstance(detached.hidden[1], list)
    assert not detached.hidden[1][0].requires_grad
    assert detached.hidden[1][1] == "untouched"
    assert not detached.extras["extra"].requires_grad
    assert detached.extras["plain"] == 3


def test_tool_packet_and_agent_trace_construction() -> None:
    state = LatentState(z_sem=torch.randn(1, 8), z_ctx=torch.randn(1, 8))
    packet = LatentPacket(state=state, confidence=torch.tensor([0.9]), trace=["observe"])
    context = ToolContext(
        memory_state=MemoryState(context=torch.zeros(1, 8)),
        observation=ObservationPacket(modalities={"vector": torch.randn(1, 4)}),
        raw_inputs={"query": "hello"},
    )
    descriptor = ToolDescriptor(
        tool_id="kb_read_v1",
        kind="retrieval",
        version="1",
        reads_slots=("z_sem", "z_ctx"),
        writes_slots=("z_ctx",),
        binary=False,
    )
    result = ToolResult(packet=packet, metrics={"score": 1.0}, side_effects={})
    trace = AgentTrace()
    trace.append(
        AgentTraceStep(
            node_type="KB_READ",
            input_packet_id="p0",
            output_packet_id="p1",
            selected_action=descriptor.tool_id,
            tool_id=descriptor.tool_id,
        )
    )

    assert context.observation is not None
    assert context.observation.device() == packet.state.z_sem.device
    assert result.packet is packet
    assert descriptor.reads_slots == ("z_sem", "z_ctx")
    assert trace.steps[0].node_type == "KB_READ"
    assert trace.steps[0].tool_id == "kb_read_v1"


def test_target_spec_construction() -> None:
    target = TargetSpec(
        modality="text",
        data=torch.ones(2, 4, dtype=torch.long),
        metadata={"objective": "teacher_forced"},
    )

    assert target.modality == "text"
    assert target.data is not None
    assert target.data.shape == (2, 4)
    assert target.metadata["objective"] == "teacher_forced"


def test_token_state_detach_preserves_metadata_and_detaches_tensors() -> None:
    token_state = TokenState(
        tokens=torch.randn(2, 3, 4, requires_grad=True),
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
        modality="text",
        segment_ids=torch.zeros(2, 3, dtype=torch.long),
        positions=torch.arange(3).expand(2, 3),
        metadata={"source": "unit"},
    )

    detached = token_state.detach()

    assert not detached.tokens.requires_grad
    assert detached.attention_mask is not None
    assert detached.segment_ids is not None
    assert detached.positions is not None
    assert detached.modality == "text"
    assert detached.metadata == {"source": "unit"}

