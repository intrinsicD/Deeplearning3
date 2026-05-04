from __future__ import annotations

from pathlib import Path

import pytest
import torch

from omnilatent.agent import AgentGraph, AgentRuntime, NodeType, RouteDecision, SideEffectViolation, StaticRouter
from omnilatent.core.config import KBConfig
from omnilatent.kb import KBWriteTool, retrieve_top_k
from omnilatent.model.hooks import NeuralPort, NeuralPortSpec
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.config import OmniLatentConfig
from omnilatent.protocol import LatentPacket, LatentState, ToolContext, ToolDescriptor


def _config() -> OmniLatentConfig:
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        gradient_checkpointing=False,
        vocab_size=256,
        image_size=32,
        image_patch_size=8,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        audio_n_mels=32,
    )


def _packet() -> LatentPacket:
    return LatentPacket(state=LatentState(z_sem=torch.zeros(1, 4)))


def test_extension_swap_does_not_change_base_weights_after_forward() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    text = torch.randint(1, cfg.vocab_size, (1, 8))
    base_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("hook_manager.")
    }
    port = NeuralPort(NeuralPortSpec(
        name="retrieval_ext",
        kind="retrieval",
        latent_dim=cfg.hidden_dim,
        hook_tokens=2,
        target_layers=[0, 1],
    ))

    model.hook_manager.register_port(port)
    with torch.no_grad():
        model.reconstruct("text", text)
    removed = model.hook_manager.remove_port("retrieval_ext")
    assert removed is port
    model.hook_manager.register_port(port)
    with torch.no_grad():
        model.reconstruct("text", text)

    for name, tensor in model.state_dict().items():
        if name.startswith("hook_manager."):
            continue
        assert torch.equal(tensor, base_before[name]), name


def test_kb_write_only_occurs_through_explicit_kb_write_graph_node(tmp_path: Path) -> None:
    cfg = KBConfig(root_dir=str(tmp_path / "kb"), embedding_dim=32, chunk_tokens=32, chunk_overlap=0)
    tool = KBWriteTool(ToolDescriptor(tool_id="kb.write.v1", kind="kb.write", version="1"), cfg)
    context = ToolContext(raw_inputs={"text": "Explicit graph node writes AgentGraph memory."})
    graph = AgentGraph(action_nodes={"kb.write": NodeType.KB_WRITE, "STOP": NodeType.ROUTE})
    runtime = AgentRuntime(
        router=StaticRouter(["kb.write", RouteDecision("STOP", stop=True)]),
        graph=graph,
        tools={"kb.write": tool.run},
    )

    out, trace = runtime.run(_packet(), context, max_steps=4)

    assert out.metadata["kb_written_id"]
    assert Path(cfg.root_dir, cfg.chunks_path).exists()
    write_steps = [step for step in trace.steps if step.node_type == "KB_WRITE"]
    assert len(write_steps) == 1
    assert write_steps[0].side_effects["kb_written"] == out.metadata["kb_written_id"]
    assert retrieve_top_k("AgentGraph memory", cfg, top_k=1)


def test_kb_write_side_effect_rejected_when_routed_as_kb_read(tmp_path: Path) -> None:
    cfg = KBConfig(root_dir=str(tmp_path / "kb"), embedding_dim=32, chunk_tokens=32, chunk_overlap=0)
    tool = KBWriteTool(ToolDescriptor(tool_id="kb.write.v1", kind="kb.write", version="1"), cfg)
    context = ToolContext(raw_inputs={"text": "This should not be written through KB_READ."})
    graph = AgentGraph(action_nodes={"bad_write": NodeType.KB_READ})
    runtime = AgentRuntime(
        router=StaticRouter(["bad_write"]),
        graph=graph,
        tools={"bad_write": tool.run},
    )

    with pytest.raises(SideEffectViolation, match="KB_READ"):
        runtime.run(_packet(), context, max_steps=1)

    # The tool did execute before the runtime rejected the side effect, so the
    # important invariant here is that the graph refuses to accept/trace the
    # side effect as a valid non-write node.
    assert Path(cfg.root_dir, cfg.chunks_path).exists()


def test_tool_call_can_be_side_effect_capable_but_trace_names_selected_tool(tmp_path: Path) -> None:
    cfg = KBConfig(root_dir=str(tmp_path / "kb"), embedding_dim=32, chunk_tokens=32, chunk_overlap=0)
    tool = KBWriteTool(ToolDescriptor(tool_id="kb.write.v1", kind="kb.write", version="1"), cfg)
    graph = AgentGraph(action_nodes={"generic_tool": NodeType.TOOL_CALL})
    runtime = AgentRuntime(
        router=StaticRouter(["generic_tool"]),
        graph=graph,
        tools={"generic_tool": tool.run},
    )

    _, trace = runtime.run(_packet(), ToolContext(raw_inputs={"text": "generic tool write"}), max_steps=1)

    tool_steps = [step for step in trace.steps if step.node_type == "TOOL_CALL"]
    assert len(tool_steps) == 1
    assert tool_steps[0].selected_action == "generic_tool"
    assert tool_steps[0].tool_id == "generic_tool"
    assert "kb_written" in tool_steps[0].side_effects

