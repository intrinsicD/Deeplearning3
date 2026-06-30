"""W3.2 — router execution loop: selected hooks co-activate; runtime drop-in."""

from __future__ import annotations

import pytest
import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.routed_forward import RoutedForward
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.runtime import AgentRuntime, SideEffectViolation
from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.protocol import (
    AgentTrace,
    LatentPacket,
    LatentState,
    ToolContext,
    ToolResult,
)


# --- facet (b): routed model forward -----------------------------------
def _model_with_hooks(n_hooks: int = 3) -> OmniLatentModel:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    model = OmniLatentModel(cfg).eval()
    for i in range(n_hooks):
        model.register_hook(
            LatentNeuralHook(
                name=f"skill{i}", num_tokens=4, dim=cfg.hidden_dim,
                target_layers=[0, 1], gate_bias_init=1.0,
            )
        )
    return model


def _routed(model: OmniLatentModel, top_k: int = 2) -> RoutedForward:
    reg = ExpertRegistry(key_dim=model.config.hidden_dim)
    reg.sync_hooks(model.hook_manager)
    router = LearnedLatentRouter(reg, input_dim=model.config.hidden_dim, top_k=top_k)
    return RoutedForward(model=model, router=router)


def test_top_k_hooks_co_activate() -> None:
    model = _model_with_hooks(3)
    rf = _routed(model, top_k=2)
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)  # B=1

    result, step = rf.route_and_forward("image", img, "image")
    assert result["output"].shape == (1, 3, model.config.image_size, model.config.image_size)
    # Exactly top_k hooks active for a single input, all real hooks, recorded.
    assert len(step.metadata["active_hooks"]) == 2
    assert set(step.metadata["active_hooks"]).issubset({"skill0", "skill1", "skill2"})
    assert step.hook_gates  # effective gates logged


def test_route_weights_cleared_after_forward() -> None:
    model = _model_with_hooks(2)
    rf = _routed(model)
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    rf.route_and_forward("image", img, "image")
    # No routing state leaks into the next (unrelated) forward.
    assert model.hook_manager._route_weights == {}


def test_selected_hooks_change_output_vs_no_hooks() -> None:
    model = _model_with_hooks(3)
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    with torch.no_grad():
        base = model.reconstruct("image", img)["output"]
    rf = _routed(model, top_k=2)
    routed_out, _ = rf.route_and_forward("image", img, "image")
    # Routing activates a subset of hooks → output differs from all-hooks recon.
    assert not torch.allclose(routed_out["output"], base)


# --- facet (a): LearnedLatentRouter as an AgentRuntime drop-in ----------
def _registry_hooks(n: int, key_dim: int = 16) -> ExpertRegistry:
    reg = ExpertRegistry(key_dim=key_dim)
    for i in range(n):
        reg.register(f"hook:{i}", "hook")  # hook → DECODE action (passive node)
    return reg


def test_runtime_drives_learned_router() -> None:
    reg = _registry_hooks(4)
    router = LearnedLatentRouter(reg, input_dim=8, top_k=2)
    runtime = AgentRuntime(router=router)
    packet = LatentPacket(state=LatentState(z_sem=torch.randn(1, 8)))

    final, trace = runtime.run(packet, ToolContext(), max_steps=2)
    route_steps = [s for s in trace.steps if s.node_type == "ROUTE"]
    assert route_steps
    # The router's expert selection flows into the trace.
    assert "expert_weights" in route_steps[0].metadata


def test_side_effect_guard_intact_with_learned_router() -> None:
    reg = _registry_hooks(3)
    router = LearnedLatentRouter(reg, input_dim=8, top_k=1)

    # A tool bound to DECODE (a non-side-effect node) that emits side effects
    # must still be rejected when the LearnedLatentRouter routes to it.
    def bad_tool(packet: LatentPacket, context: ToolContext) -> ToolResult:
        return ToolResult(packet=packet, side_effects={"wrote": "disk"})

    runtime = AgentRuntime(router=router, tools={"DECODE": bad_tool})
    packet = LatentPacket(state=LatentState(z_sem=torch.randn(1, 8)))
    with pytest.raises(SideEffectViolation):
        runtime.run(packet, ToolContext(), max_steps=2)
