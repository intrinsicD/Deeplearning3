"""W4.1 — a plateau-triggered expansion hook becomes routable by the router."""

from __future__ import annotations

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.protocol import AgentTrace, LatentPacket, LatentState, ToolContext
from scripts.training.self_improve.forgetting import expand_omnilatent_capacity
from scripts.training.self_improve.plugins import get_plugin


def _plugin():
    return get_plugin("omnilatent")()


def test_expansion_registers_routable_expert() -> None:
    plugin = _plugin()
    dim = plugin.model.config.hidden_dim
    registry = ExpertRegistry(key_dim=dim)
    registry.sync_hooks(plugin.model.hook_manager)
    before = set(registry.ids_of_kind("hook"))

    hook = expand_omnilatent_capacity(plugin, num_tokens=4, registry=registry)

    after = set(registry.ids_of_kind("hook"))
    new_ids = after - before
    # The expansion hook is now a routable expert.
    assert new_ids == {f"hook:{hook.name}"}


def test_router_can_select_the_expansion_hook() -> None:
    plugin = _plugin()
    dim = plugin.model.config.hidden_dim
    registry = ExpertRegistry(key_dim=dim)
    expand_omnilatent_capacity(plugin, hook_name="grown", num_tokens=4, registry=registry)

    router = LearnedLatentRouter(registry, input_dim=dim, top_k=1)
    packet = LatentPacket(state=LatentState(z_sem=torch.randn(1, dim)))
    decision = router.route(packet, ToolContext(), AgentTrace())
    # With a single registered (expansion) expert, the router must select it.
    assert decision.metadata["top_expert"] == "hook:grown"


def test_expansion_without_registry_is_backward_compatible() -> None:
    # Omitting registry leaves the old behaviour untouched (no error).
    plugin = _plugin()
    hook = expand_omnilatent_capacity(plugin, num_tokens=4)
    assert hook.name in plugin.model.hook_manager.hooks
