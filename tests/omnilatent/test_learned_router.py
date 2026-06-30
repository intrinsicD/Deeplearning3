"""W2.2 — LearnedLatentRouter: sparse top-k selection over the expert registry."""

from __future__ import annotations

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import BaseRouter, LearnedLatentRouter, RouteDecision
from omnilatent.protocol import AgentTrace, LatentPacket, LatentState, ToolContext


def _registry(n: int, key_dim: int = 16) -> ExpertRegistry:
    reg = ExpertRegistry(key_dim=key_dim)
    for i in range(n):
        kind = ("hook", "tool", "kb")[i % 3]
        reg.register(f"{kind}:{i}", kind)
    return reg


def _packet(b: int, d: int) -> LatentPacket:
    return LatentPacket(state=LatentState(z_sem=torch.randn(b, d)))


def test_forward_sparse_topk_weights() -> None:
    reg = _registry(5, key_dim=16)
    router = LearnedLatentRouter(reg, input_dim=8, top_k=2)
    out = router.forward(torch.randn(4, 8))
    w = out["weights"]
    assert w.shape == (4, 5)
    # Exactly top_k experts active per row; active weights sum to 1.
    assert (w > 0).sum(dim=-1).tolist() == [2, 2, 2, 2]
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(4))
    assert out["confidence"].shape == (4,)


def test_weights_are_differentiable() -> None:
    reg = _registry(4, key_dim=16)
    router = LearnedLatentRouter(reg, input_dim=8, top_k=2)
    out = router.forward(torch.randn(2, 8))
    loss = out["weights"].sum()
    loss.backward()
    # Gradient reaches both the query projection and the expert keys.
    assert router.query_proj.weight.grad is not None
    assert any(p.grad is not None for p in reg.parameters())


def test_route_returns_valid_decision_and_expert_weights() -> None:
    reg = _registry(6, key_dim=16)
    router = LearnedLatentRouter(reg, input_dim=8, top_k=2)
    decision = router.route(_packet(3, 8), ToolContext(), AgentTrace())
    assert isinstance(decision, RouteDecision)
    assert decision.action in {"TOOL_CALL", "KB_READ", "DECODE"}
    assert 0.0 <= decision.confidence <= 1.0
    ew = decision.metadata["expert_weights"]
    assert len(ew) == 2  # top_k active experts named
    assert decision.metadata["top_expert"] in reg.ids()


def test_is_drop_in_baserouter() -> None:
    reg = _registry(3, key_dim=16)
    router = LearnedLatentRouter(reg, input_dim=8)
    assert isinstance(router, BaseRouter.__class__) or hasattr(router, "route")
    # Duck-typed BaseRouter: route(packet, context, trace) -> RouteDecision
    d = router.route(_packet(1, 8), ToolContext(), AgentTrace())
    assert isinstance(d, RouteDecision)


def test_empty_registry_falls_back() -> None:
    reg = ExpertRegistry(key_dim=16)
    router = LearnedLatentRouter(reg, input_dim=8, fallback_action="DECODE")
    d = router.route(_packet(1, 8), ToolContext(), AgentTrace())
    assert d.action == "DECODE"
    assert d.metadata["expert_weights"] == {}
    # forward on empty registry returns well-shaped empties.
    out = router.forward(torch.randn(2, 8))
    assert out["weights"].shape == (2, 0)


def test_topk_clamped_to_num_experts() -> None:
    reg = _registry(2, key_dim=16)
    router = LearnedLatentRouter(reg, input_dim=8, top_k=5)
    out = router.forward(torch.randn(1, 8))
    assert (out["weights"] > 0).sum().item() == 2


def test_expert_weights_keyed_by_id() -> None:
    reg = _registry(4, key_dim=16)
    router = LearnedLatentRouter(reg, input_dim=8, top_k=2)
    ew = router.expert_weights(_packet(3, 8))
    assert set(ew.keys()) == set(reg.ids())
    assert all(isinstance(v, torch.Tensor) for v in ew.values())
