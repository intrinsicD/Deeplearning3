"""W3.3 — credit assignment v1: load-balancing aux + positive counterfactual lift."""

from __future__ import annotations

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.routing_metrics import load_balancing_loss
from omnilatent.agent.routing_probe import (
    build_routing_probe,
    counterfactual_lift,
    fit_router,
)


def _registry(n: int, key_dim: int = 24) -> ExpertRegistry:
    reg = ExpertRegistry(key_dim=key_dim)
    for i in range(n):
        reg.register(f"hook:{i}", "hook")
    return reg


def test_load_balancing_loss_prefers_uniform() -> None:
    n, e = 64, 4
    # Uniform logits → balanced load → loss ≈ 1.0 (its minimum).
    uniform = torch.zeros(n, e)
    bal = load_balancing_loss(uniform)
    assert abs(bal.item() - 1.0) < 0.05

    # Everything routed to expert 0 → maximally imbalanced → loss ≈ E.
    collapsed = torch.zeros(n, e)
    collapsed[:, 0] = 20.0
    col = load_balancing_loss(collapsed)
    assert col.item() > bal.item() + 1.0


def test_load_balancing_loss_is_differentiable() -> None:
    logits = torch.randn(16, 5, requires_grad=True)
    load_balancing_loss(logits).backward()
    assert logits.grad is not None


def test_trained_router_has_positive_counterfactual_lift() -> None:
    torch.manual_seed(0)
    reg = _registry(5)
    router = LearnedLatentRouter(reg, input_dim=12, top_k=1)
    probe = build_routing_probe(reg, input_dim=12, samples_per_expert=40, seed=1)

    fit_router(router, probe, steps=300, lr=0.05, load_balance_weight=0.01)
    lift = counterfactual_lift(router, probe)

    # The W3.3 gate: routing the chosen expert beats both random and no-expert.
    assert lift["lift_vs_random"] > 0.0
    assert lift["lift_vs_backbone"] > 0.0
    assert lift["routing_accuracy"] > 0.8


def test_load_balancing_keeps_experts_used() -> None:
    torch.manual_seed(0)
    reg = _registry(5)
    router = LearnedLatentRouter(reg, input_dim=12, top_k=1)
    probe = build_routing_probe(reg, input_dim=12, samples_per_expert=40, seed=2)

    fit_router(router, probe, steps=300, lr=0.05, load_balance_weight=0.05)
    with torch.no_grad():
        top1 = router.forward(probe.inputs)["weights"].argmax(dim=-1)
    used = torch.bincount(top1, minlength=len(reg))
    # No expert is abandoned (each gets at least some traffic).
    assert (used > 0).all()
