"""Phase 5 (research lane) — outcome-based credit (W5.1) and OOD study (W5.3).

These are measurement harnesses: they assert the *mechanism* works and reports
a metric, not a particular research outcome.
"""

from __future__ import annotations

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.routing_probe import (
    build_routing_probe,
    evaluate_router,
    fit_router,
    fit_router_outcome_based,
    ood_abstention_study,
)


def _registry(n: int, key_dim: int = 24) -> ExpertRegistry:
    reg = ExpertRegistry(key_dim=key_dim)
    for i in range(n):
        reg.register(f"hook:{i}", "hook")
    return reg


def test_outcome_based_credit_learns_above_chance() -> None:
    # W5.1: the router learns to route from a scalar reward alone (no gold
    # label on the logits), via REINFORCE.
    torch.manual_seed(0)
    reg = _registry(4)
    router = LearnedLatentRouter(reg, input_dim=12, top_k=1)
    probe = build_routing_probe(reg, input_dim=12, samples_per_expert=40, seed=1)

    fit_router_outcome_based(router, probe, steps=600, lr=0.05)
    report = evaluate_router(router, probe)

    # Outcome-based credit should beat chance clearly (weaker than supervised
    # v1, which is the point of comparing the two).
    assert report["routing_accuracy"] > report["chance"] + 0.2


def test_ood_study_reports_confidence_gap() -> None:
    # W5.3: the study runs and reports a confidence gap. We assert it produces
    # the metric and that confidence is a valid probability, NOT that the gap
    # has a particular sign (the honest, open finding).
    torch.manual_seed(0)
    reg = _registry(5)
    router = LearnedLatentRouter(reg, input_dim=12, top_k=1)
    probe = build_routing_probe(reg, input_dim=12, samples_per_expert=40, seed=1)
    fit_router(router, probe, steps=200)

    report = ood_abstention_study(router, probe)
    assert 0.0 <= report["confidence_id"] <= 1.0
    assert 0.0 <= report["confidence_ood"] <= 1.0
    assert report["confidence_gap"] == report["confidence_id"] - report["confidence_ood"]


def test_ood_study_is_deterministic() -> None:
    torch.manual_seed(0)
    reg = _registry(3)
    router = LearnedLatentRouter(reg, input_dim=10, top_k=1)
    probe = build_routing_probe(reg, input_dim=10, samples_per_expert=20, seed=1)
    a = ood_abstention_study(router, probe, seed=42)
    b = ood_abstention_study(router, probe, seed=42)
    assert a == b
