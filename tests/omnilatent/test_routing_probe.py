"""W2.4 — routing probe: the router must select the right expert above chance."""

from __future__ import annotations

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.routing_probe import (
    build_routing_probe,
    evaluate_router,
    fit_router,
)


def _registry(n: int, key_dim: int = 24) -> ExpertRegistry:
    reg = ExpertRegistry(key_dim=key_dim)
    for i in range(n):
        reg.register(f"hook:{i}", "hook")
    return reg


def test_untrained_router_is_near_chance() -> None:
    torch.manual_seed(0)
    reg = _registry(5)
    router = LearnedLatentRouter(reg, input_dim=12, top_k=1)
    probe = build_routing_probe(reg, input_dim=12, samples_per_expert=40, seed=1)
    report = evaluate_router(router, probe)
    # An untrained router has no reason to beat chance by much.
    assert report["routing_accuracy"] <= 0.5
    assert report["chance"] == 1 / 5


def test_trained_router_routes_well_above_chance() -> None:
    torch.manual_seed(0)
    reg = _registry(5)
    router = LearnedLatentRouter(reg, input_dim=12, top_k=1)
    probe = build_routing_probe(reg, input_dim=12, samples_per_expert=40, seed=1)

    fit_router(router, probe, steps=300, lr=0.05)
    report = evaluate_router(router, probe)

    # The W2.4 gate: accuracy must be well above chance (CI fails otherwise).
    assert report["routing_accuracy"] > report["chance"] + 0.3
    assert report["routing_accuracy"] > 0.8


def test_probe_reports_calibration() -> None:
    torch.manual_seed(0)
    reg = _registry(4)
    router = LearnedLatentRouter(reg, input_dim=10, top_k=1)
    probe = build_routing_probe(reg, input_dim=10, samples_per_expert=32, seed=2)
    fit_router(router, probe, steps=200)
    report = evaluate_router(router, probe)
    assert "ece" in report
    assert 0.0 <= report["ece"] <= 1.0
