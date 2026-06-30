"""W2.3 — calibrated abstention + retrieval-as-routing."""

from __future__ import annotations

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.routing_metrics import (
    expected_calibration_error,
    routing_accuracy,
)
from omnilatent.protocol import AgentTrace, LatentPacket, LatentState, ToolContext


def _registry(n: int, key_dim: int = 16) -> ExpertRegistry:
    reg = ExpertRegistry(key_dim=key_dim)
    for i in range(n):
        reg.register(f"hook:{i}", "hook")
    return reg


def _packet(d: int) -> LatentPacket:
    return LatentPacket(state=LatentState(z_sem=torch.randn(1, d)))


def test_abstains_below_threshold() -> None:
    reg = _registry(6, key_dim=16)
    # threshold 1.0 forces abstention always (confidence < 1.0).
    router = LearnedLatentRouter(reg, input_dim=8, abstain_threshold=1.0, abstain_action="KB_READ")
    d = router.route(_packet(8), ToolContext(), AgentTrace())
    assert d.action == "KB_READ"
    assert d.metadata["abstained"] is True
    assert d.metadata["expert_weights"] == {}  # no skill forced


def test_does_not_abstain_when_confident() -> None:
    reg = _registry(3, key_dim=16)
    # threshold 0.0 ⇒ never abstain.
    router = LearnedLatentRouter(reg, input_dim=8, abstain_threshold=0.0)
    d = router.route(_packet(8), ToolContext(), AgentTrace())
    assert d.metadata["abstained"] is False
    assert len(d.metadata["expert_weights"]) >= 1


def test_retrieval_fn_attached_on_abstention() -> None:
    reg = _registry(4, key_dim=16)
    calls = {"n": 0}

    def fake_retrieval(query: torch.Tensor):
        calls["n"] += 1
        return ["doc-a", "doc-b"]

    router = LearnedLatentRouter(
        reg, input_dim=8, abstain_threshold=1.0, retrieval_fn=fake_retrieval
    )
    d = router.route(_packet(8), ToolContext(), AgentTrace())
    assert d.metadata["retrieval_candidates"] == ["doc-a", "doc-b"]
    assert calls["n"] == 1


def test_routing_accuracy_metric() -> None:
    assert routing_accuracy(["a", "b", "c"], ["a", "x", "c"]) == 2 / 3
    assert routing_accuracy([], []) == 0.0


def test_ece_is_zero_for_perfect_calibration() -> None:
    # Confidence exactly equals accuracy in each bin → ECE 0.
    confidences = [0.0, 0.0, 1.0, 1.0]
    correct = [False, False, True, True]
    assert expected_calibration_error(confidences, correct, n_bins=10) == 0.0


def test_ece_detects_overconfidence() -> None:
    # Always 90% confident but only 50% correct → ECE ~ 0.4.
    confidences = [0.9] * 10
    correct = [True, False] * 5
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert 0.35 < ece < 0.45
