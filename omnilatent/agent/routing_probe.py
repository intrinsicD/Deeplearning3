"""Synthetic routing probe (work plan W2.4).

A frozen, deterministic benchmark that measures whether a
:class:`~omnilatent.agent.router.LearnedLatentRouter` actually selects the
right expert — the non-negotiable measurement gate from Audit.md P1. Each
synthetic task is solvable by exactly one expert: every expert owns a random
prototype in input space, and a sample is that prototype plus noise with the
expert as its gold label. A router that routes correctly should land well above
chance (``1 / num_experts``); a router that ignores the input cannot.

The probe also yields per-sample ``(confidence, correct)`` pairs so the
abstention signal's calibration (ECE) can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.routing_metrics import (
    expected_calibration_error,
    load_balancing_loss,
    routing_accuracy,
)


@dataclass
class RoutingProbe:
    """A frozen routing benchmark."""

    inputs: torch.Tensor       # (N, input_dim)
    gold_idx: torch.Tensor     # (N,) long — index into expert_ids
    expert_ids: list[str]
    input_dim: int

    @property
    def num_experts(self) -> int:
        return len(self.expert_ids)

    @property
    def chance(self) -> float:
        return 1.0 / self.num_experts


def build_routing_probe(
    registry: ExpertRegistry,
    input_dim: int,
    samples_per_expert: int = 32,
    noise: float = 0.3,
    seed: int = 0,
) -> RoutingProbe:
    """Build a deterministic routing probe over the registry's experts.

    One random prototype per expert; ``samples_per_expert`` noisy draws around
    each prototype labelled with that expert.
    """
    ids = registry.ids()
    if not ids:
        raise ValueError("registry has no experts to route among")
    gen = torch.Generator().manual_seed(seed)
    e = len(ids)
    prototypes = torch.randn(e, input_dim, generator=gen)

    xs, ys = [], []
    for i in range(e):
        block = prototypes[i].unsqueeze(0) + noise * torch.randn(
            samples_per_expert, input_dim, generator=gen
        )
        xs.append(block)
        ys.append(torch.full((samples_per_expert,), i, dtype=torch.long))
    return RoutingProbe(
        inputs=torch.cat(xs, dim=0),
        gold_idx=torch.cat(ys, dim=0),
        expert_ids=list(ids),
        input_dim=input_dim,
    )


def fit_router(
    router: LearnedLatentRouter,
    probe: RoutingProbe,
    steps: int = 300,
    lr: float = 0.05,
    load_balance_weight: float = 0.0,
) -> None:
    """Train the router (query projection + expert keys) to route the probe.

    Credit assignment v1 (W3.3): full-softmax cross-entropy over the routing
    logits against the gold expert, plus an optional Switch-style
    load-balancing auxiliary (``load_balance_weight``) that discourages expert
    collapse.
    """
    # The registry is a submodule of the router, so router.parameters() already
    # includes the expert keys alongside the query projection.
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = router.forward(probe.inputs)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, probe.gold_idx)
        if load_balance_weight > 0:
            loss = loss + load_balance_weight * load_balancing_loss(logits)
        loss.backward()
        opt.step()


def counterfactual_lift(router: LearnedLatentRouter, probe: RoutingProbe) -> dict[str, float]:
    """Task-metric lift from the router's choice vs uninformed baselines (W3.3).

    The probe's task metric is "did we activate the correct expert?". We compare
    the router's routing accuracy against:

      * **random**: pick an expert uniformly at random (expected = ``chance``);
      * **backbone-only**: activate no expert at all (accuracy 0 — no task is
        solvable without its expert).

    A useful router has positive lift over both.
    """
    report = evaluate_router(router, probe)
    acc = report["routing_accuracy"]
    return {
        "routing_accuracy": acc,
        "lift_vs_random": acc - probe.chance,
        "lift_vs_backbone": acc - 0.0,
    }


def evaluate_router(router: LearnedLatentRouter, probe: RoutingProbe) -> dict[str, float]:
    """Return routing accuracy, ECE, and chance for the router on the probe."""
    router.eval()
    with torch.no_grad():
        out = router.forward(probe.inputs)
        pred_idx = out["weights"].argmax(dim=-1)
        conf = out["confidence"]

    predicted = [probe.expert_ids[int(i)] for i in pred_idx]
    gold = [probe.expert_ids[int(i)] for i in probe.gold_idx]
    acc = routing_accuracy(predicted, gold)

    correct = [bool(p == g) for p, g in zip(pred_idx.tolist(), probe.gold_idx.tolist())]
    ece = expected_calibration_error([float(c) for c in conf.tolist()], correct)
    return {"routing_accuracy": acc, "ece": ece, "chance": probe.chance}


__all__ = [
    "RoutingProbe",
    "build_routing_probe",
    "fit_router",
    "evaluate_router",
    "counterfactual_lift",
]
