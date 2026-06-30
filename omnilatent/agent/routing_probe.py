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


def fit_router_outcome_based(
    router: LearnedLatentRouter,
    probe: RoutingProbe,
    steps: int = 400,
    lr: float = 0.05,
) -> None:
    """Credit assignment **v2** — outcome-based reward, no supervised label.

    Research lane (work plan W5.1). The router never sees the gold expert as a
    target. Instead it *samples* an expert from its routing distribution, and
    the only learning signal is a scalar reward — 1.0 if the sampled expert was
    the correct one for the input (the task "succeeds"), 0.0 otherwise — trained
    with REINFORCE and a running-mean baseline. This is the bandit form of the
    probe-delta reward the harness (`self_improvement.md` §4.6) would supply in
    a real run.
    """
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    baseline = 0.0
    for _ in range(steps):
        opt.zero_grad()
        logits = router.forward(probe.inputs)["logits"]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                                  # (N,) sampled expert
        reward = (action == probe.gold_idx).float()            # scalar outcome only
        advantage = reward - baseline
        loss = -(dist.log_prob(action) * advantage.detach()).mean()
        loss.backward()
        opt.step()
        baseline = 0.9 * baseline + 0.1 * float(reward.mean().item())


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


def ood_abstention_study(
    router: LearnedLatentRouter,
    probe: RoutingProbe,
    ood_scale: float = 6.0,
    seed: int = 7,
) -> dict[str, float]:
    """OOD selection + abstention study (research lane, work plan W5.3).

    Measures whether the router's confidence — the abstention signal (W2.3) —
    actually drops on out-of-distribution input. We build OOD samples far from
    every trained prototype (Gaussian noise at ``ood_scale``×) and compare the
    router's mean confidence on in-distribution vs OOD inputs.

    Honest expectation, reported not assumed: a softmax router tends to stay
    *overconfident* off-distribution, so the ID→OOD confidence drop is usually
    small. The returned ``confidence_gap`` quantifies exactly how small — the
    measurement that tells you whether the abstention threshold can be trusted
    beyond the training distribution.
    """
    gen = torch.Generator().manual_seed(seed)
    ood = ood_scale * torch.randn(probe.inputs.shape[0], probe.input_dim, generator=gen)

    router.eval()
    with torch.no_grad():
        conf_id = float(router.forward(probe.inputs)["confidence"].mean().item())
        conf_ood = float(router.forward(ood)["confidence"].mean().item())
    return {
        "confidence_id": conf_id,
        "confidence_ood": conf_ood,
        "confidence_gap": conf_id - conf_ood,  # >0 ⇒ less confident on OOD (good)
    }


__all__ = [
    "RoutingProbe",
    "build_routing_probe",
    "fit_router",
    "fit_router_outcome_based",
    "evaluate_router",
    "counterfactual_lift",
    "ood_abstention_study",
]
