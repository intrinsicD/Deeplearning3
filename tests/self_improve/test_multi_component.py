"""Phase 6: multi-component co-training tests.

Acceptance test (design doc):
    "5 components × 10 steps each, all probes pass."

Also covers the bandit scheduler's priority formula now that there's
enough eval signal to exercise it.
"""

from __future__ import annotations

import math

import pytest
import torch

from scripts.training.self_improve import (
    BanditScheduler,
    EvalRegistry,
    Orchestrator,
    SyntheticDataStream,
    Vault,
)
from scripts.training.self_improve.plugins import available_plugins, get_plugin


# ---------------------------------------------------------------------------
# Bandit priority formula
# ---------------------------------------------------------------------------


def _seeded(c: str = "a") -> BanditScheduler:
    """Bandit configured to immediately exit warmup."""
    return BanditScheduler([c, "b"], warmup_rounds=0, seed=0)


def test_regression_gap_higher_is_better() -> None:
    """When higher is better, gap = best - latest, clamped to >=0."""
    s = _seeded()
    s.record_eval("a", 5.0, higher_is_better=True)  # best = 5
    s.record_eval("a", 3.0, higher_is_better=True)  # latest = 3
    assert s._regression_gap("a") == pytest.approx(2.0)


def test_regression_gap_lower_is_better() -> None:
    """When lower is better (e.g. MSE), gap = latest - best, clamped to >=0."""
    s = _seeded()
    s.record_eval("a", 0.1, higher_is_better=False)  # best = 0.1
    s.record_eval("a", 0.5, higher_is_better=False)  # latest = 0.5
    assert s._regression_gap("a") == pytest.approx(0.4)


def test_regression_gap_zero_when_not_regressed() -> None:
    s = _seeded()
    s.record_eval("a", 5.0, higher_is_better=True)
    s.record_eval("a", 6.0, higher_is_better=True)  # new best
    assert s._regression_gap("a") == 0.0


def test_staleness_grows_with_unrun_steps() -> None:
    s = BanditScheduler(["a"], warmup_rounds=0, eval_every=10, seed=0)
    for _ in range(7):
        s.record_step("a")
    assert s._staleness("a") == pytest.approx(0.7)


def test_uncertainty_is_std_of_recent_evals() -> None:
    s = _seeded()
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        s.record_eval("a", v, higher_is_better=True)
    # population std of {1..5} = sqrt(2)
    assert s._uncertainty("a") == pytest.approx(math.sqrt(2.0), rel=1e-4)


def test_worst_quota_routes_to_worst() -> None:
    """With worst_quota=1.0, every post-warmup pick goes to the worst
    component. Verifies the routing branch isolates correctly."""
    s = BanditScheduler(["a", "b"], warmup_rounds=0, worst_quota=1.0, seed=0)
    s.record_eval("a", 0.1, higher_is_better=False)  # best
    s.record_eval("b", 0.9, higher_is_better=False)  # worse (higher MSE)
    picks = [s.pick() for _ in range(20)]
    assert all(p == "b" for p in picks)


def test_priority_picks_highest_when_quota_zero() -> None:
    """With worst_quota=0, every post-warmup pick uses priority."""
    s = BanditScheduler(["a", "b"], warmup_rounds=0, worst_quota=0.0, seed=0)
    # a has a big regression gap; b has no eval history → priority = 0.
    s.record_eval("a", 10.0, higher_is_better=True)
    s.record_eval("a", 0.0, higher_is_better=True)  # huge regression
    picks = [s.pick() for _ in range(10)]
    assert all(p == "a" for p in picks)


def test_falls_back_to_round_robin_with_no_eval_history() -> None:
    s = BanditScheduler(["a", "b"], warmup_rounds=0, worst_quota=0.0, seed=0)
    picks = [s.pick() for _ in range(4)]
    assert picks == ["a", "b", "a", "b"]


def test_rejects_invalid_worst_quota() -> None:
    with pytest.raises(ValueError):
        BanditScheduler(["a"], worst_quota=-0.1)
    with pytest.raises(ValueError):
        BanditScheduler(["a"], worst_quota=1.5)


# ---------------------------------------------------------------------------
# End-to-end: 5 components × 10 steps each
# ---------------------------------------------------------------------------


def test_e2e_all_five_components_round_robin(tmp_path) -> None:
    """The phase-6 acceptance test (design doc):

    "5 components × 10 steps each, all probes pass."

    "Probes pass" here means every component's final eval returns a
    finite score with ≥2 finite metrics (the design doc §6.4 invariant
    that the round-trip test asserted per-plugin in phase 2).
    """
    torch.manual_seed(0)
    plugins = {n: get_plugin(n)() for n in available_plugins()}
    components = list(plugins.keys())

    vault = Vault(tmp_path / "vault")
    orch = Orchestrator(
        plugins,
        scheduler=BanditScheduler(components, warmup_rounds=2, seed=0),
        eval_registry=EvalRegistry.with_defaults(components),
        vault=vault,
        eval_every=10,
        batch_size=1,
    )

    report = orch.run(len(components) * 10)

    # Every component touched at least once.
    assert set(report.steps_per_component) == set(components)
    for c, n in report.steps_per_component.items():
        assert n >= 1, f"{c}: not picked"
    # Total step count matches.
    assert report.total_steps == len(components) * 10

    # Every component evaluated cleanly.
    assert set(report.final_evals) == set(components)
    for c, eval_report in report.final_evals.items():
        assert math.isfinite(eval_report.score), f"{c}: non-finite score"
        assert len(eval_report.metrics) >= 2, (
            f"{c}: <2 metrics in final eval ({list(eval_report.metrics)})"
        )
        for k, v in eval_report.metrics.items():
            assert math.isfinite(v), f"{c}/{k}: non-finite ({v})"


def test_bandit_warmup_distributes_picks_evenly() -> None:
    """During warmup the bandit must visit every component at least
    ``warmup_rounds`` times before applying priority."""
    components = ["a", "b", "c", "d", "e"]
    s = BanditScheduler(components, warmup_rounds=2, seed=0)
    picks = [s.pick() for _ in range(len(components) * 2)]
    counts = {c: picks.count(c) for c in components}
    assert all(counts[c] == 2 for c in components), counts
