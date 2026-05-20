"""Tests for the orchestrator's capacity-expansion + compute-budget wiring.

Section A of the followup work:

- A plateau in the eval trajectory triggers the configured
  ``ExpansionAction``; for an OmniLatent component the default action
  registers a new hook.
- ``compute_budget`` (per-component or single cap) pauses a component's
  training once its persistent step count reaches the cap; the
  scheduler can still pick it but the orchestrator no longer trains
  on it. The component appears in ``RunReport.budget_exhausted``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from scripts.training.self_improve import (
    EvalRegistry,
    EvalReport,
    Orchestrator,
    RoundRobinScheduler,
    SyntheticDataStream,
)
from scripts.training.self_improve.plugins import get_plugin


# ---------------------------------------------------------------------------
# Plateau-driven expansion
# ---------------------------------------------------------------------------


class _FixedScoreOmniLatent:
    """OmniLatentPlugin shim that returns a controllable eval score.

    Wraps a real plugin so the orchestrator's eval path works
    end-to-end (train_step + state_dict + evaluate signatures unchanged)
    but the eval score we read is whatever the test queue holds. This
    lets us manufacture a "flat" eval trajectory deterministically.
    """

    def __init__(self) -> None:
        cls = get_plugin("omnilatent")
        self._inner = cls()
        self.name = self._inner.name
        # Each call to evaluate() pops one score from this list; once
        # exhausted, the last score is repeated (so a manufactured
        # plateau can extend indefinitely).
        self.scores: list[float] = []

    @property
    def model(self):
        return self._inner.model

    @property
    def device(self):
        return self._inner.device

    @property
    def ema_teacher(self):
        return self._inner.ema_teacher

    @property
    def replay_bank(self):
        return self._inner.replay_bank

    @property
    def ewc(self):
        return self._inner.ewc

    @property
    def config(self):
        return self._inner.config

    @property
    def _trainer(self):
        return self._inner._trainer

    @_trainer.setter
    def _trainer(self, value):
        self._inner._trainer = value

    def make_synthetic_batch(self, batch_size: int = 2) -> Any:
        return self._inner.make_synthetic_batch(batch_size=batch_size)

    def train_step(self, batch: Any):
        return self._inner.train_step(batch)

    def evaluate(self, probe_set: Any = None) -> EvalReport:
        if self.scores:
            score = self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]
        else:
            score = 1.0
        return EvalReport(
            score=score, metrics={"a": score, "b": score}, higher_is_better=False,
        )

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state):
        self._inner.load_state_dict(state)


def test_plateau_triggers_expansion_on_omnilatent() -> None:
    """A flat eval trajectory fires the default ``ExpansionAction``,
    which registers a new ``LatentNeuralHook`` on the OmniLatent model.

    Trajectory: 6 flat scores at eval_every=1 ⇒ detector sees 2
    consecutive windows of 3 flat values ⇒ triggers after the 6th eval.
    """
    torch.manual_seed(0)
    plugin = _FixedScoreOmniLatent()
    # Flat trajectory ⇒ plateau.
    plugin.scores = [0.5] * 8

    pre_hooks = set(plugin.model.hook_manager.hooks.keys())

    orch = Orchestrator(
        {"omnilatent": plugin},
        scheduler=RoundRobinScheduler(["omnilatent"]),
        eval_every=1,
        plateau_config={"patience": 3, "tol": 1e-3},
    )
    report = orch.run(8)

    # At least one expansion fired.
    assert report.expansions["omnilatent"] >= 1, report.expansions
    # And it actually registered a hook.
    post_hooks = set(plugin.model.hook_manager.hooks.keys())
    assert post_hooks - pre_hooks, "no new hook on the model after expansion"


def test_plateau_does_not_trigger_when_improving() -> None:
    """A strictly improving (lower-is-better) trajectory must not fire
    expansion."""
    torch.manual_seed(0)
    plugin = _FixedScoreOmniLatent()
    # Monotone decreasing — the lower-is-better convention's "better".
    plugin.scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    pre_hooks = set(plugin.model.hook_manager.hooks.keys())
    orch = Orchestrator(
        {"omnilatent": plugin},
        scheduler=RoundRobinScheduler(["omnilatent"]),
        eval_every=1,
        plateau_config={"patience": 3, "tol": 1e-3},
    )
    report = orch.run(8)
    assert report.expansions["omnilatent"] == 0
    post_hooks = set(plugin.model.hook_manager.hooks.keys())
    assert post_hooks == pre_hooks


def test_custom_expansion_action() -> None:
    """The caller can override the default action, e.g. to wire HPWM
    or MMWM expansion paths (not implemented in core yet but the hook
    must be available to extension code)."""
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()

    calls: list[str] = []

    def custom_action(p, _orch) -> bool:
        calls.append(p.name)
        return True

    # Manufacture a plateau on gaussian_encoder.
    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_registry=EvalRegistry.with_defaults(["gaussian_encoder"]),
        eval_every=1,
        plateau_config={"patience": 2, "tol": 0.0},
        expansion_action=custom_action,
    )
    orch.run(4)
    # We don't insist on exact count; just that the custom action ran.
    assert calls, "custom expansion_action was never invoked"
    assert all(name == "gaussian_encoder" for name in calls)


def test_default_action_skips_non_omnilatent_plugins() -> None:
    """The default action is a no-op for non-OmniLatent plugins; we
    don't want to crash a Gaussian or LGQ run just because its eval
    happened to plateau."""
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()

    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_registry=EvalRegistry.with_defaults(["gaussian_encoder"]),
        eval_every=1,
        plateau_config={"patience": 2, "tol": 0.0},
    )
    report = orch.run(6)
    # No crash, and no expansion counted.
    assert report.expansions["gaussian_encoder"] == 0


# ---------------------------------------------------------------------------
# Compute budget (§6.2)
# ---------------------------------------------------------------------------


def test_compute_budget_int_applies_to_every_component() -> None:
    """``compute_budget=N`` caps every component at N steps."""
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()

    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_every=0,
        compute_budget=3,
    )
    report = orch.run(10)
    assert report.steps_per_component["gaussian_encoder"] == 3
    assert orch._steps_per_component["gaussian_encoder"] == 3
    assert "gaussian_encoder" in report.budget_exhausted


def test_compute_budget_per_component_dict() -> None:
    """Mixed per-component budgets work; unlisted components stay
    unbounded."""
    torch.manual_seed(0)
    plugins = {
        "gaussian_encoder": get_plugin("gaussian_encoder")(),
        "lgq": get_plugin("lgq")(),
    }
    orch = Orchestrator(
        plugins,
        scheduler=RoundRobinScheduler(list(plugins)),
        eval_every=0,
        compute_budget={"gaussian_encoder": 2},
    )
    report = orch.run(10)
    # gaussian_encoder capped at 2; lgq sees the rest of its turns.
    assert report.steps_per_component["gaussian_encoder"] == 2
    assert report.steps_per_component["lgq"] >= 3
    assert report.budget_exhausted == ["gaussian_encoder"]


def test_compute_budget_persists_across_runs() -> None:
    """A component that hit its cap on run #1 stays capped on run #2."""
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()
    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_every=0,
        compute_budget=2,
    )
    first = orch.run(3)
    assert first.steps_per_component["gaussian_encoder"] == 2
    second = orch.run(3)
    assert second.steps_per_component["gaussian_encoder"] == 0, (
        "budget-exhausted component still trained on a subsequent run"
    )
    assert "gaussian_encoder" in second.budget_exhausted


def test_compute_budget_rejects_unknown_component() -> None:
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()
    with pytest.raises(KeyError):
        Orchestrator(
            {"gaussian_encoder": plugin},
            compute_budget={"not_a_plugin": 5},
        )


def test_no_compute_budget_is_unbounded() -> None:
    """Without ``compute_budget``, all components run unbounded — the
    pre-existing behavior."""
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()
    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_every=0,
    )
    report = orch.run(5)
    assert report.steps_per_component["gaussian_encoder"] == 5
    assert report.budget_exhausted == []
