"""Phase 2 acceptance test: save → mutate → rollback → identical metrics.

This is the single most important guarantee the vault makes: after a
rollback the plugin's eval score and per-metric values must match the
pre-save snapshot exactly (modulo float noise). Without it the
snapshot-and-rollback gate in §4.6 of the design doc cannot do its job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.training.self_improve import EvalRegistry, Vault
from scripts.training.self_improve.plugins import available_plugins, get_plugin


# Some components have inherently noisier eval paths (e.g. MMWM's
# stochastic regularizers fire even in eval mode). The tolerances below
# are tight enough to catch real regressions but loose enough to absorb
# float-summation order on different hardware.
_TOL = {
    "gaussian_encoder": 1e-5,
    "lgq": 1e-4,
    "omnilatent": 1e-4,
    "mmwm": 1e-3,
    "hpwm": 1e-3,
}


@pytest.mark.parametrize("name", list(available_plugins()))
def test_round_trip_identical_metrics(name: str, tmp_path: Path) -> None:
    # Seed at the start so the test is order-independent. Without this,
    # earlier tests in the suite can perturb the global RNG enough that
    # the small-model gaussian_encoder trajectory moves the eval probe
    # by less than ``_TOL["gaussian_encoder"]`` (1e-5), making the
    # "training moved the score" guard vacuous.
    import torch as _torch
    _torch.manual_seed(0)

    try:
        plugin = get_plugin(name)()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    registry = EvalRegistry.with_defaults([name])
    probe = registry.get(name)
    vault = Vault(tmp_path / "vault")

    # 1. Initial eval — this is the score we expect to recover after rollback.
    report_before = plugin.evaluate(probe)

    # 2. Snapshot to vault.
    sid = vault.save(name, plugin, report_before, step=0)

    # 3. Mutate the plugin with several training steps on a fixed batch
    # (overfit slightly so the eval probe — a *different* batch —
    # moves visibly above ``_TOL[name]``). One step is not always
    # enough: e.g. random-data MSE on a tiny autoencoder barely moves.
    train_batch = plugin.make_synthetic_batch()
    for _ in range(5):
        step_report = plugin.train_step(train_batch)
        assert step_report.is_finite()

    # Eval after mutation MUST differ — otherwise the test isn't proving
    # anything about restoration.
    report_after_step = plugin.evaluate(probe)
    assert report_after_step.score != pytest.approx(
        report_before.score, rel=0, abs=_TOL[name],
    ), (
        f"{name}: training did not move the eval score above tol={_TOL[name]}; "
        f"round-trip test is vacuous (before={report_before.score}, "
        f"after={report_after_step.score})"
    )

    # 4. Rollback.
    snap = vault.rollback(name, plugin, snapshot_id=sid)
    assert snap.id == sid

    # 5. Re-evaluate — must match the pre-save numbers.
    report_restored = plugin.evaluate(probe)

    tol = _TOL[name]
    assert report_restored.score == pytest.approx(
        report_before.score, rel=0, abs=tol,
    ), (
        f"{name}: rollback score drifted by "
        f"{abs(report_restored.score - report_before.score):.3e} > tol={tol}"
    )

    assert set(report_restored.metrics) == set(report_before.metrics)
    for k in report_before.metrics:
        assert report_restored.metrics[k] == pytest.approx(
            report_before.metrics[k], rel=0, abs=tol,
        ), (
            f"{name}: metric {k!r} drifted by "
            f"{abs(report_restored.metrics[k] - report_before.metrics[k]):.3e} > tol={tol}"
        )
