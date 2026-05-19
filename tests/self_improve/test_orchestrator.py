"""Tests for the phase-5 orchestrator + scheduler + data stream + CLI.

Acceptance test (design doc phase 5):
    "End-to-end test: 50 steps on HPWM via the orchestrator matches
     direct training within tolerance."

We verify this on a smaller / cheaper plugin
(:class:`GaussianEncoderPlugin`) and a larger one
(:class:`LGQPlugin`), and run a brief CLI smoke test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from scripts.training.self_improve import (
    EvalRegistry,
    Orchestrator,
    RoundRobinScheduler,
    SyntheticDataStream,
    Vault,
)
from scripts.training.self_improve.cli import main as cli_main
from scripts.training.self_improve.plugins import available_plugins, get_plugin
from scripts.training.self_improve.scheduler import BanditScheduler


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


def test_round_robin_cycles() -> None:
    sched = RoundRobinScheduler(["a", "b", "c"])
    picks = [sched.pick() for _ in range(7)]
    assert picks == ["a", "b", "c", "a", "b", "c", "a"]


def test_round_robin_rejects_empty() -> None:
    with pytest.raises(ValueError):
        RoundRobinScheduler([])


def test_bandit_warmup_is_round_robin() -> None:
    sched = BanditScheduler(["a", "b"], warmup_rounds=2)
    picks = [sched.pick() for _ in range(6)]
    assert picks[:4] == ["a", "b", "a", "b"]


def test_bandit_records_best_under_higher_is_better() -> None:
    sched = BanditScheduler(["a"])
    sched.record_eval("a", 1.0, higher_is_better=True)
    sched.record_eval("a", 0.5, higher_is_better=True)
    sched.record_eval("a", 2.0, higher_is_better=True)
    assert sched.best["a"] == 2.0
    assert sched.latest["a"] == 2.0


def test_bandit_records_best_under_lower_is_better() -> None:
    sched = BanditScheduler(["a"])
    sched.record_eval("a", 1.0, higher_is_better=False)
    sched.record_eval("a", 0.5, higher_is_better=False)
    sched.record_eval("a", 2.0, higher_is_better=False)
    assert sched.best["a"] == 0.5


# ---------------------------------------------------------------------------
# Orchestrator parity test
# ---------------------------------------------------------------------------


def _run_orchestrator_path(steps: int = 30) -> list[float]:
    """Train gaussian_encoder for ``steps`` via the orchestrator."""
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()
    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        data_stream=SyntheticDataStream(batch_size=2),
        eval_every=0,  # disable eval so the trajectories don't diverge
        seed=0,
    )
    report = orch.run(steps)
    return report.step_losses["gaussian_encoder"]


def _run_direct_path(steps: int = 30) -> list[float]:
    """Same plugin, same seed, same batches — but called directly."""
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()
    stream = SyntheticDataStream(batch_size=2)
    losses = []
    for _ in range(steps):
        batch = stream.next(plugin, batch_size=2)
        report = plugin.train_step(batch)
        losses.append(report.loss)
    return losses


def test_orchestrator_matches_direct_training() -> None:
    """The phase-5 acceptance claim: routing a single-component run
    through the orchestrator produces the same loss trajectory as
    calling ``plugin.train_step`` directly.

    Same seed, same data stream, same number of steps. Tolerance is
    very tight; if the orchestrator ever introduced a hidden side
    effect (extra RNG calls, missing optimizer state, etc.) this test
    would catch it.
    """
    orch_losses = _run_orchestrator_path(steps=30)
    direct_losses = _run_direct_path(steps=30)

    assert len(orch_losses) == len(direct_losses) == 30
    for i, (a, b) in enumerate(zip(orch_losses, direct_losses)):
        assert a == pytest.approx(b, rel=0, abs=1e-7), (
            f"step {i}: orchestrator={a} direct={b}"
        )


def test_orchestrator_matches_direct_training_hpwm() -> None:
    """The literal phase-5 acceptance line from the design doc:

        "End-to-end test: 50 steps on HPWM via the orchestrator
         matches direct training within tolerance."

    Same construction protocol as the gaussian_encoder parity test;
    we just substitute HPWM. HPWM has a lazy DINO backbone that
    materializes on the first forward, so the orchestrator and direct
    paths both pay the same one-time RNG cost.
    """
    torch.manual_seed(0)
    HPWMPlugin = get_plugin("hpwm")
    plugin_orch = HPWMPlugin()
    orch = Orchestrator(
        {"hpwm": plugin_orch},
        scheduler=RoundRobinScheduler(["hpwm"]),
        data_stream=SyntheticDataStream(batch_size=1),
        eval_every=0,
        batch_size=1,
        seed=0,
    )
    orch_losses = orch.run(50).step_losses["hpwm"]

    torch.manual_seed(0)
    plugin_direct = HPWMPlugin()
    stream = SyntheticDataStream(batch_size=1)
    direct_losses = []
    for _ in range(50):
        report = plugin_direct.train_step(stream.next(plugin_direct, batch_size=1))
        direct_losses.append(report.loss)

    # HPWM has stochastic temporal-state initialization; a tiny FP
    # tolerance is reasonable.
    assert len(orch_losses) == len(direct_losses) == 50
    for i, (a, b) in enumerate(zip(orch_losses, direct_losses)):
        assert a == pytest.approx(b, rel=0, abs=1e-5), (
            f"step {i}: orchestrator={a} direct={b}"
        )


# ---------------------------------------------------------------------------
# Orchestrator + vault end-to-end
# ---------------------------------------------------------------------------


def test_orchestrator_evaluates_and_snapshots(tmp_path: Path) -> None:
    """With ``eval_every`` set and a vault attached, the orchestrator
    saves at least one snapshot per component over the course of the run.
    """
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()

    vault = Vault(tmp_path / "vault")
    eval_registry = EvalRegistry.with_defaults(["gaussian_encoder"])

    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_registry=eval_registry,
        vault=vault,
        eval_every=5,
        seed=0,
    )
    report = orch.run(20)

    # 20 steps, eval every 5 → at least 4 evals → ≥ 1 snapshot.
    snapshots = vault.list("gaussian_encoder")
    assert len(snapshots) >= 1
    # Final eval was computed and stored on the run report.
    assert "gaussian_encoder" in report.final_evals


def test_orchestrator_records_rollback_on_regression(tmp_path: Path) -> None:
    """Manufacture a regression: force the plugin's eval to return a
    bad score after a good one, and check that the orchestrator rolls
    back and increments the rollback counter.
    """
    torch.manual_seed(0)
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()

    # Train a few steps so the model has a non-trivial state to roll back to.
    stream = SyntheticDataStream(batch_size=2)
    for _ in range(5):
        plugin.train_step(stream.next(plugin, batch_size=2))

    # Manually seed the vault with a "good" snapshot (the current state).
    vault = Vault(tmp_path / "vault")
    from scripts.training.self_improve import EvalReport
    vault.save(
        "gaussian_encoder",
        plugin,
        EvalReport(score=0.001, metrics={"mse": 0.001, "l1": 0.01}, higher_is_better=False),
        step=5,
    )

    # Now corrupt the plugin so its next eval will score worse, and run
    # the orchestrator for one eval cycle.
    with torch.no_grad():
        for p in plugin.model.parameters():
            p.add_(torch.ones_like(p))  # massive perturbation

    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_registry=EvalRegistry.with_defaults(["gaussian_encoder"]),
        vault=vault,
        eval_every=1,  # eval after every step
        seed=0,
    )
    report = orch.run(1)
    assert report.rollbacks["gaussian_encoder"] >= 1


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_dry_run_runs(caplog) -> None:
    """``--dry-run`` builds the plugins, evaluates each once, and exits 0."""
    caplog.set_level(logging.INFO)
    rc = cli_main([
        "--components", "gaussian_encoder",
        "--dry-run",
        "--seed", "0",
    ])
    assert rc == 0
    assert any("dry-run eval" in r.message for r in caplog.records)


def test_cli_runs_a_few_steps(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    rc = cli_main([
        "--components", "gaussian_encoder",
        "--max-steps", "4",
        "--eval-every", "2",
        "--batch-size", "2",
        "--vault-root", str(tmp_path / "vault"),
        "--seed", "0",
    ])
    assert rc == 0
    # Final eval was logged.
    assert any("final eval" in r.message for r in caplog.records)


def test_cli_rejects_unknown_component() -> None:
    with pytest.raises(SystemExit):
        cli_main(["--components", "definitely-not-a-plugin"])
