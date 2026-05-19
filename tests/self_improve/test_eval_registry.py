"""Tests for the probe-set registry and the per-plugin ``evaluate`` path."""

from __future__ import annotations

import pytest
import torch

from scripts.training.self_improve import EvalRegistry, ProbeSet
from scripts.training.self_improve.eval_registry import (
    build_gaussian_encoder_probe,
    build_hpwm_probe,
    build_lgq_probe,
    build_mmwm_probe,
    build_omnilatent_probe,
)
from scripts.training.self_improve.plugins import available_plugins, get_plugin


def test_default_builders_are_deterministic() -> None:
    a = build_gaussian_encoder_probe(seed=7)
    b = build_gaussian_encoder_probe(seed=7)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_with_defaults_registers_all_components() -> None:
    reg = EvalRegistry.with_defaults()
    for name in available_plugins():
        assert reg.has(name)


def test_registry_rejects_duplicate_components() -> None:
    reg = EvalRegistry()
    reg.register(build_gaussian_encoder_probe())
    with pytest.raises(KeyError):
        reg.register(build_gaussian_encoder_probe())


def test_registry_raises_on_missing_component() -> None:
    reg = EvalRegistry()
    with pytest.raises(KeyError):
        reg.get("gaussian_encoder")


@pytest.mark.parametrize("name", list(available_plugins()))
def test_plugin_evaluate_runs_and_returns_two_metrics(name: str) -> None:
    """Every plugin's ``evaluate`` must return at least two metrics
    (design doc §6.4 reward-hacking resistance) and a finite score."""
    try:
        plugin = get_plugin(name)()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    report = plugin.evaluate()
    assert torch.isfinite(torch.tensor(report.score)), f"{name}: non-finite score"
    assert len(report.metrics) >= 2, (
        f"{name}: evaluate must return ≥2 metrics; got {list(report.metrics)}"
    )
    for k, v in report.metrics.items():
        assert torch.isfinite(torch.tensor(v)), f"{name}: non-finite metric {k}={v}"


@pytest.mark.parametrize("name", list(available_plugins()))
def test_evaluate_is_deterministic_in_eval_mode(name: str) -> None:
    """Same weights + same probe set → identical metrics. This is the
    invariant the vault round-trip test depends on."""
    try:
        plugin = get_plugin(name)()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    r1 = plugin.evaluate()
    r2 = plugin.evaluate()

    assert r1.score == pytest.approx(r2.score, rel=0, abs=1e-6)
    assert set(r1.metrics) == set(r2.metrics)
    for k in r1.metrics:
        assert r1.metrics[k] == pytest.approx(r2.metrics[k], rel=0, abs=1e-6)
