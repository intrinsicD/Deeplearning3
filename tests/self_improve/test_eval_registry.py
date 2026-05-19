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


def test_gaussian_evaluate_uses_plugin_in_ch() -> None:
    """Default probe must match the instantiated plugin's channel count,
    not the builder default. Regression: previously hard-coded in_ch=1.
    """
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    # The underlying gaussian model has spatial dims (7×7) baked in, so
    # ``image_size`` is fixed at 28; the failure mode the review flagged
    # is the channel mismatch when ``in_ch != 1``.
    plugin = GaussianEncoderPlugin(in_ch=3)
    report = plugin.evaluate()
    assert len(report.metrics) >= 2
    for k, v in report.metrics.items():
        assert torch.isfinite(torch.tensor(v)), f"non-finite {k}={v}"


def test_mmwm_evaluate_uses_plugin_config_dims() -> None:
    """Default probe dims must come from the plugin's ModelConfig, not from
    hard-coded builder defaults. Regression: previously crashed on any
    ModelConfig with non-default vector/action dims.
    """
    from MMWM.config import ModelConfig

    MMWMPlugin = get_plugin("mmwm")
    cfg = ModelConfig()
    cfg.encoder_kwargs = dict(cfg.encoder_kwargs)
    cfg.encoder_kwargs["vector_input_dim"] = 64
    cfg.action_encoder_kwargs = dict(cfg.action_encoder_kwargs)
    cfg.action_encoder_kwargs["action_dim"] = 16

    plugin = MMWMPlugin(model_config=cfg)
    report = plugin.evaluate()
    assert len(report.metrics) >= 2
    for k, v in report.metrics.items():
        assert torch.isfinite(torch.tensor(v)), f"non-finite {k}={v}"


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
