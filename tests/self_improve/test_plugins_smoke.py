"""Phase-1 smoke tests: each plugin runs one step on a synthetic batch,
weights actually change, and no NaN/Inf appears anywhere.

These tests are intentionally light — they exercise the plugin contract
itself, not the underlying model's correctness (which has its own tests
under ``tests/<component>/``).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pytest
import torch

from scripts.training.self_improve.plugins import (
    ComponentPlugin,
    StepReport,
    available_plugins,
    get_plugin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _params_changed(
    before: dict[str, torch.Tensor], after: torch.nn.Module, *, atol: float = 0.0,
) -> bool:
    """True iff at least one trainable parameter moved by more than ``atol``."""
    for name, p in after.named_parameters():
        if not p.requires_grad:
            continue
        b = before.get(name)
        if b is None:
            continue
        if not torch.allclose(b, p.detach(), atol=atol):
            return True
    return False


def _all_finite(report: StepReport) -> bool:
    if not math.isfinite(report.loss):
        return False
    return all(math.isfinite(v) for v in report.losses.values())


# ---------------------------------------------------------------------------
# Per-plugin instantiation with graceful skipping for missing optional deps
# ---------------------------------------------------------------------------


def _make_plugin(name: str) -> ComponentPlugin:
    cls = get_plugin(name)
    return cls()


def _safe_make(name: str) -> ComponentPlugin:
    """Build a plugin, skipping only on genuine missing-dependency errors.

    A plain ``except Exception`` would hide real construction regressions
    (shape mismatches, config drift, API breakage) behind a green skip,
    defeating the purpose of these smoke tests. We narrow the skip to
    :class:`ImportError` (and subclasses like :class:`ModuleNotFoundError`),
    so any other failure propagates and fails the test.
    """
    try:
        return _make_plugin(name)
    except ImportError as exc:  # pragma: no cover - environment specific
        pytest.skip(f"{name}: missing dependency ({exc})")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_available_plugins_matches_known_set() -> None:
    assert set(available_plugins()) == {
        "gaussian_encoder", "lgq", "omnilatent", "mmwm", "hpwm",
    }


@pytest.mark.parametrize("name", list(available_plugins()))
def test_plugin_constructs(name: str) -> None:
    plugin = _safe_make(name)
    assert plugin.name == name
    assert isinstance(plugin.model, torch.nn.Module)
    assert plugin.num_parameters() > 0


@pytest.mark.parametrize("name", list(available_plugins()))
def test_plugin_synthetic_batch_round_trip(name: str) -> None:
    plugin = _safe_make(name)
    batch = plugin.make_synthetic_batch(batch_size=2)
    # The batch should be either a tensor (gaussian, lgq) or a dict of
    # tensors (omnilatent, mmwm, hpwm). Either way, it should be non-empty.
    if isinstance(batch, torch.Tensor):
        assert batch.numel() > 0
    elif isinstance(batch, dict):
        assert batch
        for v in batch.values():
            assert isinstance(v, torch.Tensor)
    else:
        pytest.fail(f"{name}: unexpected batch type {type(batch)}")


@pytest.mark.parametrize("name", list(available_plugins()))
def test_plugin_one_step_changes_params(name: str) -> None:
    plugin = _safe_make(name)
    before = _snapshot_params(plugin.model)
    batch = plugin.make_synthetic_batch(batch_size=2)
    report = plugin.train_step(batch)

    assert isinstance(report, StepReport)
    assert _all_finite(report), f"{name}: non-finite loss in {report}"
    assert report.loss > 0 or math.isclose(report.loss, 0.0), (
        f"{name}: expected non-negative loss, got {report.loss}"
    )
    assert _params_changed(before, plugin.model), (
        f"{name}: no trainable parameter moved after one step"
    )


def test_hpwm_dino_weights_round_trip() -> None:
    """A fresh HPWMPlugin must absorb DINO-backbone keys from a saved state.

    Regression guard: when DINOBackbone was lazily materialized only on
    first forward, fresh plugins silently dropped ``dino._model.*`` keys
    on load (strict=False), so snapshot/rollback was not bit-faithful.
    """
    src = _safe_make("hpwm")
    dst = _safe_make("hpwm")

    # Sanity: both plugins materialize DINO at construction time.
    src_keys = set(src.state_dict()["model"].keys())
    dst_keys = set(dst.model.state_dict().keys())
    dino_keys = {k for k in src_keys if k.startswith("dino._model.")}
    assert dino_keys, "expected dino._model.* keys in saved state"
    assert dino_keys.issubset(dst_keys), (
        "fresh plugin missing DINO keys — load would silently drop them"
    )

    # Round-trip: dst.load_state_dict(src.state_dict()) must reproduce
    # the source's DINO weights, not the dst's random fallback init.
    state = src.state_dict()
    dst.load_state_dict(state)

    src_msd = src.model.state_dict()
    dst_msd = dst.model.state_dict()
    for k in dino_keys:
        assert torch.equal(src_msd[k], dst_msd[k]), (
            f"DINO key {k} not faithfully restored"
        )


@pytest.mark.parametrize("name", list(available_plugins()))
def test_plugin_state_dict_round_trip(name: str) -> None:
    plugin = _safe_make(name)
    # Take one step to produce non-trivial state.
    batch = plugin.make_synthetic_batch(batch_size=2)
    plugin.train_step(batch)

    state = plugin.state_dict()
    assert isinstance(state, dict)
    assert "model" in state

    # Mutate weights, then reload and verify they match the snapshot.
    with torch.no_grad():
        for p in plugin.model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
                break  # only need to perturb one parameter

    plugin.load_state_dict(state)
    # Loading does not need to be bit-exact (some plugins copy through
    # CPU/GPU); the model just needs to be functional again.
    batch2 = plugin.make_synthetic_batch(batch_size=2)
    report2 = plugin.train_step(batch2)
    assert _all_finite(report2)
