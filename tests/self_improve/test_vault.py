"""Tests for the content-addressed snapshot vault."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.training.self_improve import (
    ComponentPlugin,
    EvalReport,
    SnapshotID,
    Vault,
)
from scripts.training.self_improve.plugins import get_plugin


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return Vault(tmp_path / "vault")


def _make_gaussian() -> ComponentPlugin:
    return get_plugin("gaussian_encoder")()


def _eval_report(score: float = 0.5) -> EvalReport:
    return EvalReport(score=score, metrics={"mse": score, "l1": score * 0.5})


def test_save_creates_content_addressed_file(vault: Vault) -> None:
    plugin = _make_gaussian()
    sid = vault.save("gaussian_encoder", plugin, _eval_report(), step=0)

    expected = vault.root / "gaussian_encoder" / f"{sid.hash}.pt"
    assert expected.exists()
    assert len(sid.hash) == 64  # sha-256 hex
    assert sid.step == 0
    assert sid.component == "gaussian_encoder"


def test_save_then_load_round_trips_state(vault: Vault) -> None:
    plugin = _make_gaussian()
    report = _eval_report(score=0.123)
    sid = vault.save("gaussian_encoder", plugin, report, step=5)

    snap = vault.load(sid)
    assert snap.id == sid
    assert snap.report.score == pytest.approx(0.123)
    assert snap.report.metrics["mse"] == pytest.approx(0.123)
    assert "model" in snap.state

    # State payload should match what the plugin would produce now.
    fresh_keys = set(plugin.state_dict()["model"].keys())
    assert set(snap.state["model"].keys()) == fresh_keys


def test_identical_state_dedupes_by_hash(vault: Vault) -> None:
    plugin = _make_gaussian()
    sid1 = vault.save("gaussian_encoder", plugin, _eval_report(), step=0)
    sid2 = vault.save("gaussian_encoder", plugin, _eval_report(), step=0)

    # Same content + same step → single index entry.
    assert sid1.hash == sid2.hash
    files = list((vault.root / "gaussian_encoder").glob("*.pt"))
    assert len(files) == 1
    assert len(vault.list("gaussian_encoder")) == 1


def test_best_returns_highest_score_when_higher_is_better(vault: Vault) -> None:
    plugin = _make_gaussian()

    # Three snapshots with different scores. Mutate weights to force
    # distinct hashes.
    sids: list[SnapshotID] = []
    for step, score in [(0, 1.0), (1, 5.0), (2, 3.0)]:
        with torch.no_grad():
            for p in plugin.model.parameters():
                p.add_(0.01)
                break
        sids.append(vault.save(
            "gaussian_encoder",
            plugin,
            EvalReport(score=score, metrics={"x": score}, higher_is_better=True),
            step=step,
        ))

    best = vault.best("gaussian_encoder")
    assert best is not None
    assert best.step == 1  # score=5.0 was the highest


def test_best_returns_lowest_score_when_lower_is_better(vault: Vault) -> None:
    plugin = _make_gaussian()

    for step, score in [(0, 1.0), (1, 0.2), (2, 0.5)]:
        with torch.no_grad():
            for p in plugin.model.parameters():
                p.add_(0.01)
                break
        vault.save(
            "gaussian_encoder",
            plugin,
            EvalReport(score=score, metrics={"mse": score}, higher_is_better=False),
            step=step,
        )

    best = vault.best("gaussian_encoder")
    assert best is not None
    assert best.step == 1  # score=0.2 was the lowest


def test_best_is_none_for_unknown_component(vault: Vault) -> None:
    assert vault.best("nonexistent") is None


def test_rollback_restores_weights(vault: Vault) -> None:
    plugin = _make_gaussian()

    # Capture the weight of a known parameter, then save.
    original = next(plugin.model.parameters()).detach().clone()
    sid = vault.save("gaussian_encoder", plugin, _eval_report(), step=0)

    # Mutate the plugin in-place.
    with torch.no_grad():
        for p in plugin.model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    mutated = next(plugin.model.parameters()).detach().clone()
    assert not torch.equal(original, mutated)

    # Rollback should restore the original weights bit-for-bit.
    vault.rollback("gaussian_encoder", plugin, snapshot_id=sid)
    restored = next(plugin.model.parameters()).detach().clone()
    assert torch.equal(original, restored)


def test_rollback_to_best_when_no_snapshot_id(vault: Vault) -> None:
    plugin = _make_gaussian()

    # Save twice; the second snapshot has a better (lower) score.
    # Both saves declare higher_is_better=False — the vault's convention
    # is fixed per component at first save and is verified by
    # ``test_save_rejects_inconsistent_convention`` below.
    vault.save(
        "gaussian_encoder", plugin,
        EvalReport(score=1.0, metrics={"mse": 1.0}, higher_is_better=False),
        step=0,
    )
    with torch.no_grad():
        for p in plugin.model.parameters():
            p.add_(0.05)
            break
    best_state = next(plugin.model.parameters()).detach().clone()
    vault.save(
        "gaussian_encoder", plugin,
        EvalReport(score=0.1, metrics={"mse": 0.1}, higher_is_better=False),
        step=1,
    )

    # Mutate and roll back; should land on the better (step=1) state.
    with torch.no_grad():
        for p in plugin.model.parameters():
            p.add_(0.5)
            break
    vault.rollback("gaussian_encoder", plugin)
    after = next(plugin.model.parameters()).detach().clone()
    assert torch.equal(best_state, after)


def test_save_rejects_inconsistent_convention(vault: Vault) -> None:
    """Component convention (higher_is_better) is fixed at first save."""
    plugin = _make_gaussian()
    vault.save(
        "gaussian_encoder", plugin,
        EvalReport(score=1.0, metrics={"x": 1.0}, higher_is_better=True),
        step=0,
    )
    with pytest.raises(ValueError, match="higher_is_better"):
        vault.save(
            "gaussian_encoder", plugin,
            EvalReport(score=0.1, metrics={"x": 0.1}, higher_is_better=False),
            step=1,
        )


def test_higher_is_better_query_matches_first_save(vault: Vault) -> None:
    plugin = _make_gaussian()
    vault.save(
        "gaussian_encoder", plugin,
        EvalReport(score=0.5, higher_is_better=False),
        step=0,
    )
    assert vault.higher_is_better("gaussian_encoder") is False


def test_index_persists_across_vault_instances(vault: Vault) -> None:
    plugin = _make_gaussian()
    sid = vault.save("gaussian_encoder", plugin, _eval_report(), step=0)

    # Re-open the vault on the same directory.
    reopened = Vault(vault.root)
    assert reopened.best("gaussian_encoder") == sid
    assert reopened.list("gaussian_encoder") == [sid]
