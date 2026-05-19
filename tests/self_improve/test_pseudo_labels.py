"""Tests for the phase-7 pseudo-label broker.

Design-doc verification lines:

(a) labels flow with cyclic edges enabled and guards firing
(b) --strict-acyclic rejects the default cyclic config and accepts the
    DAG variant
(c) divergence guard severs a synthetically-poisoned edge and auto-heals
    after recovery

These don't require running the actual five plugins together; we use
toy producer/consumer plugins so the broker mechanics are isolated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from scripts.training.self_improve import (
    ComponentPlugin,
    EdgeConfig,
    EvalReport,
    PseudoLabelBroker,
    StepReport,
    Vault,
)
from scripts.training.self_improve.pseudo_labels import (
    detect_cycles,
    enforce_acyclic_or_raise,
    validate_cyclic_edges_have_guards,
)


# ---------------------------------------------------------------------------
# Toy plugin
# ---------------------------------------------------------------------------


class _ToyPlugin(ComponentPlugin):
    """One-parameter trainable plugin for broker tests.

    ``model`` is a single learnable scalar; ``label_fn`` users can read
    the scalar to produce a label.
    """

    def __init__(self, name: str, init_value: float = 0.0) -> None:
        self._name = name
        self._param = nn.Parameter(torch.tensor(float(init_value)))
        self._model = nn.ParameterList([self._param])
        self._device = torch.device("cpu")

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def make_synthetic_batch(self, batch_size: int = 2) -> torch.Tensor:
        return torch.zeros(batch_size)

    def train_step(self, batch: Any) -> StepReport:
        return StepReport(loss=0.0)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def test_detect_cycles_empty_graph() -> None:
    assert detect_cycles([]) == []


def test_detect_cycles_dag() -> None:
    edges = [
        EdgeConfig(producer="a", consumer="b"),
        EdgeConfig(producer="b", consumer="c"),
        EdgeConfig(producer="a", consumer="c"),
    ]
    assert detect_cycles(edges) == []


def test_detect_cycles_two_cycle() -> None:
    edges = [
        EdgeConfig(producer="a", consumer="b"),
        EdgeConfig(producer="b", consumer="a"),
    ]
    cycles = detect_cycles(edges)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b"}


def test_detect_cycles_three_cycle() -> None:
    edges = [
        EdgeConfig(producer="a", consumer="b"),
        EdgeConfig(producer="b", consumer="c"),
        EdgeConfig(producer="c", consumer="a"),
    ]
    cycles = detect_cycles(edges)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}


def test_design_doc_default_graph_is_cyclic() -> None:
    """The design doc §5 producer table contains HPWM<->OmniLatent and
    OmniLatent<->MMWM cycles. Verify that detect_cycles flags them."""
    edges = [
        EdgeConfig(producer="lgq", consumer="hpwm"),
        EdgeConfig(producer="lgq", consumer="mmwm"),
        EdgeConfig(producer="hpwm", consumer="omnilatent"),
        EdgeConfig(producer="omnilatent", consumer="mmwm"),
        EdgeConfig(producer="omnilatent", consumer="hpwm"),
        EdgeConfig(producer="mmwm", consumer="omnilatent"),
        EdgeConfig(producer="gaussian_encoder", consumer="lgq"),
    ]
    cycles = detect_cycles(edges)
    flat = {n for cycle in cycles for n in cycle}
    # mmwm, omnilatent, hpwm participate in a cycle (the doc's
    # cross-modal anchoring loop).
    assert "omnilatent" in flat
    assert "hpwm" in flat or "mmwm" in flat


def test_design_doc_dag_variant_is_acyclic() -> None:
    """The §5.2 rule-6 ``--strict-acyclic`` DAG variant."""
    edges = [
        EdgeConfig(producer="gaussian_encoder", consumer="lgq"),
        EdgeConfig(producer="lgq", consumer="hpwm"),
        EdgeConfig(producer="lgq", consumer="mmwm"),
        EdgeConfig(producer="hpwm", consumer="omnilatent"),
        EdgeConfig(producer="omnilatent", consumer="mmwm"),
    ]
    assert detect_cycles(edges) == []


# ---------------------------------------------------------------------------
# --strict-acyclic startup check
# ---------------------------------------------------------------------------


def test_strict_acyclic_accepts_dag() -> None:
    enforce_acyclic_or_raise([
        EdgeConfig(producer="a", consumer="b"),
        EdgeConfig(producer="b", consumer="c"),
    ])


def test_strict_acyclic_rejects_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        enforce_acyclic_or_raise([
            EdgeConfig(producer="a", consumer="b"),
            EdgeConfig(producer="b", consumer="a"),
        ])


def test_validate_cyclic_edges_accepts_dag() -> None:
    """The non-strict startup check is a no-op when the graph is acyclic."""
    validate_cyclic_edges_have_guards([
        EdgeConfig(producer="a", consumer="b"),
        EdgeConfig(producer="b", consumer="c"),
    ])


def test_validate_cyclic_edges_requires_min_stale_steps() -> None:
    """Cyclic edges must declare min_stale_steps >= floor."""
    with pytest.raises(ValueError, match="min_stale_steps"):
        validate_cyclic_edges_have_guards(
            [
                EdgeConfig(producer="a", consumer="b", min_stale_steps=10),
                EdgeConfig(producer="b", consumer="a", min_stale_steps=10),
            ],
            min_stale_steps_floor=100,
        )


# ---------------------------------------------------------------------------
# Snapshot-mediated reads + staleness budget
# ---------------------------------------------------------------------------


def _build_broker(tmp_path: Path) -> tuple[PseudoLabelBroker, Vault, _ToyPlugin, _ToyPlugin]:
    vault = Vault(tmp_path / "vault")
    prod = _ToyPlugin("prod", init_value=1.0)
    cons = _ToyPlugin("cons", init_value=0.0)
    broker = PseudoLabelBroker(vault)
    broker.register_producer("prod", prod)
    return broker, vault, prod, cons


def test_no_labels_when_producer_has_no_snapshot(tmp_path: Path) -> None:
    broker, _, _, _ = _build_broker(tmp_path)
    broker.register_edge(
        EdgeConfig("prod", "cons", min_stale_steps=0, confidence_threshold=0.0),
        label_fn=lambda x, p: (x, 1.0),
    )
    broker.finalize()
    out = broker.sample("cons", [1, 2, 3])
    assert out == []


def test_labels_use_snapshot_not_live_weights(tmp_path: Path) -> None:
    """Snapshot-mediated read: producer's *current* weights have drifted
    far from the snapshot, but labels must reflect the snapshot."""
    broker, vault, prod, _ = _build_broker(tmp_path)

    # Save a snapshot with producer value = 10.
    with torch.no_grad():
        prod._param.fill_(10.0)
    vault.save(
        "prod", prod,
        EvalReport(score=1.0, metrics={"a": 1.0, "b": 1.0}, higher_is_better=True),
        step=0,
    )

    # Now mutate the live producer to value = 999.
    with torch.no_grad():
        prod._param.fill_(999.0)

    def _label(_sample: Any, p: _ToyPlugin) -> tuple[float, float]:
        return float(p._param.item()), 1.0

    broker.register_edge(
        EdgeConfig("prod", "cons", min_stale_steps=0, confidence_threshold=0.0),
        label_fn=_label,
    )
    broker.finalize()
    batch = broker.sample("cons", [0.0, 0.0])
    assert len(batch) == 1
    # Every label should be 10 (the snapshot value), not 999 (live).
    assert all(v == 10.0 for v in batch[0].labels)
    # Producer must be restored to live weights after the call.
    assert float(prod._param.item()) == 999.0


def test_staleness_budget_holds_snapshot_until_steps_elapse(tmp_path: Path) -> None:
    """``min_stale_steps`` prevents refresh until enough consumer steps."""
    broker, vault, prod, _ = _build_broker(tmp_path)
    # Save snapshot v1.
    with torch.no_grad():
        prod._param.fill_(1.0)
    vault.save("prod", prod, EvalReport(score=1.0, metrics={"a": 1.0, "b": 1.0}, higher_is_better=True), step=0)

    broker.register_edge(
        EdgeConfig("prod", "cons", min_stale_steps=5, confidence_threshold=0.0),
        label_fn=lambda x, p: (float(p._param.item()), 1.0),
    )
    broker.finalize()
    batch_v1 = broker.sample("cons", [0])
    snap_v1 = batch_v1[0].producer_snapshot

    # Save a newer snapshot v2 (better score).
    with torch.no_grad():
        prod._param.fill_(2.0)
    vault.save("prod", prod, EvalReport(score=2.0, metrics={"a": 2.0, "b": 2.0}, higher_is_better=True), step=10)

    # Without consumer steps, the broker must NOT refresh.
    batch_still_v1 = broker.sample("cons", [0])
    assert batch_still_v1[0].producer_snapshot == snap_v1
    assert broker.refresh_count("prod", "cons") == 1

    # After ``min_stale_steps`` consumer steps, refresh fires.
    for _ in range(5):
        broker.record_consumer_step("cons")
    batch_v2 = broker.sample("cons", [0])
    assert batch_v2[0].producer_snapshot != snap_v1
    assert broker.refresh_count("prod", "cons") == 2


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------


def test_confidence_gating_drops_low_confidence_labels(tmp_path: Path) -> None:
    broker, vault, prod, _ = _build_broker(tmp_path)
    vault.save("prod", prod, EvalReport(score=1.0, metrics={"a": 1.0, "b": 1.0}, higher_is_better=True), step=0)

    broker.register_edge(
        EdgeConfig("prod", "cons", min_stale_steps=0, confidence_threshold=0.7),
        # Confidence depends on the sample value.
        label_fn=lambda s, p: (s, float(s)),
    )
    broker.finalize()
    batch = broker.sample("cons", [0.1, 0.5, 0.8, 0.95])
    # Only confidences >= 0.7 survive.
    assert batch[0].confidences == [0.8, 0.95]
    assert batch[0].labels == [0.8, 0.95]


# ---------------------------------------------------------------------------
# Divergence guard
# ---------------------------------------------------------------------------


def test_sever_stops_emitting_labels(tmp_path: Path) -> None:
    broker, vault, prod, _ = _build_broker(tmp_path)
    vault.save("prod", prod, EvalReport(score=1.0, metrics={"a": 1.0, "b": 1.0}, higher_is_better=True), step=0)
    broker.register_edge(
        EdgeConfig("prod", "cons", min_stale_steps=0, confidence_threshold=0.0),
        label_fn=lambda x, p: (x, 1.0),
    )
    broker.finalize()
    assert broker.sample("cons", [0]) != []
    broker.mark_edge_severed("prod", "cons")
    assert broker.is_severed("prod", "cons")
    assert broker.sample("cons", [0]) == []


def test_auto_heal_after_clean_steps(tmp_path: Path) -> None:
    """Once severed, the broker auto-heals after ``heal_after_steps``."""
    broker, vault, prod, _ = _build_broker(tmp_path)
    vault.save("prod", prod, EvalReport(score=1.0, metrics={"a": 1.0, "b": 1.0}, higher_is_better=True), step=0)
    broker.register_edge(
        EdgeConfig(
            "prod", "cons",
            min_stale_steps=0,
            confidence_threshold=0.0,
            heal_after_steps=3,
        ),
        label_fn=lambda x, p: (x, 1.0),
    )
    broker.finalize()
    broker.mark_edge_severed("prod", "cons")
    assert broker.is_severed("prod", "cons")

    # After 3 clean consumer steps, edge heals.
    for _ in range(3):
        broker.record_consumer_step("cons")
    assert not broker.is_severed("prod", "cons")
    assert broker.sample("cons", [0]) != []


# ---------------------------------------------------------------------------
# Broker finalization rejects misconfigurations
# ---------------------------------------------------------------------------


def test_strict_acyclic_broker_rejects_cyclic_config(tmp_path: Path) -> None:
    broker = PseudoLabelBroker(Vault(tmp_path / "vault"), strict_acyclic=True)
    broker.register_edge(
        EdgeConfig("a", "b", min_stale_steps=1000, confidence_threshold=0.5),
        label_fn=lambda x, p: (x, 1.0),
    )
    broker.register_edge(
        EdgeConfig("b", "a", min_stale_steps=1000, confidence_threshold=0.5),
        label_fn=lambda x, p: (x, 1.0),
    )
    with pytest.raises(ValueError, match="cycle"):
        broker.finalize()


def test_strict_acyclic_broker_accepts_dag(tmp_path: Path) -> None:
    broker = PseudoLabelBroker(Vault(tmp_path / "vault"), strict_acyclic=True)
    broker.register_edge(
        EdgeConfig("a", "b"),
        label_fn=lambda x, p: (x, 1.0),
    )
    broker.finalize()


def test_non_strict_broker_requires_guards_on_cyclic_edges(tmp_path: Path) -> None:
    broker = PseudoLabelBroker(Vault(tmp_path / "vault"), strict_acyclic=False)
    broker.register_edge(
        EdgeConfig("a", "b", min_stale_steps=10),  # below floor
        label_fn=lambda x, p: (x, 1.0),
    )
    broker.register_edge(
        EdgeConfig("b", "a", min_stale_steps=10),
        label_fn=lambda x, p: (x, 1.0),
    )
    with pytest.raises(ValueError, match="min_stale_steps"):
        broker.finalize()


def test_self_edge_rejected(tmp_path: Path) -> None:
    broker = PseudoLabelBroker(Vault(tmp_path / "vault"))
    with pytest.raises(ValueError, match="self-edge"):
        broker.register_edge(
            EdgeConfig("a", "a"),
            label_fn=lambda x, p: (x, 1.0),
        )
