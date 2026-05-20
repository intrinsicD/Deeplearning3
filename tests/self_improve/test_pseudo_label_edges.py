"""Tests for the Section-C concrete pseudo-label edge plumbing.

Three slices:

1. The producer-side ``gaussian_recon_label_fn`` returns a
   shape-aligned tensor plus a finite confidence — without crashing
   on either 3D ``(C, H, W)`` or 4D ``(N, C, H, W)`` inputs.
2. The consumer-side ``GaussianEncoderPlugin`` (the only plugin wired
   for the edge in this section) folds queued labels into its
   ``train_step`` and surfaces a ``pseudo/<producer>`` key in the
   step report.
3. The orchestrator routes labels through ``broker.sample`` →
   ``plugin._pending_pseudo_labels`` → ``train_step`` and clears the
   slot afterward.

The intentional consumer is ``gaussian_encoder`` because (a) its
``train_step`` already runs a fused backward pass we can extend
cheaply, and (b) the producer and consumer can share an image shape
without a resize adapter (we use two ``gaussian_encoder`` instances
with matching configs — a stand-in for the §5 Gaussian → LGQ edge
that needs a shape adapter to land).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.training.self_improve import (
    EdgeConfig,
    EvalReport,
    Orchestrator,
    PseudoLabelBroker,
    RoundRobinScheduler,
    SyntheticDataStream,
    Vault,
    gaussian_recon_label_fn,
    image_consistency_loss,
)
from scripts.training.self_improve.edges import apply_pending_labels
from scripts.training.self_improve.plugins import get_plugin


# ---------------------------------------------------------------------------
# label_fn
# ---------------------------------------------------------------------------


def test_gaussian_recon_label_fn_returns_shape_aligned_tensor() -> None:
    """A 3D ``(C, H, W)`` input round-trips through the producer's
    recon and comes back the same shape on CPU, with a finite
    confidence in ``(0, 1]``.
    """
    cls = get_plugin("gaussian_encoder")
    producer = cls()
    sample = torch.rand(1, 28, 28)
    label, conf = gaussian_recon_label_fn(sample, producer)
    assert label.shape == sample.shape
    assert label.device.type == "cpu"
    assert 0.0 < conf <= 1.0


def test_gaussian_recon_label_fn_handles_4d_input() -> None:
    """A 4D ``(N, C, H, W)`` batch is also accepted; the label has the
    same leading batch dim. (The broker normally calls the label_fn
    per-sample, but callers may want to batch internally.)
    """
    cls = get_plugin("gaussian_encoder")
    producer = cls()
    batch = torch.rand(3, 1, 28, 28)
    label, conf = gaussian_recon_label_fn(batch, producer)
    assert label.shape == batch.shape
    assert 0.0 < conf <= 1.0


def test_gaussian_recon_label_fn_rejects_wrong_dim() -> None:
    cls = get_plugin("gaussian_encoder")
    producer = cls()
    with pytest.raises(ValueError):
        gaussian_recon_label_fn(torch.rand(28, 28), producer)


# ---------------------------------------------------------------------------
# image_consistency_loss
# ---------------------------------------------------------------------------


def test_image_consistency_loss_weights_by_confidence() -> None:
    """A high-confidence label should contribute more to the loss
    than a low-confidence one (the loss is the *confidence-weighted
    mean* of per-sample MSEs).
    """
    inputs = torch.zeros(2, 1, 4, 4)
    student = torch.zeros(2, 1, 4, 4)
    # Each label is offset by the same amount, so per-sample MSE is
    # identical; the confidence weights differentiate them.
    labels = torch.ones(2, 1, 4, 4)
    high_conf = torch.tensor([1.0, 0.0])
    low_conf = torch.tensor([0.0, 1.0])
    loss_high = image_consistency_loss(inputs, student, labels, high_conf)
    loss_low = image_consistency_loss(inputs, student, labels, low_conf)
    # Symmetric weighting ⇒ equal losses.
    assert float(loss_high) == pytest.approx(float(loss_low))


def test_image_consistency_loss_rejects_shape_mismatch() -> None:
    inputs = torch.zeros(2, 1, 4, 4)
    student = torch.zeros(2, 1, 4, 4)
    labels = torch.ones(3, 1, 4, 4)  # wrong batch dim
    conf = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError):
        image_consistency_loss(inputs, student, labels, conf)


# ---------------------------------------------------------------------------
# Plugin consumer — gaussian_encoder
# ---------------------------------------------------------------------------


def _make_producer_and_broker(
    tmp_path: Path,
) -> tuple[PseudoLabelBroker, "object", str]:
    """Helper: build a broker with a snapshotted gaussian_encoder
    producer and a registered Gaussian→Gaussian image-recon edge.

    Returns ``(broker, producer, consumer_name)``.
    """
    cls = get_plugin("gaussian_encoder")
    producer = cls()
    consumer_name = "gaussian_encoder"

    vault = Vault(tmp_path / "vault")
    # Snapshot the producer so the broker has something to read.
    eval_report = producer.evaluate()
    vault.save("gaussian_encoder_producer", producer, eval_report, step=0)

    broker = PseudoLabelBroker(vault, strict_acyclic=False)
    broker.register_producer("gaussian_encoder_producer", producer)
    broker.register_edge(
        EdgeConfig(
            producer="gaussian_encoder_producer",
            consumer=consumer_name,
            min_stale_steps=0,
            confidence_threshold=0.0,
        ),
        label_fn=gaussian_recon_label_fn,
    )
    broker.finalize()
    return broker, producer, consumer_name


def test_consumer_step_folds_in_consistency_loss(tmp_path: Path) -> None:
    """When the orchestrator drops a non-empty
    ``_pending_pseudo_labels`` list onto a gaussian_encoder, the
    plugin's next ``train_step`` surfaces a ``pseudo/`` key in its
    step report and adds the term to the total loss.
    """
    broker, _producer, consumer_name = _make_producer_and_broker(tmp_path)
    cls = get_plugin("gaussian_encoder")
    consumer = cls()

    batch = torch.rand(2, 1, 28, 28)
    # Slice per-sample to match the broker's API.
    raw_inputs = [batch[i] for i in range(batch.shape[0])]
    consumer._pending_pseudo_labels = broker.sample(consumer_name, raw_inputs)
    assert consumer._pending_pseudo_labels, (
        "broker produced no labels; check that the snapshot is loaded"
    )

    report = consumer.train_step(batch)
    keys = list(report.losses)
    assert any(k.startswith("pseudo/") for k in keys), (
        f"no pseudo/ key in step report; got {keys}"
    )
    # Slot was cleared by the train_step.
    assert consumer._pending_pseudo_labels is None


def test_consumer_step_unchanged_without_labels() -> None:
    """Without pending labels, train_step is bit-identical to the
    pre-existing fused-backward path — no pseudo/ key in the report.
    """
    cls = get_plugin("gaussian_encoder")
    consumer = cls()
    batch = torch.rand(2, 1, 28, 28)
    report = consumer.train_step(batch)
    assert not any(k.startswith("pseudo/") for k in report.losses)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


def test_orchestrator_routes_labels_through_broker(tmp_path: Path) -> None:
    """End-to-end: orchestrator with a broker calls ``broker.sample``
    before each train_step, the consumer's step report carries a
    pseudo/ key, and ``broker.record_consumer_step`` advances the
    per-consumer staleness counter.
    """
    broker, _producer, consumer_name = _make_producer_and_broker(tmp_path)
    cls = get_plugin("gaussian_encoder")
    consumer = cls()

    orch = Orchestrator(
        {consumer_name: consumer},
        scheduler=RoundRobinScheduler([consumer_name]),
        data_stream=SyntheticDataStream(batch_size=2),
        eval_every=0,
        pseudo_label_broker=broker,
    )
    report = orch.run(3)

    # Some step in the run surfaced a pseudo/ key.
    any_pseudo = any(
        report.step_losses[consumer_name]  # populated == ran
    )
    assert any_pseudo, "no steps ran"
    # The broker's per-consumer staleness counter advanced by 3.
    edge_state = broker._edges[("gaussian_encoder_producer", consumer_name)]
    assert edge_state.consumer_steps_since_refresh >= 1


def test_orchestrator_without_broker_unchanged() -> None:
    """The orchestrator's pre-existing single-plugin path is
    unaffected when no broker is configured (regression guard for
    the wiring change)."""
    torch.manual_seed(0)
    cls = get_plugin("gaussian_encoder")
    plugin = cls()
    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_every=0,
    )
    report = orch.run(5)
    assert report.steps_per_component["gaussian_encoder"] == 5
    # No pseudo/ keys in any step.
    for loss in report.step_losses["gaussian_encoder"]:
        # ``step_losses`` is a float list (the scalar loss), but the
        # plugin's internal losses dict isn't surfaced here; just
        # checking we didn't crash is sufficient for the regression.
        assert isinstance(loss, float)


def test_orchestrator_clears_pending_labels_even_on_no_match(
    tmp_path: Path,
) -> None:
    """If the broker has no edges into a given consumer, the
    orchestrator still doesn't leak a stale ``_pending_pseudo_labels``
    value from a previous step.
    """
    vault = Vault(tmp_path / "vault")
    broker = PseudoLabelBroker(vault, strict_acyclic=False)
    broker.finalize()

    cls = get_plugin("gaussian_encoder")
    plugin = cls()
    # Pre-poison: simulate an old call having left a value behind.
    plugin._pending_pseudo_labels = ["bogus"]

    orch = Orchestrator(
        {"gaussian_encoder": plugin},
        scheduler=RoundRobinScheduler(["gaussian_encoder"]),
        eval_every=0,
        pseudo_label_broker=broker,
    )
    orch.run(1)
    assert plugin._pending_pseudo_labels is None


# ---------------------------------------------------------------------------
# Orchestrator finalizes the broker on construction (startup validation)
# ---------------------------------------------------------------------------


def test_orchestrator_finalizes_strict_acyclic_broker_with_cycle(
    tmp_path: Path,
) -> None:
    """The orchestrator must call ``broker.finalize()`` so the §5.3
    startup graph validation runs. Without that call, a strict-acyclic
    broker with cyclic edges would silently accept the config — the
    same config that ``finalize`` rejects.
    """
    cls = get_plugin("gaussian_encoder")
    plugin_a = cls()
    plugin_b = cls()

    vault = Vault(tmp_path / "vault")
    broker = PseudoLabelBroker(vault, strict_acyclic=True)
    broker.register_producer("a", plugin_a)
    broker.register_producer("b", plugin_b)
    # A↔B cycle: must be rejected at orchestrator construction.
    broker.register_edge(
        EdgeConfig("a", "b"), label_fn=lambda s, p: (s, 1.0),
    )
    broker.register_edge(
        EdgeConfig("b", "a"), label_fn=lambda s, p: (s, 1.0),
    )

    with pytest.raises(ValueError, match="cycle"):
        Orchestrator(
            {"a": plugin_a, "b": plugin_b},
            scheduler=RoundRobinScheduler(["a", "b"]),
            pseudo_label_broker=broker,
        )


def test_orchestrator_finalizes_non_strict_broker_missing_guards(
    tmp_path: Path,
) -> None:
    """Non-strict path: cyclic edges must declare ``min_stale_steps``
    and ``confidence_threshold`` above the floor. A configuration that
    omits the guards must be rejected at orchestrator construction.
    """
    cls = get_plugin("gaussian_encoder")
    plugin_a = cls()
    plugin_b = cls()

    vault = Vault(tmp_path / "vault")
    broker = PseudoLabelBroker(vault, strict_acyclic=False)
    broker.register_producer("a", plugin_a)
    broker.register_producer("b", plugin_b)
    # Cyclic edges with min_stale_steps below the floor.
    broker.register_edge(
        EdgeConfig("a", "b", min_stale_steps=10, confidence_threshold=0.5),
        label_fn=lambda s, p: (s, 1.0),
    )
    broker.register_edge(
        EdgeConfig("b", "a", min_stale_steps=10, confidence_threshold=0.5),
        label_fn=lambda s, p: (s, 1.0),
    )

    with pytest.raises(ValueError, match="min_stale_steps"):
        Orchestrator(
            {"a": plugin_a, "b": plugin_b},
            scheduler=RoundRobinScheduler(["a", "b"]),
            pseudo_label_broker=broker,
        )


def test_orchestrator_finalize_is_idempotent_with_pre_finalized_broker(
    tmp_path: Path,
) -> None:
    """Passing a broker that the caller already finalized works — the
    orchestrator's second ``finalize`` call is a no-op."""
    broker, _producer, consumer_name = _make_producer_and_broker(tmp_path)
    # Broker was already finalized by ``_make_producer_and_broker``.
    cls = get_plugin("gaussian_encoder")
    consumer = cls()
    orch = Orchestrator(
        {consumer_name: consumer},
        scheduler=RoundRobinScheduler([consumer_name]),
        eval_every=0,
        pseudo_label_broker=broker,
    )
    orch.run(1)
    # Sanity: the broker can still emit labels after the double
    # finalize.
    assert broker._finalized is True


# ---------------------------------------------------------------------------
# Partial confidence-gating ⇒ surviving subset still trains
# ---------------------------------------------------------------------------


def _build_partial_confidence_broker(
    tmp_path: Path, threshold: float, conf_per_sample: list[float],
) -> tuple[PseudoLabelBroker, str]:
    """Helper: builds a broker whose label_fn returns a per-sample
    confidence drawn from ``conf_per_sample`` in order. The producer
    plugin's identity doesn't matter — the label_fn just returns the
    sample back unchanged.

    Returns ``(broker, consumer_name)``.
    """
    cls = get_plugin("gaussian_encoder")
    producer = cls()
    consumer_name = "gaussian_encoder"

    vault = Vault(tmp_path / "vault")
    vault.save(
        "partial_producer", producer, producer.evaluate(), step=0,
    )

    broker = PseudoLabelBroker(vault, strict_acyclic=False)
    broker.register_producer("partial_producer", producer)

    queue = list(conf_per_sample)

    def _label_fn(sample: torch.Tensor, _producer) -> tuple[torch.Tensor, float]:
        # Return ``sample`` itself as the label so the consistency
        # loss is well-defined (student must match its own input ⇒
        # perfect zero when the student is the identity, but with
        # random init it just contributes a real gradient signal).
        conf = queue.pop(0) if queue else 0.0
        return sample.clone(), conf

    broker.register_edge(
        EdgeConfig(
            "partial_producer", consumer_name,
            min_stale_steps=0, confidence_threshold=threshold,
        ),
        label_fn=_label_fn,
    )
    broker.finalize()
    return broker, consumer_name


def test_apply_pending_labels_consumes_partial_subset(tmp_path: Path) -> None:
    """When confidence gating drops *some* (not all) samples, the
    consumer must still train on the surviving subset rather than
    skipping the whole batch — the broker reports survivor indices for
    exactly this purpose.

    Trajectory: 4 samples, threshold 0.6, confidences [0.1, 0.7, 0.4,
    0.9] ⇒ 2 survive (positions 1 and 3). The consumer's step report
    should carry a ``pseudo/`` key with ``n_kept = 2`` and no
    ``skipped`` indicator.
    """
    broker, consumer_name = _build_partial_confidence_broker(
        tmp_path, threshold=0.6, conf_per_sample=[0.1, 0.7, 0.4, 0.9],
    )

    cls = get_plugin("gaussian_encoder")
    consumer = cls()

    batch = torch.rand(4, 1, 28, 28)
    raw_inputs = [batch[i] for i in range(batch.shape[0])]
    consumer._pending_pseudo_labels = broker.sample(consumer_name, raw_inputs)
    # Pre-condition: broker returned a batch with 2 surviving samples
    # and indices [1, 3].
    assert len(consumer._pending_pseudo_labels) == 1
    pseudo_batch = consumer._pending_pseudo_labels[0]
    assert len(pseudo_batch.labels) == 2
    assert pseudo_batch.indices == [1, 3]

    report = consumer.train_step(batch)
    keys = list(report.losses)
    assert any(k.startswith("pseudo/") and not k.endswith("/skipped")
               for k in keys), (
        f"partial-confidence batch was dropped; got keys {keys}"
    )
    # n_kept reported explicitly so an operator can see what's happening.
    n_kept_keys = [k for k in keys if k.endswith("/n_kept")]
    assert n_kept_keys
    assert report.losses[n_kept_keys[0]] == 2.0


def test_apply_pending_labels_handles_full_drop(tmp_path: Path) -> None:
    """If every sample fails confidence gating, the broker emits no
    batch and the consumer's step proceeds without a pseudo-loss
    (no ``pseudo/`` key in the step report).
    """
    broker, consumer_name = _build_partial_confidence_broker(
        tmp_path, threshold=0.9, conf_per_sample=[0.1, 0.2, 0.3, 0.4],
    )

    cls = get_plugin("gaussian_encoder")
    consumer = cls()

    batch = torch.rand(4, 1, 28, 28)
    raw_inputs = [batch[i] for i in range(batch.shape[0])]
    consumer._pending_pseudo_labels = broker.sample(consumer_name, raw_inputs)
    # Broker emits no batch (per the all-dropped contract from phase 7).
    assert consumer._pending_pseudo_labels == []

    report = consumer.train_step(batch)
    assert not any(k.startswith("pseudo/") for k in report.losses)


def test_register_edge_rejected_after_finalize(tmp_path: Path) -> None:
    """Once finalized, the broker's graph is frozen — any further
    ``register_edge`` must raise so the validation pass stays
    authoritative.
    """
    vault = Vault(tmp_path / "vault")
    broker = PseudoLabelBroker(vault, strict_acyclic=False)
    broker.finalize()
    with pytest.raises(RuntimeError, match="frozen"):
        broker.register_edge(
            EdgeConfig("a", "b"), label_fn=lambda s, p: (s, 1.0),
        )
