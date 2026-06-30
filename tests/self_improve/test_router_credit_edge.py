"""W5.5 — router-credit pseudo-label edge is covered by the §5 guards."""

from __future__ import annotations

from pathlib import Path

import torch

from omnilatent.model.hooks import LatentNeuralHook
from scripts.training.self_improve.edges import router_credit_label_fn
from scripts.training.self_improve.plugins import get_plugin
from scripts.training.self_improve.pseudo_labels import EdgeConfig, PseudoLabelBroker
from scripts.training.self_improve.vault import Vault


def _producer_with_hooks(n_hooks: int = 2):
    producer = get_plugin("omnilatent")()
    dim = producer.model.config.hidden_dim
    for i in range(n_hooks):
        producer.model.register_hook(
            LatentNeuralHook(name=f"h{i}", num_tokens=4, dim=dim, target_layers=[0, 1])
        )
    return producer


def _img(producer, b: int = 2) -> torch.Tensor:
    c = producer.model.config
    return torch.rand(b, c.image_channels, c.image_size, c.image_size)


# --- label_fn correctness ----------------------------------------------
def test_router_credit_label_fn_shape_and_confidence() -> None:
    producer = _producer_with_hooks(2)
    sample = _img(producer, 1)[0]  # (C, H, W)
    label, conf = router_credit_label_fn(sample, producer)
    assert label.shape == (2,)            # one credit per hook
    assert 0.0 < conf <= 1.0


def test_router_credit_label_fn_no_hooks_zero_confidence() -> None:
    producer = get_plugin("omnilatent")()  # no hooks
    label, conf = router_credit_label_fn(_img(producer, 1)[0], producer)
    assert label.numel() == 0 and conf == 0.0


# --- §5 guard coverage: sever + auto-heal ------------------------------
def _broker(tmp_path: Path, heal_after: int = 3, conf_thresh: float = 0.0):
    producer = _producer_with_hooks(2)
    vault = Vault(tmp_path / "vault")
    vault.save("omni_producer", producer, producer.evaluate(), step=0)
    broker = PseudoLabelBroker(vault, strict_acyclic=False)
    broker.register_producer("omni_producer", producer)
    broker.register_edge(
        EdgeConfig(
            producer="omni_producer",
            consumer="omni_consumer",
            min_stale_steps=0,
            confidence_threshold=conf_thresh,
            heal_after_steps=heal_after,
        ),
        label_fn=router_credit_label_fn,
    )
    broker.finalize()
    return broker, producer


def test_router_credit_flows_then_severs_then_heals(tmp_path: Path) -> None:
    broker, producer = _broker(tmp_path, heal_after=3)
    raw = [r for r in _img(producer, 2)]

    # 1. Router credit flows through the edge.
    assert broker.sample("omni_consumer", raw), "router-credit edge produced no labels"

    # 2. Divergence guard severs the (poisoned) edge — labels stop.
    broker.mark_edge_severed("omni_producer", "omni_consumer")
    assert broker.is_severed("omni_producer", "omni_consumer")
    assert broker.sample("omni_consumer", raw) == []

    # 3. Auto-heal after heal_after_steps clean consumer steps — labels resume.
    for _ in range(3):
        broker.record_consumer_step("omni_consumer")
    assert not broker.is_severed("omni_producer", "omni_consumer")
    assert broker.sample("omni_consumer", raw), "edge did not heal"


def test_confidence_gate_drops_uninformative_router_credit(tmp_path: Path) -> None:
    # With 2 hooks the credit softmax peaks at ~0.5; a 0.9 threshold drops it.
    broker, producer = _broker(tmp_path, conf_thresh=0.9)
    raw = [r for r in _img(producer, 2)]
    assert broker.sample("omni_consumer", raw) == []
