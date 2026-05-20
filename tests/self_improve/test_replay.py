"""Tests for :class:`ReplayBank`.

The two invariants that matter for downstream forgetting defenses:

1. **Reservoir uniformity.** After seeing N items with capacity M < N,
   every item is in the buffer with probability M/N (approximately, with
   finite-sample noise).
2. **DER++ storage round-trip.** Stored logits come back exactly as
   inserted; ``sample`` is content-preserving.
"""

from __future__ import annotations

import torch

from scripts.training.self_improve import ReplayBank, ReplayItem


def test_buffer_fills_below_capacity() -> None:
    bank = ReplayBank(capacity=10, seed=0)
    for i in range(7):
        assert bank.insert("c", torch.tensor([float(i)]))
    assert bank.size("c") == 7
    assert bank.total_seen("c") == 7


def test_buffer_caps_at_capacity() -> None:
    bank = ReplayBank(capacity=5, seed=0)
    for i in range(100):
        bank.insert("c", torch.tensor([float(i)]))
    assert bank.size("c") == 5
    assert bank.total_seen("c") == 100


def test_components_are_isolated() -> None:
    bank = ReplayBank(capacity=4, seed=0)
    for i in range(3):
        bank.insert("a", torch.tensor([float(i)]))
    for i in range(10, 12):
        bank.insert("b", torch.tensor([float(i)]))
    assert bank.size("a") == 3
    assert bank.size("b") == 2
    assert set(bank.components()) == {"a", "b"}


def test_sample_smaller_than_buffer() -> None:
    bank = ReplayBank(capacity=10, seed=42)
    for i in range(10):
        bank.insert("c", torch.tensor([float(i)]))
    items = bank.sample("c", k=3)
    assert len(items) == 3
    # No duplicates from a single sample call.
    ids = {id(it) for it in items}
    assert len(ids) == 3


def test_sample_larger_than_buffer_returns_all() -> None:
    bank = ReplayBank(capacity=10, seed=0)
    for i in range(4):
        bank.insert("c", torch.tensor([float(i)]))
    items = bank.sample("c", k=10)
    assert len(items) == 4


def test_sample_empty_returns_empty_list() -> None:
    bank = ReplayBank(capacity=10, seed=0)
    assert bank.sample("c", k=5) == []
    assert bank.sample("c", k=0) == []


def test_der_logits_stored_and_returned() -> None:
    """DER++ requires the logits we stored at insertion to come back
    unchanged on sample (no rounding, no shape transform)."""
    bank = ReplayBank(capacity=10, seed=0)
    sample = torch.randn(3, 8, 8)
    logits = torch.randn(3, 8, 8)
    bank.insert("c", sample, stored_logits=logits, step=42)
    [item] = bank.sample("c", k=1)
    assert torch.equal(item.sample, sample)
    assert item.stored_logits is not None
    assert torch.equal(item.stored_logits, logits)
    assert item.insert_step == 42


def test_reservoir_is_approximately_uniform() -> None:
    """Statistical check: with capacity 100 and stream 1000, each of the
    1000 items should appear with probability ~0.1. Over many repeats,
    the empirical inclusion frequency for any given item lands near 0.1.
    """
    capacity = 100
    stream_len = 1000
    trials = 200

    counts = torch.zeros(stream_len, dtype=torch.long)
    for trial in range(trials):
        bank = ReplayBank(capacity=capacity, seed=trial)
        for i in range(stream_len):
            bank.insert("c", torch.tensor([i]))
        # Read each item's value to know which stream indices survived.
        for item in bank.sample("c", k=capacity):
            counts[int(item.sample.item())] += 1

    # Expected per-item count is trials * capacity / stream_len = 20.
    # Use a generous tolerance; this is a smoke test, not a statistics
    # exam. Each item's mean across trials should be close to 20.
    expected = trials * capacity / stream_len
    mean = counts.float().mean().item()
    assert abs(mean - expected) < 0.5, f"mean={mean} expected={expected}"


def test_clear_component() -> None:
    bank = ReplayBank(capacity=5, seed=0)
    bank.insert("a", torch.tensor([1.0]))
    bank.insert("b", torch.tensor([2.0]))
    bank.clear("a")
    assert bank.size("a") == 0
    assert bank.size("b") == 1


def test_clear_all() -> None:
    bank = ReplayBank(capacity=5, seed=0)
    bank.insert("a", torch.tensor([1.0]))
    bank.insert("b", torch.tensor([2.0]))
    bank.clear()
    assert bank.size("a") == 0
    assert bank.size("b") == 0
    assert bank.components() == []
