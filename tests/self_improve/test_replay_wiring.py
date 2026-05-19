"""Cross-plugin smoke test for the phase-3 forgetting wiring.

Verifies that every plugin's ``train_step`` is replay/EMA-aware:

- ``attach_replay`` makes subsequent steps populate the buffer.
- After the buffer is non-empty, a step adds a ``replay/*`` (or DER++)
  loss key to the step report.
- ``attach_ema`` makes subsequent steps advance the EMA's update counter.

These are *integration* checks; the math is tested in
``test_replay.py`` and ``test_ema.py``. The intent is to catch a plugin
that silently ignores the attached aux state.
"""

from __future__ import annotations

import pytest

from scripts.training.self_improve import ReplayBank
from scripts.training.self_improve.plugins import available_plugins, get_plugin


@pytest.mark.parametrize("name", list(available_plugins()))
def test_attach_replay_populates_buffer(name: str) -> None:
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    bank = ReplayBank(capacity=32, seed=0)
    plugin.attach_replay(bank, batch_size=1)

    batch = plugin.make_synthetic_batch(batch_size=2)
    plugin.train_step(batch)
    assert bank.size(name) > 0, f"{name}: buffer empty after attach_replay + step"


@pytest.mark.parametrize("name", list(available_plugins()))
def test_replay_loss_appears_after_buffer_fills(name: str) -> None:
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    bank = ReplayBank(capacity=32, seed=0)
    plugin.attach_replay(bank, batch_size=1)

    # First step seeds the buffer; second step should fold in a replay loss.
    batch = plugin.make_synthetic_batch(batch_size=2)
    plugin.train_step(batch)
    report = plugin.train_step(batch)

    has_replay = any(
        k.startswith("replay/") or k == "replay_task" or k == "der_consistency"
        for k in report.losses
    )
    assert has_replay, (
        f"{name}: no replay/DER loss in step report after buffer filled; "
        f"got keys {list(report.losses)}"
    )


@pytest.mark.parametrize("name", list(available_plugins()))
def test_attach_ema_updates_after_step(name: str) -> None:
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    plugin.attach_ema(decay=0.99)
    assert plugin.ema_teacher is not None
    assert plugin.ema_teacher.num_updates == 0

    batch = plugin.make_synthetic_batch(batch_size=2)
    plugin.train_step(batch)
    assert plugin.ema_teacher.num_updates == 1
