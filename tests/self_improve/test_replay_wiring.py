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


# ---------------------------------------------------------------------------
# Persistence — vault rollback / resume must restore EMA teacher state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(available_plugins()))
def test_ema_state_round_trips_through_state_dict(name: str) -> None:
    """The vault stores ``plugin.state_dict()`` opaquely; if the EMA
    teacher isn't included there, ``rollback()`` restores the student
    but leaves the teacher pointing at whatever state the freshly
    re-attached EMA happened to capture. After that, every distillation
    step computes loss against a teacher that no longer reflects the
    restored student history.

    Round-trip protocol:
      1. Construct a plugin, attach an EMA teacher.
      2. Take a few training steps so the EMA diverges from a freshly
         constructed teacher's initial state.
      3. Capture ``state_dict()`` and ``ema_teacher.state_dict()``.
      4. Build a fresh plugin (no EMA), call ``load_state_dict``.
      5. Verify the loaded plugin's ema_teacher exists, decay matches,
         num_updates matches, and the EMA tensors are bit-identical.
    """
    import torch as _torch

    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    plugin.attach_ema(decay=0.5)
    for _ in range(3):
        plugin.train_step(plugin.make_synthetic_batch(batch_size=2))

    expected_params = {
        k: v.detach().clone()
        for k, v in plugin.ema_teacher._params.items()
    }
    expected_updates = plugin.ema_teacher.num_updates
    saved = plugin.state_dict()
    assert "_ema" in saved, f"{name}: state_dict missing _ema key"

    # Fresh plugin without attach_ema.
    fresh = cls()
    assert fresh.ema_teacher is None
    fresh.load_state_dict(saved)

    assert fresh.ema_teacher is not None, (
        f"{name}: load_state_dict did not reattach EMA from checkpoint"
    )
    assert fresh.ema_teacher.decay == pytest.approx(0.5)
    assert fresh.ema_teacher.num_updates == expected_updates
    for k, expected in expected_params.items():
        loaded = fresh.ema_teacher._params[k]
        assert _torch.equal(loaded, expected), f"{name}: EMA param {k!r} drifted"


# ---------------------------------------------------------------------------
# Phase 4 wiring: ``attach_ewc``
# ---------------------------------------------------------------------------


# Phase 4 wires EWC into OmniLatent and MMWM only; other plugins accept
# ``attach_ewc`` (inherited from base) but their ``train_step`` doesn't
# invoke ``ewc.consolidate`` / ``ewc.post_step`` yet, so the counters
# don't advance. Parametrize over only the wired plugins for now.
_EWC_WIRED_PLUGINS = ("omnilatent", "mmwm")


@pytest.mark.parametrize("name", list(_EWC_WIRED_PLUGINS))
def test_attach_ewc_advances_counters(name: str) -> None:
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    plugin.attach_ewc(fisher_decay=0.9, lam_ewc=1.0, lam_si=1.0)
    assert plugin.ewc is not None
    assert plugin.ewc.num_fisher_updates == 0
    assert plugin.ewc.num_si_updates == 0

    batch = plugin.make_synthetic_batch(batch_size=2)
    report = plugin.train_step(batch)
    assert plugin.ewc.num_fisher_updates == 1
    assert plugin.ewc.num_si_updates == 1
    assert "ewc/penalty" in report.losses


def test_hpwm_lr_scheduler_advances_with_replay() -> None:
    """The replay branch performs a second ``optimizer.step()``; the
    LR scheduler must advance with it. Without this fix, enabling
    replay silently trains HPWM at higher LRs than the configured
    cosine schedule prescribes (the scheduler lags the optimizer by
    one step per train_step call once the buffer is non-empty).
    """
    HPWMPlugin = get_plugin("hpwm")
    plugin = HPWMPlugin()
    bank = ReplayBank(capacity=8, seed=0)
    plugin.attach_replay(bank, batch_size=1)

    initial_scheduler_steps = plugin.scheduler.last_epoch

    # Step 1: buffer is empty, only the main optimizer.step runs ⇒ +1.
    plugin.train_step(plugin.make_synthetic_batch(batch_size=1))
    after_step1 = plugin.scheduler.last_epoch
    assert after_step1 - initial_scheduler_steps == 1, (
        f"first step (no replay yet) should advance scheduler by 1; "
        f"got {after_step1 - initial_scheduler_steps}"
    )

    # Step 2: buffer is non-empty, both main and replay optimizer.steps
    # fire ⇒ +2.
    plugin.train_step(plugin.make_synthetic_batch(batch_size=1))
    after_step2 = plugin.scheduler.last_epoch
    assert after_step2 - after_step1 == 2, (
        f"second step (replay active) should advance scheduler by 2; "
        f"got {after_step2 - after_step1}"
    )
