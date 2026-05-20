"""Section-B followup tests: fused DER++ + EWC coverage extensions.

Phase 3's initial wiring used an "extra-step replay" pattern on
lgq / omnilatent / mmwm / hpwm — a second ``optimizer.step`` on a
replayed batch — which works for non-forgetting but isn't true DER++
and doubles the optimizer call count when the buffer is active. This
follow-up converts ``omnilatent``, ``mmwm``, and ``hpwm`` to fused
DER++: one backward over (task + replay_task + DER consistency).

Phase 4 wired EWC into omnilatent + mmwm. This follow-up extends EWC
to gaussian_encoder + hpwm. ``lgq`` stays as the holdout because its
dual-optimizer (generator + discriminator) + AMP scaler trainer step
isn't easily restructured around an external backward.

The tests here lock in the *behavioral* contracts the followup
restructure changed:

- Fused DER++: one ``optimizer.step()`` per ``train_step``, regardless
  of whether the replay buffer is non-empty.
- DER++ logits storage: every newly buffered sample carries a
  ``stored_logits`` tensor on the wired plugins (was ``None`` before).
- EWC wiring: ``ewc.consolidate`` and ``ewc.post_step`` advance their
  counters and ``ewc/penalty`` shows up in the step report for each
  newly-wired plugin.
"""

from __future__ import annotations

import pytest
import torch

from scripts.training.self_improve import ReplayBank
from scripts.training.self_improve.plugins import get_plugin


# Plugins now using the fused-DER++ replay pattern (the followup restructure).
_FUSED_DER_PLUGINS = ("gaussian_encoder", "omnilatent", "mmwm", "hpwm")
# Plugins newly wired for EWC in the followup.
_NEW_EWC_PLUGINS = ("gaussian_encoder", "hpwm")
# Plugins explicitly left out of the followup (GAN dual-optimizer makes
# the restructure invasive; tracked as future work).
_HOLDOUT_PLUGINS = ("lgq",)


# ---------------------------------------------------------------------------
# Fused DER++ — single optimizer step
# ---------------------------------------------------------------------------


def _count_optimizer_steps(plugin) -> int:
    """Read the plugin's persistent optimizer-step counter.

    Different plugins use different counter names — pick the one each
    one increments on every optimizer.step call.
    """
    name = getattr(plugin, "name", None)
    if name == "hpwm":
        return int(plugin.step_count)
    if name in ("omnilatent", "mmwm"):
        return int(plugin._trainer.global_step)
    if name == "gaussian_encoder":
        return int(plugin._trainer.step_count)
    raise KeyError(f"unknown plugin name: {name!r}")


@pytest.mark.parametrize("name", list(_FUSED_DER_PLUGINS))
def test_fused_der_takes_one_optimizer_step_per_train_step(name: str) -> None:
    """A train_step must take exactly one optimizer.step regardless of
    whether the replay buffer is empty or not.

    With the previous "extra-step" pattern this would have been 1 on
    the first call (empty buffer) and 2 on the second (replay active).
    """
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    bank = ReplayBank(capacity=8, seed=0)
    plugin.attach_replay(bank, batch_size=1)

    pre = _count_optimizer_steps(plugin)
    plugin.train_step(plugin.make_synthetic_batch(batch_size=2))
    after_step1 = _count_optimizer_steps(plugin)
    assert after_step1 - pre == 1, (
        f"{name}: first train_step (empty buffer) should advance the "
        f"optimizer-step counter by 1; got {after_step1 - pre}"
    )

    # Second train_step: replay buffer is now non-empty.
    plugin.train_step(plugin.make_synthetic_batch(batch_size=2))
    after_step2 = _count_optimizer_steps(plugin)
    assert after_step2 - after_step1 == 1, (
        f"{name}: second train_step (replay active) should still take "
        f"a single optimizer step; got {after_step2 - after_step1}"
    )


# ---------------------------------------------------------------------------
# DER++ logits storage — every insert carries a stored teacher signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_FUSED_DER_PLUGINS))
def test_buffer_inserts_carry_stored_logits(name: str) -> None:
    """The DER++ stored-logits payload is the whole point of the
    fused-DER restructure. Verify the wired plugins actually populate
    ``ReplayItem.stored_logits`` rather than leaving it ``None``.
    """
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    bank = ReplayBank(capacity=8, seed=0)
    plugin.attach_replay(bank, batch_size=1)
    plugin.train_step(plugin.make_synthetic_batch(batch_size=2))

    assert bank.size(name) > 0, f"{name}: buffer empty after a step"
    item = bank.sample(name, k=1)[0]
    assert item.stored_logits is not None, (
        f"{name}: DER teacher signal not stored on buffer insert"
    )
    assert isinstance(item.stored_logits, torch.Tensor)


@pytest.mark.parametrize("name", list(_HOLDOUT_PLUGINS))
def test_holdout_plugin_documented(name: str) -> None:
    """Locks in the holdout: LGQ still uses extra-step replay; the
    buffer entries carry no DER teacher signal. If we ever restructure
    LGQ's GAN trainer to expose the backward, this test should be
    removed and ``lgq`` should join ``_FUSED_DER_PLUGINS``.
    """
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"{name}: missing dependency ({exc})")

    bank = ReplayBank(capacity=8, seed=0)
    plugin.attach_replay(bank, batch_size=1)
    plugin.train_step(plugin.make_synthetic_batch(batch_size=2))
    item = bank.sample(name, k=1)[0]
    assert item.stored_logits is None, (
        f"{name}: holdout plugin unexpectedly stored DER logits — if the "
        "GAN trainer was restructured, update the test catalogues."
    )


# ---------------------------------------------------------------------------
# EWC wiring on newly-supported plugins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_NEW_EWC_PLUGINS))
def test_newly_wired_ewc_advances_counters(name: str) -> None:
    """After ``attach_ewc``, the next train_step should call
    ``consolidate`` (Fisher accumulation) and ``post_step`` (SI path
    integral), advancing both counters. The ``ewc/penalty`` key should
    appear in the step report.
    """
    cls = get_plugin(name)
    try:
        plugin = cls()
    except ImportError as exc:  # pragma: no cover - env-specific
        pytest.skip(f"{name}: missing dependency ({exc})")

    plugin.attach_ewc(fisher_decay=0.9, lam_ewc=1.0, lam_si=1.0)
    assert plugin.ewc is not None
    assert plugin.ewc.num_fisher_updates == 0
    assert plugin.ewc.num_si_updates == 0

    report = plugin.train_step(plugin.make_synthetic_batch(batch_size=2))
    assert plugin.ewc.num_fisher_updates == 1
    assert plugin.ewc.num_si_updates == 1
    assert "ewc/penalty" in report.losses


def test_ewc_penalty_actually_affects_gradients() -> None:
    """A regression-style check: with high lam_ewc, the EWC penalty
    should noticeably bias the optimizer trajectory compared to a
    matched run with ``lam_ewc=0``. We measure cumulative parameter
    drift from the anchor over several steps and assert that the
    high-lambda run drifts strictly less.

    The first step from the anchor has zero penalty gradient (because
    ``θ - θ* = 0``), so the test takes 8 steps post-anchor to give the
    regularizer a chance to bite.
    """
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")

    def _measure(lam: float) -> float:
        torch.manual_seed(0)
        plugin = GaussianEncoderPlugin(lr=1e-2)
        plugin.attach_ewc(fisher_decay=0.5, lam_ewc=lam, lam_si=0.0)
        # Fill Fisher with non-zero values by taking a couple of steps
        # *before* snapshotting the anchor.
        for _ in range(2):
            plugin.train_step(plugin.make_synthetic_batch(batch_size=2))
        plugin.ewc.snapshot_anchor(plugin.model)
        anchor = {
            n: p.detach().clone()
            for n, p in plugin.model.named_parameters()
        }
        for _ in range(8):
            plugin.train_step(plugin.make_synthetic_batch(batch_size=2))
        drift = 0.0
        for n, p in plugin.model.named_parameters():
            drift += float((p.detach() - anchor[n]).pow(2).sum().item())
        return drift

    drift_no_ewc = _measure(lam=0.0)
    drift_strong_ewc = _measure(lam=1e6)
    assert drift_strong_ewc < drift_no_ewc, (
        f"lam_ewc=1e6 did not reduce drift: "
        f"strong={drift_strong_ewc:.4e} vs none={drift_no_ewc:.4e}"
    )
