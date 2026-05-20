"""Phase 8 tests: plateau-triggered hook-based capacity expansion.

Acceptance test (design doc):
    "Manufactured plateau triggers expansion; new hook trains; backbone
     weights unchanged."
"""

from __future__ import annotations

import pytest
import torch

from scripts.training.self_improve.forgetting import (
    PlateauDetector,
    expand_omnilatent_capacity,
)
from scripts.training.self_improve.plugins import get_plugin


# ---------------------------------------------------------------------------
# PlateauDetector
# ---------------------------------------------------------------------------


def test_plateau_detector_does_not_fire_before_window_fills() -> None:
    d = PlateauDetector(patience=3, tol=0.0)
    for v in (1.0, 2.0, 3.0):  # only 3 records < 2*patience
        assert d.record(v) is False


def test_plateau_detector_fires_when_no_improvement() -> None:
    d = PlateauDetector(patience=3, tol=0.0, higher_is_better=True)
    # 6 flat scores → recent best = prev best → plateau.
    for v in (1.0, 1.0, 1.0, 1.0, 1.0, 1.0):
        last = d.record(v)
    assert last is True
    assert d.triggered is True


def test_plateau_detector_does_not_fire_when_improving() -> None:
    d = PlateauDetector(patience=3, tol=0.0, higher_is_better=True)
    triggered = False
    for v in (1.0, 1.0, 1.0, 2.0, 2.0, 2.0):
        triggered = d.record(v)
    assert triggered is False
    assert d.triggered is False


def test_plateau_detector_lower_is_better() -> None:
    """With lower-is-better (e.g. MSE) the polarity flips."""
    d = PlateauDetector(patience=3, tol=0.0, higher_is_better=False)
    # MSE decreasing → improvement → no trigger.
    triggered = False
    for v in (1.0, 1.0, 1.0, 0.5, 0.5, 0.5):
        triggered = d.record(v)
    assert triggered is False


def test_plateau_detector_latches_on_trigger() -> None:
    """Once triggered, subsequent records keep returning True so the
    orchestrator doesn't fire expansion in a loop."""
    d = PlateauDetector(patience=2, tol=0.0)
    for v in (1.0, 1.0, 1.0, 1.0):
        d.record(v)
    assert d.triggered
    # Inject a "great" score; detector stays latched until reset.
    assert d.record(100.0) is True
    d.reset()
    assert d.triggered is False
    # After reset, detection works fresh.
    for v in (5.0, 5.0, 5.0, 5.0):
        d.record(v)
    assert d.triggered


def test_plateau_detector_rejects_invalid_patience() -> None:
    with pytest.raises(ValueError):
        PlateauDetector(patience=0)


def test_plateau_detector_respects_tol() -> None:
    """Small improvements within tol still count as plateau."""
    d = PlateauDetector(patience=2, tol=1.0, higher_is_better=True)
    # Improvement of 0.5 < tol → plateau.
    triggered = False
    for v in (1.0, 1.0, 1.4, 1.5):
        triggered = d.record(v)
    assert triggered is True


# ---------------------------------------------------------------------------
# OmniLatent capacity expansion
# ---------------------------------------------------------------------------


def test_expand_omnilatent_freezes_backbone_and_adds_hook() -> None:
    """The phase-8 acceptance test.

    Manufacture a plateau (the trigger itself is tested above; here we
    just call the expander directly), verify:

    1. The backbone parameters are all frozen (requires_grad=False).
    2. A new hook is registered on the model's hook_manager.
    3. The new hook's parameters are trainable.
    4. The backbone weight VALUES are unchanged after expansion (just
       the requires_grad flag flipped; the optimizer hasn't moved them).
    """
    torch.manual_seed(0)
    OmniLatentPlugin = get_plugin("omnilatent")
    plugin = OmniLatentPlugin()

    # Snapshot backbone weights for the "values unchanged" check below.
    backbone_pre = {
        n: p.detach().clone()
        for n, p in plugin.model.named_parameters()
    }

    hook = expand_omnilatent_capacity(plugin, num_tokens=4)

    # 1. Backbone frozen.
    for n, p in plugin.model.named_parameters():
        if n.startswith("hook_manager.hooks." + hook.name):
            assert p.requires_grad, f"{n}: new hook should be trainable"
        else:
            assert not p.requires_grad, f"{n}: backbone should be frozen"

    # 2. Hook registered.
    assert hook.name in plugin.model.hook_manager.hooks

    # 3. Hook trainable.
    trainable = [p for p in hook.parameters() if p.requires_grad]
    assert len(trainable) > 0, "new hook has no trainable parameters"

    # 4. Backbone values unchanged.
    for n, p in plugin.model.named_parameters():
        if n in backbone_pre:
            assert torch.equal(p.detach(), backbone_pre[n]), (
                f"{n}: backbone value changed during expansion"
            )


def test_expansion_then_train_step_does_not_move_backbone() -> None:
    """After expansion, taking a train step updates the hook's
    parameters but leaves the backbone weights unchanged (the
    forgetting-defense guarantee from §4.4).

    The expander rebuilds the plugin's optimizer to manage the new
    hook's params, so a plain ``plugin.train_step`` call after
    expansion is sufficient — no manual optimizer surgery required.
    """
    torch.manual_seed(0)
    OmniLatentPlugin = get_plugin("omnilatent")
    plugin = OmniLatentPlugin()
    hook = expand_omnilatent_capacity(plugin, num_tokens=4)

    backbone_pre = {
        n: p.detach().clone()
        for n, p in plugin.model.named_parameters()
        if not n.startswith("hook_manager.hooks." + hook.name)
    }
    hook_pre = {
        n: p.detach().clone()
        for n, p in hook.named_parameters()
    }

    batch = plugin.make_synthetic_batch(batch_size=2)
    plugin.train_step(batch)

    # Backbone unchanged.
    for n, p in plugin.model.named_parameters():
        if n in backbone_pre:
            assert torch.equal(p.detach(), backbone_pre[n]), (
                f"{n}: backbone moved during post-expansion train step"
            )

    # At least one hook parameter changed.
    any_changed = False
    for n, p in hook.named_parameters():
        if not torch.equal(p.detach(), hook_pre[n]):
            any_changed = True
            break
    assert any_changed, "no hook parameter moved during the train step"


def test_expansion_rejects_duplicate_hook_name() -> None:
    OmniLatentPlugin = get_plugin("omnilatent")
    plugin = OmniLatentPlugin()
    expand_omnilatent_capacity(plugin, hook_name="extra")
    with pytest.raises(KeyError):
        expand_omnilatent_capacity(plugin, hook_name="extra")


def test_expansion_rejects_wrong_plugin_type() -> None:
    """Calling the expander on a non-OmniLatent plugin must fail
    loudly rather than silently no-op."""
    GaussianEncoderPlugin = get_plugin("gaussian_encoder")
    plugin = GaussianEncoderPlugin()
    with pytest.raises(TypeError, match="OmniLatentPlugin"):
        expand_omnilatent_capacity(plugin)


def test_expansion_rebuilds_optimizer_over_hook_params() -> None:
    """After expansion, the plugin's optimizer must reference the new
    hook's parameters (and only those). Without this rebuild, the
    optimizer still holds references to the frozen pre-expansion
    parameter list and ``plugin.train_step`` silently updates nothing.
    """
    torch.manual_seed(0)
    OmniLatentPlugin = get_plugin("omnilatent")
    plugin = OmniLatentPlugin()
    hook = expand_omnilatent_capacity(plugin, num_tokens=4)

    # Every parameter the optimizer manages is one of the hook's params.
    hook_param_ids = {id(p) for p in hook.parameters()}
    opt = plugin._trainer.optimizer
    managed_ids = {
        id(p) for group in opt.param_groups for p in group["params"]
    }
    assert managed_ids, "optimizer manages no parameters at all"
    assert managed_ids.issubset(hook_param_ids), (
        "optimizer references parameters outside the new hook: "
        f"{managed_ids - hook_param_ids}"
    )
    # And all hook trainable params are present.
    assert managed_ids == hook_param_ids


def test_expansion_opt_out_leaves_optimizer_alone() -> None:
    """When ``rebuild_optimizer=False`` is requested, the caller owns
    the optimizer-swap responsibility. The test exists to lock the
    opt-out contract in case future plugin code grows a separate
    optimizer-management strategy.
    """
    torch.manual_seed(0)
    OmniLatentPlugin = get_plugin("omnilatent")
    plugin = OmniLatentPlugin()
    pre_optimizer = plugin._trainer.optimizer
    expand_omnilatent_capacity(plugin, num_tokens=4, rebuild_optimizer=False)
    assert plugin._trainer.optimizer is pre_optimizer
