"""Tests for online EWC + Synaptic Intelligence (`EWCSI`).

Invariants:

- Initialization captures the current params as the anchor; ``F`` and
  ``ω`` start at zero.
- ``consolidate`` updates Fisher with the running-EMA formula.
- ``post_step`` accumulates positive ``-g·Δθ`` into ``ω``; ignores
  negative contributions.
- ``penalty`` is zero at the anchor (no movement), grows quadratically
  with ``(θ - θ*)``, and respects ``lam_ewc`` / ``lam_si`` weights.
- ``snapshot_anchor`` zeroes ``ω`` and updates ``θ*``; does *not* touch
  ``F`` (which tracks global sensitivity, not task-specific).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scripts.training.self_improve import EWCSI


def _tiny_model() -> nn.Module:
    m = nn.Linear(4, 4, bias=False)
    nn.init.normal_(m.weight, std=0.5)
    return m


def _fill_grads(model: nn.Module, scale: float = 1.0) -> None:
    """Populate ``.grad`` on every trainable parameter with deterministic
    values so ``consolidate`` has something to work with.
    """
    g = torch.Generator().manual_seed(0)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g) * scale


def test_init_anchor_matches_model() -> None:
    m = _tiny_model()
    ewc = EWCSI(m)
    for n, p in m.named_parameters():
        assert torch.equal(ewc.anchor[n], p)
        assert torch.equal(ewc.fisher[n], torch.zeros_like(p))
        assert torch.equal(ewc.omega[n], torch.zeros_like(p))


def test_consolidate_updates_fisher() -> None:
    m = _tiny_model()
    ewc = EWCSI(m, fisher_decay=0.5)
    _fill_grads(m, scale=2.0)
    grads_before = {n: p.grad.clone() for n, p in m.named_parameters()}

    ewc.consolidate(m)

    # Initial fisher was zero; expect F = 0.5*0 + 0.5*grad² = 0.5*grad²
    for n, p in m.named_parameters():
        expected = 0.5 * grads_before[n] ** 2
        assert torch.allclose(ewc.fisher[n], expected, atol=1e-6)


def test_consolidate_decays_old_fisher() -> None:
    m = _tiny_model()
    ewc = EWCSI(m, fisher_decay=0.9)
    # Seed the fisher with non-zero values.
    for n in ewc.fisher:
        ewc.fisher[n].fill_(4.0)

    _fill_grads(m, scale=1.0)
    g2 = {n: p.grad.detach() ** 2 for n, p in m.named_parameters()}

    ewc.consolidate(m)
    # F = 0.9 * 4.0 + 0.1 * grad² = 3.6 + 0.1*grad²
    for n in ewc.fisher:
        expected = 3.6 + 0.1 * g2[n]
        assert torch.allclose(ewc.fisher[n], expected, atol=1e-6)


def test_si_accumulates_only_positive_contributions() -> None:
    """ω_i += max(0, -g_i · Δθ_i). A step in the same direction as
    -grad (i.e. SGD) yields positive contribution; the reverse step
    yields zero (clamped)."""
    m = _tiny_model()
    ewc = EWCSI(m)
    # Set grads.
    g = torch.ones_like(m.weight)
    m.weight.grad = g.clone()

    # Snapshot via consolidate (also resets fisher; we don't care for SI).
    ewc.consolidate(m)

    # Take a step in the loss-decreasing direction (Δθ = -g).
    with torch.no_grad():
        m.weight.sub_(g)
    ewc.post_step(m)
    # contrib = -g · Δθ = -1 · -1 = +1, clamped >=0, so ω = 1.
    assert torch.allclose(ewc.omega["weight"], torch.ones_like(m.weight))

    # Take a step in the loss-INCREASING direction (Δθ = +g).
    m.weight.grad = g.clone()
    ewc.consolidate(m)
    with torch.no_grad():
        m.weight.add_(g)
    ewc.post_step(m)
    # contrib = -g · Δθ = -1·1 = -1, clamped to 0, ω unchanged.
    assert torch.allclose(ewc.omega["weight"], torch.ones_like(m.weight))


def test_penalty_zero_at_anchor() -> None:
    m = _tiny_model()
    ewc = EWCSI(m, lam_ewc=10.0, lam_si=10.0)
    # Make F and ω nonzero so the importance term isn't trivially zero.
    for n in ewc.fisher:
        ewc.fisher[n].fill_(1.0)
        ewc.omega[n].fill_(1.0)
    # θ == anchor → (θ−θ*)² = 0 → penalty = 0.
    penalty = ewc.penalty(m)
    assert float(penalty.item()) == pytest.approx(0.0, abs=1e-6)


def test_penalty_grows_with_displacement() -> None:
    m = _tiny_model()
    ewc = EWCSI(m, lam_ewc=1.0, lam_si=0.0)  # SI off for clarity
    # Set F = 1 everywhere; anchor is current weights.
    for n in ewc.fisher:
        ewc.fisher[n].fill_(1.0)

    # Move θ by 1 in every dim; penalty = 0.5 * Σ F·(θ−θ*)² = 0.5 * Σ 1·1 = 0.5*numel.
    with torch.no_grad():
        m.weight.add_(torch.ones_like(m.weight))
    penalty = ewc.penalty(m)
    expected = 0.5 * m.weight.numel()
    assert float(penalty.item()) == pytest.approx(expected, rel=1e-5)


def test_penalty_grad_flows_to_theta() -> None:
    """The penalty must be differentiable w.r.t. parameters; that's how
    the regularizer enters the optimizer step."""
    m = _tiny_model()
    ewc = EWCSI(m, lam_ewc=1.0)
    for n in ewc.fisher:
        ewc.fisher[n].fill_(1.0)
    with torch.no_grad():
        m.weight.add_(0.5)

    m.zero_grad()
    penalty = ewc.penalty(m)
    penalty.backward()
    # dpenalty/dθ = F · (θ - θ*) = 1 · 0.5 = 0.5 everywhere.
    assert torch.allclose(m.weight.grad, torch.full_like(m.weight, 0.5), atol=1e-6)


def test_snapshot_anchor_zeroes_omega_keeps_fisher() -> None:
    m = _tiny_model()
    ewc = EWCSI(m)
    for n in ewc.fisher:
        ewc.fisher[n].fill_(7.0)
        ewc.omega[n].fill_(3.0)

    with torch.no_grad():
        m.weight.add_(1.0)
    ewc.snapshot_anchor(m)

    for n in ewc.omega:
        assert torch.equal(ewc.omega[n], torch.zeros_like(ewc.omega[n]))
    for n in ewc.fisher:
        assert torch.allclose(ewc.fisher[n], torch.full_like(ewc.fisher[n], 7.0))
    # Anchor now matches current weights.
    for n, p in m.named_parameters():
        assert torch.equal(ewc.anchor[n], p)


def test_invalid_fisher_decay_raises() -> None:
    m = _tiny_model()
    with pytest.raises(ValueError):
        EWCSI(m, fisher_decay=0.0)
    with pytest.raises(ValueError):
        EWCSI(m, fisher_decay=1.0)


def test_state_dict_round_trip() -> None:
    m = _tiny_model()
    ewc = EWCSI(m, fisher_decay=0.7, lam_ewc=2.0, lam_si=3.0)
    for n in ewc.fisher:
        ewc.fisher[n].fill_(1.5)
        ewc.omega[n].fill_(0.25)
    ewc.num_fisher_updates = 11

    state = ewc.state_dict()
    ewc2 = EWCSI(_tiny_model(), fisher_decay=0.9, lam_ewc=0.0, lam_si=0.0)
    ewc2.load_state_dict(state)

    assert ewc2.fisher_decay == 0.7
    assert ewc2.lam_ewc == 2.0
    assert ewc2.lam_si == 3.0
    assert ewc2.num_fisher_updates == 11
    for n in ewc.fisher:
        assert torch.equal(ewc.fisher[n], ewc2.fisher[n])
        assert torch.equal(ewc.omega[n], ewc2.omega[n])
        assert torch.equal(ewc.anchor[n], ewc2.anchor[n])
