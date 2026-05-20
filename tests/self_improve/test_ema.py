"""Tests for :class:`EMATeacher`.

Invariants:

- Initialization bit-copies the student.
- Update step matches ``θ_ema = d·θ_ema + (1-d)·θ_student``.
- ``swap_into`` restores the original weights on exit even if the inner
  block raises.
- The teacher's forward is well-defined on the same inputs as the student.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scripts.training.self_improve import EMATeacher


def _tiny_model() -> nn.Module:
    m = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return m


def test_init_copies_student() -> None:
    m = _tiny_model()
    ema = EMATeacher(m, decay=0.9)
    for k, p in m.named_parameters():
        assert torch.equal(ema._params[k], p)


def test_update_blends_correctly() -> None:
    m = _tiny_model()
    init_params = {k: p.detach().clone() for k, p in m.named_parameters()}
    ema = EMATeacher(m, decay=0.9)

    # Mutate the student.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p))  # student = init + 1

    ema.update(m)
    # Expected EMA = 0.9 * init + 0.1 * (init + 1) = init + 0.1
    for k, p in m.named_parameters():
        expected = init_params[k] + 0.1
        assert torch.allclose(ema._params[k], expected, atol=1e-6)


def test_update_increments_counter() -> None:
    m = _tiny_model()
    ema = EMATeacher(m, decay=0.99)
    assert ema.num_updates == 0
    ema.update(m)
    ema.update(m)
    assert ema.num_updates == 2


def test_swap_into_restores_on_normal_exit() -> None:
    m = _tiny_model()
    ema = EMATeacher(m, decay=0.5)
    # Make EMA distinct from student.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.ones_like(p))
    ema.update(m)
    # Now student = init + 1, ema = init + 0.5.
    student_snapshot = {k: p.detach().clone() for k, p in m.named_parameters()}

    with ema.swap_into(m):
        # Inside: model weights == ema._params
        for k, p in m.named_parameters():
            assert torch.equal(p, ema._params[k])

    # After: model weights restored.
    for k, p in m.named_parameters():
        assert torch.equal(p, student_snapshot[k])


def test_swap_into_restores_on_exception() -> None:
    m = _tiny_model()
    ema = EMATeacher(m, decay=0.5)
    ema.update(m)  # ema diverges 0
    student_snapshot = {k: p.detach().clone() for k, p in m.named_parameters()}

    with pytest.raises(RuntimeError, match="boom"):
        with ema.swap_into(m):
            raise RuntimeError("boom")

    for k, p in m.named_parameters():
        assert torch.equal(p, student_snapshot[k])


def test_invalid_decay_raises() -> None:
    m = _tiny_model()
    with pytest.raises(ValueError):
        EMATeacher(m, decay=0.0)
    with pytest.raises(ValueError):
        EMATeacher(m, decay=1.0)


def test_state_dict_round_trip() -> None:
    m = _tiny_model()
    ema = EMATeacher(m, decay=0.9)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.7)
    ema.update(m)
    ema.update(m)
    state = ema.state_dict()

    ema2 = EMATeacher(_tiny_model(), decay=0.5)
    ema2.load_state_dict(state)
    assert ema2.decay == 0.9
    assert ema2.num_updates == 2
    for k in ema._params:
        assert torch.equal(ema._params[k], ema2._params[k])


def test_swap_into_produces_distinct_forward() -> None:
    """Whole point of the EMA: forward(student) ≠ forward(teacher) once
    the EMA has lagged the student."""
    m = _tiny_model()
    ema = EMATeacher(m, decay=0.5)
    # Bump the student substantially.
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(3.0)
    ema.update(m)  # EMA = 0.5·init + 0.5·(3·init) = 2·init
    # Now student ≠ EMA.
    x = torch.randn(2, 4)
    y_student = m(x).detach()
    with ema.swap_into(m), torch.no_grad():
        y_teacher = m(x).detach()
    assert not torch.allclose(y_student, y_teacher)
