"""W3.1 — content-conditioned hook gates: route weight scales/skip hooks."""

from __future__ import annotations

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook, NeuralPortManager
from omnilatent.model.omnilatent import OmniLatentModel


def _model() -> OmniLatentModel:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    return OmniLatentModel(cfg).eval()


def _hook(model: OmniLatentModel, gate_bias: float = 0.0) -> str:
    model.register_hook(
        LatentNeuralHook(
            name="h", num_tokens=4, dim=model.config.hidden_dim,
            target_layers=[0, 1], gate_bias_init=gate_bias,
        )
    )
    return "h"


def _recon(model: OmniLatentModel, img: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.reconstruct("image", img)["output"]


def test_route_weight_zero_gives_exact_recovery() -> None:
    torch.manual_seed(0)
    model = _model()
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)

    out_nohook = _recon(model, img)          # no hook registered yet
    _hook(model, gate_bias=2.0)              # strong hook so it would matter
    model.hook_manager.set_route_weights({"h": 0.0})
    out_zero = _recon(model, img)

    # Skipping a route-weight-0 hook reproduces the no-hook output exactly.
    torch.testing.assert_close(out_zero, out_nohook)


def test_route_weight_one_matches_unconditioned() -> None:
    torch.manual_seed(0)
    model = _model()
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    _hook(model, gate_bias=2.0)

    out_unconditioned = _recon(model, img)   # no route weights set
    model.hook_manager.set_route_weights({"h": 1.0})
    out_one = _recon(model, img)

    torch.testing.assert_close(out_one, out_unconditioned)


def test_intermediate_weight_changes_output() -> None:
    torch.manual_seed(0)
    model = _model()
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    _hook(model, gate_bias=2.0)

    model.hook_manager.set_route_weights({"h": 1.0})
    out_full = _recon(model, img)
    model.hook_manager.set_route_weights({"h": 0.3})
    out_partial = _recon(model, img)
    assert not torch.allclose(out_full, out_partial)


def test_clearing_route_weights_restores_default() -> None:
    model = _model()
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    _hook(model)
    base = _recon(model, img)
    model.hook_manager.set_route_weights({"h": 0.0})
    model.hook_manager.set_route_weights(None)  # clear
    restored = _recon(model, img)
    torch.testing.assert_close(restored, base)


def test_effective_gate_is_logged() -> None:
    model = _model()
    img = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    _hook(model, gate_bias=0.0)  # sigmoid(0) = 0.5 static gate
    model.hook_manager.set_route_weights({"h": 0.5})
    _recon(model, img)
    log = model.hook_manager.gate_log()
    # effective gate = 0.5 (static) * 0.5 (route) = 0.25
    assert any(abs(v - 0.25) < 1e-4 for v in log.values())


def test_per_batch_route_weights() -> None:
    # A (B,) weight zeroing one batch row is allowed (the hook stays active for
    # the batch as a whole since not all rows are zero).
    mgr = NeuralPortManager()
    mgr.register_hook(LatentNeuralHook(name="h", num_tokens=2, dim=16, target_layers=[0]))
    mgr.begin_forward(3)
    mgr.set_route_weights({"h": torch.tensor([0.0, 1.0, 1.0])})
    x = torch.randn(3, 5, 16)
    out = mgr.pre_layer(0, x)
    # 2 hook tokens appended for the whole batch.
    assert out.shape == (3, 7, 16)
    # Row 0 (weight 0) contributes zero-valued hook tokens.
    assert torch.allclose(out[0, 5:], torch.zeros(2, 16))
    assert not torch.allclose(out[1, 5:], torch.zeros(2, 16))
