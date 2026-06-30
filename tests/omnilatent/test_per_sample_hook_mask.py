"""Bug 2 — a zero-weight sample gets exact no-hook behaviour within a routed batch."""

from __future__ import annotations

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel


def _model() -> OmniLatentModel:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    model = OmniLatentModel(cfg).eval()
    model.register_hook(
        LatentNeuralHook(
            name="X", num_tokens=4, dim=cfg.hidden_dim,
            target_layers=[0, 1], gate_bias_init=3.0,  # strong, so it clearly matters
        )
    )
    return model


def _recon(model: OmniLatentModel, img: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.reconstruct("image", img)["output"]


def test_zero_weight_sample_matches_no_hook_in_routed_batch() -> None:
    torch.manual_seed(0)
    model = _model()
    img = torch.randn(2, 3, model.config.image_size, model.config.image_size)

    # Sample 0 activates hook X (weight 1); sample 1 has weight 0.
    model.hook_manager.set_route_weights({"X": torch.tensor([1.0, 0.0])})
    routed = _recon(model, img)

    # Reference: a forward where hook X is inactive for BOTH samples.
    model.hook_manager.set_route_weights({"X": torch.tensor([0.0, 0.0])})
    no_hook = _recon(model, img)

    # Sample 1 (weight 0) must be byte-for-byte the no-hook result — the zero
    # hook tokens must not have diluted its attention.
    torch.testing.assert_close(routed[1], no_hook[1])
    # Sample 0 (weight 1) must differ (the hook is doing something).
    assert not torch.allclose(routed[0], no_hook[0])


def test_all_active_is_unaffected_by_masking() -> None:
    torch.manual_seed(0)
    model = _model()
    img = torch.randn(2, 3, model.config.image_size, model.config.image_size)
    # Both samples weight 1 → no per-sample masking → identical to scalar 1.
    model.hook_manager.set_route_weights({"X": torch.tensor([1.0, 1.0])})
    per_batch = _recon(model, img)
    model.hook_manager.set_route_weights({"X": 1.0})
    scalar = _recon(model, img)
    torch.testing.assert_close(per_batch, scalar)


def test_no_nan_from_masking() -> None:
    torch.manual_seed(1)
    model = _model()
    img = torch.randn(3, 3, model.config.image_size, model.config.image_size)
    model.hook_manager.set_route_weights({"X": torch.tensor([1.0, 0.0, 0.0])})
    out = _recon(model, img)
    assert torch.isfinite(out).all()
