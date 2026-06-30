"""W0.5 — generate() uses hooks; forward_multimodal fuses all inputs (A9)."""

from __future__ import annotations

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel


def _model() -> OmniLatentModel:
    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    return OmniLatentModel(cfg).eval()


def test_generate_engages_hook_manager() -> None:
    model = _model()
    model.register_hook(
        LatentNeuralHook(
            name="probe",
            num_tokens=4,
            dim=model.config.hidden_dim,
            target_layers=[0, 1],
        )
    )

    calls = {"begin_forward": 0, "backbone_got_hooks": 0}
    orig_begin = model.hook_manager.begin_forward

    def spy_begin(B):
        calls["begin_forward"] += 1
        return orig_begin(B)

    model.hook_manager.begin_forward = spy_begin

    orig_backbone = model.backbone.forward

    def spy_backbone(*args, **kwargs):
        if kwargs.get("hook_manager") is not None:
            calls["backbone_got_hooks"] += 1
        return orig_backbone(*args, **kwargs)

    model.backbone.forward = spy_backbone

    image = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    out = model.generate("image", image, max_len=3)

    assert out.shape == (1, 3)
    # The hook path must be exercised during generation.
    assert calls["begin_forward"] >= 1
    assert calls["backbone_got_hooks"] >= 1


def test_generate_without_hooks_still_works() -> None:
    model = _model()
    image = torch.randn(1, 3, model.config.image_size, model.config.image_size)
    out = model.generate("image", image, max_len=4)
    assert out.shape == (1, 4)


def test_forward_multimodal_fuses_all_inputs() -> None:
    model = _model()
    cfg = model.config
    inputs = {
        "text": torch.randint(1, cfg.vocab_size, (2, 6)),
        "image": torch.randn(2, 3, cfg.image_size, cfg.image_size),
    }
    results = model.forward_multimodal(inputs, target_modalities=["text"])

    # The fused prefix must contain BOTH sources, not just one.
    srcs = results["text"]["source_modalities"]
    assert set(srcs) == {"text", "image"}


def test_forward_multimodal_each_target_sees_every_source() -> None:
    model = _model()
    cfg = model.config
    inputs = {
        "text": torch.randint(1, cfg.vocab_size, (2, 5)),
        "image": torch.randn(2, 3, cfg.image_size, cfg.image_size),
    }
    results = model.forward_multimodal(inputs)
    assert set(results.keys()) == {"text", "image"}
    for tgt in results:
        assert set(results[tgt]["source_modalities"]) == {"text", "image"}
