"""W0.7 — the public MMWM AV training path constructs without the A6 bugs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from MMWM.config import ModelConfig, build_model
from MMWM.decoders import ImageDecoderHead
from MMWM.losses import WorldModelLoss

REPO = Path(__file__).resolve().parents[2]


def test_script_parses() -> None:
    src = (REPO / "scripts" / "training" / "train_mmwm_av.py").read_text()
    ast.parse(src)  # raises SyntaxError on regression


def test_image_decoder_uses_output_kwargs() -> None:
    # The decoder expects output_channels/output_size; the script previously
    # passed out_channels/out_size (Audit.md A6).
    ImageDecoderHead(latent_dim=512, output_channels=3, output_size=64)
    with pytest.raises(TypeError):
        ImageDecoderHead(latent_dim=512, out_channels=3, out_size=64)


def test_model_builds_with_av_decoder_configs() -> None:
    cfg = ModelConfig(
        encoder_kwargs={
            "text_vocab_size": 256,
            "text_embed_dim": 256,
            "vector_input_dim": 16,
            "image_channels": 3,
            "audio_channels": 1,
            "hidden_dim": 256,
        },
        action_encoder_kwargs={"action_dim": 8, "action_embed_dim": 128},
        decoder_configs=[
            ("text_autoregressive_head", {
                "vocab_size": 256, "latent_dim": 128,
                "text_embed_dim": 256, "hidden_dim": 256,
            }),
            ("image_reconstruction", {
                "latent_dim": 512, "output_channels": 3, "output_size": 64,
            }),
        ],
    )
    build_model(cfg, skip_validation=True)


def test_worldmodelloss_signature_not_modelconfig() -> None:
    # First positional arg is `weights` (LossWeights), not a ModelConfig.
    loss = WorldModelLoss(learned_uncertainty=True)
    assert len(list(loss.parameters())) == 11
