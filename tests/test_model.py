"""Tests for the OmniLatent model: construction, forward passes, shapes."""

from __future__ import annotations

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.utils import ALL_MODALITIES, count_parameters


@pytest.fixture
def config() -> OmniLatentConfig:
    """Small config for fast tests."""
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        gradient_checkpointing=False,
        vocab_size=256,
        text_max_len=32,
        image_size=32,
        image_patch_size=8,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        audio_n_mels=32,
        audio_max_frames=64,
    )


@pytest.fixture
def model(config: OmniLatentConfig) -> OmniLatentModel:
    return OmniLatentModel(config)


@pytest.fixture
def sample_data(config: OmniLatentConfig) -> dict[str, torch.Tensor]:
    B = 2
    return {
        "text": torch.randint(1, config.vocab_size, (B, 16)),
        "audio": torch.randn(B, config.audio_n_mels, 64),
        "image": torch.randn(B, 3, config.image_size, config.image_size),
        "video": torch.randn(
            B, 3, config.video_max_frames, config.video_size, config.video_size
        ),
    }


class TestModelConstruction:
    def test_model_creates(self, model: OmniLatentModel) -> None:
        assert model is not None

    def test_parameter_count_reasonable(self, model: OmniLatentModel) -> None:
        n = count_parameters(model)
        # Should be well under 200M for this small config
        assert n < 10_000_000  # < 10M for the tiny test config
        assert n > 0

    def test_has_all_encoders(self, model: OmniLatentModel) -> None:
        for mod in ALL_MODALITIES:
            assert mod in model.encoders

    def test_has_all_decoders(self, model: OmniLatentModel) -> None:
        for mod in ALL_MODALITIES:
            assert mod in model.decoders


class TestForwardPass:
    def test_self_reconstruction_text(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model.reconstruct("text", sample_data["text"])
        B, T = sample_data["text"].shape
        # Output should be (B, T, vocab_size)
        assert result["output"].shape[0] == B
        assert result["output"].shape[2] == model.config.vocab_size

    def test_self_reconstruction_audio(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model.reconstruct("audio", sample_data["audio"])
        B = sample_data["audio"].shape[0]
        assert result["output"].shape[0] == B
        assert result["output"].shape[1] == model.config.audio_n_mels

    def test_self_reconstruction_image(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model.reconstruct("image", sample_data["image"])
        B = sample_data["image"].shape[0]
        assert result["output"].shape == (
            B,
            model.config.image_channels,
            model.config.image_size,
            model.config.image_size,
        )

    def test_self_reconstruction_video(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model.reconstruct("video", sample_data["video"])
        B = sample_data["video"].shape[0]
        assert result["output"].shape[0] == B
        assert result["output"].shape[1] == model.config.video_channels

    def test_cross_modal_image_to_text(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model("image", sample_data["image"], "text")
        assert result["output"].shape[0] == sample_data["image"].shape[0]
        assert result["output"].shape[2] == model.config.vocab_size

    def test_cross_modal_text_to_image(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model("text", sample_data["text"], "image")
        # Text decoder for image: each text token gets decoded as an image patch
        assert result["output"].shape[0] == sample_data["text"].shape[0]

    def test_all_modality_pairs(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Verify all 16 modality combinations produce valid output."""
        for src in ALL_MODALITIES:
            for tgt in ALL_MODALITIES:
                result = model(src, sample_data[src], tgt)
                assert result["output"] is not None
                assert result["output"].shape[0] == 2  # batch size
                assert not torch.isnan(result["output"]).any(), (
                    f"NaN in {src}→{tgt}"
                )


class TestMultiModalForward:
    def test_multi_input(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        results = model.forward_multimodal(
            inputs={"text": sample_data["text"], "image": sample_data["image"]},
            target_modalities=["text", "image"],
        )
        assert "text" in results
        assert "image" in results

    def test_all_modalities_input(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        results = model.forward_multimodal(
            inputs=sample_data,
            target_modalities=ALL_MODALITIES,
        )
        for mod in ALL_MODALITIES:
            assert mod in results
            assert not torch.isnan(results[mod]["output"]).any()
