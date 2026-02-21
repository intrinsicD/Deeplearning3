"""Tests for the Latent Reasoning Module (Chain of Continuous Thought)."""

from __future__ import annotations

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.model.reasoning import LatentReasoningModule, ReasoningBottleneckLoss
from omnilatent.utils import ALL_MODALITIES, count_parameters


@pytest.fixture
def config() -> OmniLatentConfig:
    """Small config with reasoning enabled for fast tests."""
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
        reasoning_enabled=True,
        reasoning_num_thoughts=8,
        reasoning_num_layers=2,
        reasoning_num_heads=4,
        reasoning_gate_bias_init=-4.0,
    )


@pytest.fixture
def config_no_reasoning() -> OmniLatentConfig:
    """Config with reasoning disabled."""
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
        reasoning_enabled=False,
    )


@pytest.fixture
def model(config: OmniLatentConfig) -> OmniLatentModel:
    return OmniLatentModel(config)


@pytest.fixture
def model_no_reasoning(config_no_reasoning: OmniLatentConfig) -> OmniLatentModel:
    return OmniLatentModel(config_no_reasoning)


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


class TestReasoningModuleConstruction:
    def test_module_creates(self, config: OmniLatentConfig) -> None:
        module = LatentReasoningModule(config)
        assert module is not None

    def test_parameter_count(self, config: OmniLatentConfig) -> None:
        module = LatentReasoningModule(config)
        n = count_parameters(module)
        assert n > 0
        # Should be much smaller than backbone
        assert n < 5_000_000

    def test_thought_token_shape(self, config: OmniLatentConfig) -> None:
        module = LatentReasoningModule(config)
        assert module.thought_tokens.shape == (
            1, config.reasoning_num_thoughts, config.hidden_dim
        )

    def test_gate_init_near_zero(self, config: OmniLatentConfig) -> None:
        module = LatentReasoningModule(config)
        gate = module.gate.item()
        assert gate < 0.05  # sigmoid(-4) ≈ 0.018

    def test_num_layers(self, config: OmniLatentConfig) -> None:
        module = LatentReasoningModule(config)
        assert len(module.layers) == config.reasoning_num_layers


class TestReasoningModuleForward:
    def test_forward_shapes(self, config: OmniLatentConfig) -> None:
        module = LatentReasoningModule(config)
        src = torch.randn(2, 10, config.hidden_dim)
        thought_out, bottleneck = module(src)
        assert thought_out.shape == (2, config.reasoning_num_thoughts, config.hidden_dim)
        assert bottleneck.shape == (2, config.hidden_dim)

    def test_no_nans(self, config: OmniLatentConfig) -> None:
        module = LatentReasoningModule(config)
        src = torch.randn(2, 10, config.hidden_dim)
        thought_out, bottleneck = module(src)
        assert not torch.isnan(thought_out).any()
        assert not torch.isnan(bottleneck).any()

    def test_thought_tokens_are_gated(self, config: OmniLatentConfig) -> None:
        """At init, thought tokens should be very small due to gating."""
        module = LatentReasoningModule(config)
        src = torch.randn(2, 10, config.hidden_dim)
        thought_out, _ = module(src)
        # With gate ≈ 0.018, outputs should be much smaller than 1
        assert thought_out.abs().mean().item() < 0.5


class TestReasoningIntegration:
    def test_model_with_reasoning_creates(self, model: OmniLatentModel) -> None:
        assert model.reasoning is not None

    def test_model_without_reasoning(self, model_no_reasoning: OmniLatentModel) -> None:
        assert model_no_reasoning.reasoning is None

    def test_forward_text_self_reconstruct(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model.reconstruct("text", sample_data["text"])
        assert result["output"] is not None
        assert not torch.isnan(result["output"]).any()
        # Should include reasoning outputs
        assert "reasoning_bottleneck" in result
        assert "source_summary" in result

    def test_forward_all_modality_pairs(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """All 16 modality pairs should work with reasoning enabled."""
        for src in ALL_MODALITIES:
            for tgt in ALL_MODALITIES:
                result = model(
                    src, sample_data[src], tgt,
                    target_data=sample_data.get(tgt),
                )
                assert result["output"] is not None
                assert result["output"].shape[0] == 2
                assert not torch.isnan(result["output"]).any(), f"NaN in {src}→{tgt}"

    def test_no_reasoning_output_when_disabled(
        self,
        model_no_reasoning: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        result = model_no_reasoning.reconstruct("text", sample_data["text"])
        assert "reasoning_bottleneck" not in result
        assert "source_summary" not in result

    def test_reasoning_gradient_flow(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Verify gradients flow through the reasoning module."""
        result = model.reconstruct("text", sample_data["text"])
        loss = result["output"].sum()
        loss.backward()

        # Thought tokens should receive gradients
        assert model.reasoning.thought_tokens.grad is not None
        assert model.reasoning.thought_tokens.grad.abs().sum() > 0

        # Gate should receive gradients
        assert model.reasoning.gate_bias.grad is not None

    def test_reasoning_with_hooks(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Reasoning and hooks should compose correctly."""
        from omnilatent.model.hooks import LatentNeuralHook

        hook = LatentNeuralHook(
            name="test",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=[0, 1],
        )
        model.register_hook(hook)

        result = model.reconstruct("text", sample_data["text"])
        assert not torch.isnan(result["output"]).any()

        model.remove_hook("test")

    def test_generate_with_reasoning(
        self,
        model: OmniLatentModel,
        sample_data: dict[str, torch.Tensor],
    ) -> None:
        """Autoregressive generation should work with reasoning."""
        generated = model.generate(
            "image",
            sample_data["image"],
            max_len=5,
        )
        assert generated.shape == (2, 5)
        assert not torch.isnan(generated.float()).any()


class TestBottleneckLoss:
    def test_loss_computes(self) -> None:
        loss_fn = ReasoningBottleneckLoss()
        pred = torch.randn(2, 64)
        target = torch.randn(2, 64)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_loss_zero_for_identical(self) -> None:
        loss_fn = ReasoningBottleneckLoss()
        x = torch.randn(2, 64)
        loss = loss_fn(x, x)
        # Should be near-zero (MSE=0, cosine=1)
        assert loss.item() < 0.01

    def test_loss_gradient(self) -> None:
        loss_fn = ReasoningBottleneckLoss()
        pred = torch.randn(2, 64, requires_grad=True)
        target = torch.randn(2, 64)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0
