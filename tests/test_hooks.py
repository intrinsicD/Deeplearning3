"""Tests for the Latent Neural Hook system."""

from __future__ import annotations

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import HookManager, LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.utils import count_parameters


@pytest.fixture
def config() -> OmniLatentConfig:
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        gradient_checkpointing=False,
        vocab_size=256,
        image_size=32,
        image_patch_size=8,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        audio_n_mels=32,
    )


@pytest.fixture
def model(config: OmniLatentConfig) -> OmniLatentModel:
    return OmniLatentModel(config)


class TestHookCreation:
    def test_create_hook(self, config: OmniLatentConfig) -> None:
        hook = LatentNeuralHook(
            name="test_hook",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=[0, 1, 2, 3],
        )
        assert hook.name == "test_hook"
        assert hook.num_tokens == 4
        assert hook.dim == config.hidden_dim

    def test_hook_has_parameters(self, config: OmniLatentConfig) -> None:
        hook = LatentNeuralHook(
            name="test",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=[0, 1],
        )
        n = count_parameters(hook)
        assert n > 0

    def test_gate_init_near_zero(self, config: OmniLatentConfig) -> None:
        hook = LatentNeuralHook(
            name="test",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=[0],
            gate_bias_init=-4.0,
        )
        gate_val = hook.gate_value(0).item()
        assert gate_val < 0.05  # sigmoid(-4) ≈ 0.018

    def test_gate_scales_hook_tokens(self, config: OmniLatentConfig) -> None:
        """Hook tokens scaled by gate should be near-zero at init."""
        hook = LatentNeuralHook(
            name="test",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=[0],
            gate_bias_init=-4.0,
        )
        tokens = hook.get_hook_tokens(batch_size=2)
        gated = tokens * hook.gate_value(0)
        # sigmoid(-4) ≈ 0.018, so gated tokens should be very small
        assert gated.abs().max().item() < 0.01


class TestHookManager:
    def test_register_and_remove(self, config: OmniLatentConfig) -> None:
        manager = HookManager()
        hook = LatentNeuralHook("h1", 4, config.hidden_dim, [0])
        manager.register_hook(hook)
        assert manager.has_hooks()
        removed = manager.remove_hook("h1")
        assert removed is not None
        assert not manager.has_hooks()

    def test_remove_nonexistent(self) -> None:
        manager = HookManager()
        assert manager.remove_hook("nope") is None

    def test_multiple_hooks(self, config: OmniLatentConfig) -> None:
        manager = HookManager()
        h1 = LatentNeuralHook("h1", 4, config.hidden_dim, [0, 1])
        h2 = LatentNeuralHook("h2", 8, config.hidden_dim, [2, 3])
        manager.register_hook(h1)
        manager.register_hook(h2)
        assert len(manager.hooks) == 2


class TestHookIntegration:
    def test_forward_with_hook(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
    ) -> None:
        hook = LatentNeuralHook(
            name="test",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=list(range(config.num_layers)),
        )
        model.register_hook(hook)

        text = torch.randint(1, config.vocab_size, (2, 16))
        result = model.reconstruct("text", text)
        assert result["output"] is not None
        assert not torch.isnan(result["output"]).any()

    def test_hook_does_not_change_shape(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
    ) -> None:
        text = torch.randint(1, config.vocab_size, (2, 16))

        # Without hook
        result_no_hook = model.reconstruct("text", text)
        shape_no_hook = result_no_hook["output"].shape

        # With hook
        hook = LatentNeuralHook(
            name="test",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=[0, 1],
        )
        model.register_hook(hook)
        result_hook = model.reconstruct("text", text)
        shape_hook = result_hook["output"].shape

        model.remove_hook("test")

        # Shapes must match -- hooks don't change the output shape
        assert shape_no_hook == shape_hook

    def test_hook_gradient_flow(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
    ) -> None:
        """Verify gradients flow through hooks."""
        hook = LatentNeuralHook(
            name="grad_test",
            num_tokens=4,
            dim=config.hidden_dim,
            target_layers=[0, 1, 2, 3],
        )
        model.register_hook(hook)

        text = torch.randint(1, config.vocab_size, (2, 16))
        result = model.reconstruct("text", text)

        loss = result["output"].sum()
        loss.backward()

        # Hook tokens should have gradients
        assert hook.hook_tokens.grad is not None
        assert hook.hook_tokens.grad.abs().sum() > 0

        # Gates should have gradients
        for g in hook.gates.values():
            assert g.grad is not None

        model.remove_hook("grad_test")

    def test_multiple_hooks_compose(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
    ) -> None:
        """Multiple hooks can be active simultaneously."""
        h1 = LatentNeuralHook("h1", 2, config.hidden_dim, [0, 1])
        h2 = LatentNeuralHook("h2", 3, config.hidden_dim, [1, 2, 3])
        model.register_hook(h1)
        model.register_hook(h2)

        text = torch.randint(1, config.vocab_size, (2, 16))
        result = model.reconstruct("text", text)
        assert not torch.isnan(result["output"]).any()

        model.remove_hook("h1")
        model.remove_hook("h2")

    def test_hook_on_cross_modal(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
    ) -> None:
        """Hooks work for cross-modal passes too."""
        hook = LatentNeuralHook(
            "cross_test", 4, config.hidden_dim, [0, 2]
        )
        model.register_hook(hook)

        image = torch.randn(2, 3, config.image_size, config.image_size)
        result = model("image", image, "text")
        assert not torch.isnan(result["output"]).any()

        model.remove_hook("cross_test")
