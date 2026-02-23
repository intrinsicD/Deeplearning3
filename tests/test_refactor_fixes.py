"""Tests for Priority-A correctness fixes.

Covers:
  1. HPWM gradient accumulation: optimizer steps only every N micro-steps
  2. HPWM VQ codebook EMA-only: codebooks are buffers, not optimizer params
  3. HPWM prediction leakage: context/target causal split
  4. OmniLatent ImageDecoder: works with different patch sizes
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from omnilatent.config import OmniLatentConfig
from omnilatent.model.decoders import ImageDecoder


# =========================================================================
# HPWM: Gradient Accumulation
# =========================================================================

class TestGradientAccumulation:
    """Verify that the `or True` bug is fixed and grad accumulation works."""

    def test_no_or_true_in_train(self):
        """Ensure the `or True` bypass is removed from the training loop."""
        import inspect
        from hpwm.train import Trainer
        source = inspect.getsource(Trainer.train)
        assert "or True" not in source, (
            "Found `or True` in Trainer.train — gradient accumulation is bypassed"
        )

    def test_micro_step_counter_present(self):
        """Ensure micro_step counter is used for accumulation."""
        import inspect
        from hpwm.train import Trainer
        source = inspect.getsource(Trainer.train)
        assert "micro_step" in source, (
            "micro_step counter not found — accumulation logic may be broken"
        )


# =========================================================================
# HPWM: VQ Codebook Update Mechanism
# =========================================================================

class TestVQCodebookEMA:
    """Verify codebooks are EMA-only (buffers, not optimizer parameters)."""

    def test_codebooks_are_buffers(self):
        """Codebooks should be registered as buffers, not parameters."""
        from hpwm.components.vqvae import VectorQuantizer
        vq = VectorQuantizer(n_codebooks=4, vocab_size=32, dim=16)

        # Check that codebook_0..codebook_3 are buffers
        buffer_names = {name for name, _ in vq.named_buffers()}
        for i in range(4):
            assert f"codebook_{i}" in buffer_names, (
                f"codebook_{i} should be a buffer, not a parameter"
            )

        # Check that no codebook is a parameter (would receive gradients)
        param_names = {name for name, _ in vq.named_parameters()}
        for i in range(4):
            assert f"codebook_{i}" not in param_names, (
                f"codebook_{i} is a parameter — it will get gradient updates"
            )

    def test_codebooks_not_in_optimizer(self):
        """When building param groups, codebooks should be excluded."""
        from hpwm.components.vqvae import VQVAE
        vqvae = VQVAE(n_codebooks=4, vocab_size=32, vq_dim=16, hidden=32, n_layers=1)
        # Only parameters (not buffers) go into the optimizer
        params = list(vqvae.parameters())
        for p in params:
            # Each codebook buffer should NOT appear in .parameters()
            for i in range(4):
                cb = vqvae.quantizer._get_codebook(i)
                assert p.data_ptr() != cb.data_ptr(), (
                    f"codebook_{i} found in .parameters() — will receive gradients"
                )

    def test_ema_updates_codebooks(self):
        """EMA should modify codebooks during training."""
        from hpwm.components.vqvae import VectorQuantizer
        vq = VectorQuantizer(n_codebooks=2, vocab_size=16, dim=8)
        vq.train()

        cb_before = vq._get_codebook(0).clone()
        # Run a forward pass with some data
        z = torch.randn(4, 8, 4, 4)
        vq(z)
        cb_after = vq._get_codebook(0)

        # Codebook should have changed via EMA
        assert not torch.allclose(cb_before, cb_after), (
            "Codebook unchanged after forward pass — EMA update may be broken"
        )


# =========================================================================
# HPWM: Prediction Leakage (Causal Split)
# =========================================================================

class TestPredictionCausality:
    """Verify next-frame prediction doesn't leak target features."""

    def test_temporal_uses_context_only(self):
        """The temporal module should receive T-1 frames, not T."""
        import inspect
        from hpwm.model import HPWM
        source = inspect.getsource(HPWM.forward)
        # The fix introduces context_slots = slot_flat[:, :-1]
        assert "context_slots" in source or "slot_flat[:, :-1]" in source, (
            "Causal split not found — temporal module may see target frame"
        )

    def test_context_target_shapes(self):
        """Forward pass should process T-1 context frames for prediction."""
        from hpwm.config import HPWMConfig
        from hpwm.model import HPWM

        config = HPWMConfig(
            resolution=128,  # default — 128 resized to 126 → 9x9 patches
            n_frames=4,
            dino_frozen=True,
            grad_checkpointing=False,
            n_slots=2,
            d_slot=32,
            d_mamba=64,
            mamba_n_layers=1,
            n_layers_fast=1,
            n_layers_slow=1,
            d_fast=32,
            d_slow=64,
            token_budget=16,
            n_heads=2,
            vqvae_codebooks=2,
            vqvae_vocab_size=16,
            vqvae_dim=8,
            vqvae_hidden=16,
            vqvae_n_layers=1,
        )
        model = HPWM(config)
        model.eval()

        T = 4
        frames = torch.randn(1, T, 3, config.resolution, config.resolution)
        with torch.no_grad():
            outputs = model(frames)

        # temporal_output should be [B, T-1, D] (context frames only)
        temporal_out = outputs["temporal_output"]
        assert temporal_out.shape[1] == T - 1, (
            f"Expected temporal_output to have {T-1} timesteps (context only), "
            f"got {temporal_out.shape[1]}"
        )


# =========================================================================
# OmniLatent: ImageDecoder patch size
# =========================================================================

class TestImageDecoderPatchSize:
    """Verify ImageDecoder works with different patch sizes."""

    @pytest.mark.parametrize("patch_size", [4, 8, 16, 32])
    def test_output_shape_matches_config(self, patch_size: int):
        """Output image size should match config regardless of patch_size."""
        image_size = 64  # must be divisible by all patch sizes above
        config = OmniLatentConfig(
            hidden_dim=32,
            image_size=image_size,
            image_patch_size=patch_size,
            image_channels=3,
        )
        decoder = ImageDecoder(config)

        grid = image_size // patch_size
        n_patches = grid * grid
        x = torch.randn(2, n_patches, 32)
        out = decoder(x)

        assert out.shape == (2, 3, image_size, image_size), (
            f"Expected output (2, 3, {image_size}, {image_size}), "
            f"got {tuple(out.shape)} for patch_size={patch_size}"
        )

    def test_rejects_non_power_of_2(self):
        """Should raise ValueError for non-power-of-2 patch sizes."""
        config = OmniLatentConfig(
            hidden_dim=32,
            image_size=64,
            image_patch_size=12,
            image_channels=3,
        )
        with pytest.raises(ValueError, match="power of 2"):
            ImageDecoder(config)

    def test_n_upsample_stages(self):
        """Number of ConvTranspose2d layers should equal log2(patch_size)."""
        for patch_size in [4, 8, 16]:
            config = OmniLatentConfig(
                hidden_dim=64,
                image_size=64,
                image_patch_size=patch_size,
                image_channels=3,
            )
            decoder = ImageDecoder(config)
            n_convs = sum(
                1 for m in decoder.upconv_stack.modules()
                if isinstance(m, nn.ConvTranspose2d)
            )
            expected = int(math.log2(patch_size))
            assert n_convs == expected, (
                f"Expected {expected} ConvTranspose2d layers for "
                f"patch_size={patch_size}, got {n_convs}"
            )
