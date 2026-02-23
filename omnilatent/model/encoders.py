"""Modality encoders: project raw signals into the unified latent space.

Each encoder converts a modality-specific input tensor into a sequence of
latent tokens (B, N, D) that the backbone transformer can process.

Design for 8 GB: encoders are deliberately lightweight (conv stacks /
patch embeddings), with the heavy lifting done by the shared backbone.

All spatial/temporal encoders include learned absolute positional embeddings,
since the backbone uses RoPE only for 1-D sequence position.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from omnilatent.config import OmniLatentConfig
from omnilatent.model.layers import RMSNorm


class ModalityEmbedding(nn.Module):
    """Prepends a learnable modality-indicator token to a sequence."""

    def __init__(self, num_modalities: int, dim: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(num_modalities, dim)

    def forward(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        B = x.shape[0]
        mod_tok = self.tokens(
            torch.tensor(modality_id, device=x.device)
        ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        return torch.cat([mod_tok, x], dim=1)


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        self.tok_embed = nn.Embedding(config.vocab_size, D)
        # Projection MLP to spread information beyond the embedding subspace.
        # Without this, the encoder output is rank-limited to the embedding
        # matrix rank, which collapses to ~64 during training.
        self.proj = nn.Sequential(
            nn.Linear(D, D, bias=False),
            nn.SiLU(),
            nn.Linear(D, D, bias=False),
        )
        self.norm = RMSNorm(D)
        self.scale = math.sqrt(D)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) long tensor → (B, T, D)."""
        x = self.tok_embed(token_ids) * self.scale
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# Audio Encoder  (mel spectrogram → tokens via 1-D conv stack)
# ---------------------------------------------------------------------------
class AudioEncoder(nn.Module):
    """Converts a mel spectrogram (B, n_mels, T_frames) into latent tokens.

    Uses a stack of 1-D convolutions that progressively compress the time
    axis while expanding channels to hidden_dim.  A residual refinement
    layer at full width prevents information loss from the strided
    downsampling stages.  Includes learned positional embeddings.

    The conv stack has a total stride of 4 (2×2), which must match
    audio_patch_frames.  Changing audio_patch_frames without updating
    the conv strides will break shape consistency.
    """

    # Total downsampling factor of the conv stack (stride 2 × stride 2)
    ENCODER_STRIDE: int = 4

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        if config.audio_patch_frames != self.ENCODER_STRIDE:
            raise ValueError(
                f"audio_patch_frames={config.audio_patch_frames} does not match "
                f"AudioEncoder conv stride={self.ENCODER_STRIDE}. "
                f"Either set audio_patch_frames={self.ENCODER_STRIDE} or update "
                f"the conv stack strides in AudioEncoder."
            )
        D = config.hidden_dim
        # Stage 1: (B, n_mels, T) → (B, D//4, T//2)
        self.conv1 = nn.Conv1d(config.audio_n_mels, D // 4, kernel_size=5, stride=2, padding=2)
        # Stage 2: (B, D//4, T//2) → (B, D//2, T//4)
        self.conv2 = nn.Conv1d(D // 4, D // 2, kernel_size=5, stride=2, padding=2)
        # Stage 3: (B, D//2, T//4) → (B, D, T//4)
        self.conv3 = nn.Conv1d(D // 2, D, kernel_size=3, stride=1, padding=1)
        # Residual refinement at full width to recover lost information
        self.conv4 = nn.Conv1d(D, D, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()
        self.norm = RMSNorm(D)
        # Positional embeddings: max tokens = audio_max_frames // audio_patch_frames
        max_tokens = config.audio_max_frames // config.audio_patch_frames
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, D) * 0.02)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T_frames) → (B, N_tokens, D)."""
        x = self.act(self.conv1(mel))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x + self.act(self.conv4(x))   # residual refinement
        x = x.transpose(1, 2)             # (B, T', D)
        return self.norm(x + self.pos_embed[:, :x.shape[1]])


# ---------------------------------------------------------------------------
# Image Encoder  (ViT-style patch embedding)
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        P = config.image_patch_size
        C = config.image_channels
        self.patch_embed = nn.Conv2d(
            C, D, kernel_size=P, stride=P, bias=True
        )
        self.norm = RMSNorm(D)
        # Learned 2D positional embedding (flattened)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.image_num_patches, D) * 0.02
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) → (B, N_patches, D)."""
        x = self.patch_embed(images)             # (B, D, H', W')
        x = rearrange(x, "b d h w -> b (h w) d")
        return self.norm(x + self.pos_embed)


# ---------------------------------------------------------------------------
# Video Encoder  (spatio-temporal patch embedding with temporal compression)
# ---------------------------------------------------------------------------
class VideoEncoder(nn.Module):
    """Converts a video tensor into latent tokens.

    Uses a 3-D convolution to create spatio-temporal patches, then
    flattens into a token sequence.  Includes learned spatio-temporal
    positional embeddings.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        self.patch_embed = nn.Conv3d(
            config.video_channels,
            D,
            kernel_size=(config.video_temporal_patch, config.video_patch_size, config.video_patch_size),
            stride=(config.video_temporal_patch, config.video_patch_size, config.video_patch_size),
            bias=True,
        )
        self.norm = RMSNorm(D)
        # Learned positional embedding
        num_patches = (
            (config.video_max_frames // config.video_temporal_patch)
            * config.video_spatial_patches
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, D) * 0.02)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, C, T_frames, H, W) → (B, N_tokens, D)."""
        x = self.patch_embed(video)  # (B, D, T', H', W')
        x = rearrange(x, "b d t h w -> b (t h w) d")
        return self.norm(x + self.pos_embed[:, :x.shape[1]])
