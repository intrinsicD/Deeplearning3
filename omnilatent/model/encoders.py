"""Modality encoders: project raw signals into the unified latent space.

Each encoder converts a modality-specific input tensor into a sequence of
latent tokens (B, N, D) that the backbone transformer can process.

Design for 8 GB: encoders are deliberately lightweight (conv stacks /
patch embeddings), with the heavy lifting done by the shared backbone.
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
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.norm = RMSNorm(config.hidden_dim)
        self.scale = math.sqrt(config.hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) long tensor → (B, T, D)."""
        return self.norm(self.tok_embed(token_ids) * self.scale)


# ---------------------------------------------------------------------------
# Audio Encoder  (mel spectrogram → tokens via 1-D conv stack)
# ---------------------------------------------------------------------------
class AudioEncoder(nn.Module):
    """Converts a mel spectrogram (B, n_mels, T_frames) into latent tokens.

    Uses a stack of 1-D convolutions that progressively compress the time
    axis while expanding channels to hidden_dim.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        self.conv_stack = nn.Sequential(
            # (B, n_mels, T) → (B, D//4, T//2)
            nn.Conv1d(config.audio_n_mels, D // 4, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            # (B, D//4, T//2) → (B, D//2, T//4)
            nn.Conv1d(D // 4, D // 2, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            # (B, D//2, T//4) → (B, D, T//4)
            nn.Conv1d(D // 2, D, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.norm = RMSNorm(D)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T_frames) → (B, N_tokens, D)."""
        x = self.conv_stack(mel)        # (B, D, T')
        x = x.transpose(1, 2)           # (B, T', D)
        return self.norm(x)


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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) → (B, N_patches, D)."""
        x = self.patch_embed(images)             # (B, D, H', W')
        x = rearrange(x, "b d h w -> b (h w) d")
        return self.norm(x)


# ---------------------------------------------------------------------------
# Video Encoder  (spatio-temporal patch embedding with temporal compression)
# ---------------------------------------------------------------------------
class VideoEncoder(nn.Module):
    """Converts a video tensor into latent tokens.

    Uses a 3-D convolution to create spatio-temporal patches, then
    flattens into a token sequence.  For a 16-frame 112x112 video with
    temporal_patch=4, spatial_patch=16:
        temporal tokens  = 16/4 = 4
        spatial tokens   = (112/16)^2 = 49
        total tokens     = 4 * 49 = 196  (very manageable!)
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

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, C, T_frames, H, W) → (B, N_tokens, D)."""
        x = self.patch_embed(video)  # (B, D, T', H', W')
        x = rearrange(x, "b d t h w -> b (t h w) d")
        return self.norm(x)
