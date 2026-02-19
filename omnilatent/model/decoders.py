"""Modality decoders: project latent tokens back to raw signals.

Each decoder takes a subsequence of latent tokens (B, N, D) from the
backbone and produces modality-specific output.

All decoders are lightweight -- the backbone does the heavy compute.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from omnilatent.config import OmniLatentConfig
from omnilatent.model.layers import RMSNorm


# ---------------------------------------------------------------------------
# Text Decoder  (project to vocabulary logits)
# ---------------------------------------------------------------------------
class TextDecoder(nn.Module):
    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → logits (B, N, vocab_size)."""
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# Audio Decoder  (latent tokens → mel spectrogram via transposed convs)
# ---------------------------------------------------------------------------
class AudioDecoder(nn.Module):
    """Mirror of AudioEncoder: latent tokens → mel spectrogram."""

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        D = config.hidden_dim
        self.norm = RMSNorm(D)
        self.deconv_stack = nn.Sequential(
            # (B, D, T') → (B, D//2, T')
            nn.ConvTranspose1d(D, D // 2, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            # (B, D//2, T') → (B, D//4, T'*2)
            nn.ConvTranspose1d(D // 2, D // 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            # (B, D//4, T'*2) → (B, n_mels, T'*4)
            nn.ConvTranspose1d(D // 4, config.audio_n_mels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → mel: (B, n_mels, T_reconstructed)."""
        x = self.norm(x)
        x = x.transpose(1, 2)        # (B, D, N)
        return self.deconv_stack(x)   # (B, n_mels, T_out)


# ---------------------------------------------------------------------------
# Image Decoder  (latent tokens → pixel patches → image)
# ---------------------------------------------------------------------------
class ImageDecoder(nn.Module):
    """Projects latent patch tokens back to pixel space.

    Each token is projected to a flat patch of pixels, then reshaped
    into the full image.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config
        P = config.image_patch_size
        C = config.image_channels
        D = config.hidden_dim
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, C * P * P, bias=True)
        self.grid_size = config.image_size // P

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_patches, D) → images: (B, C, H, W).

        N_patches must equal grid_size^2.
        """
        x = self.head(self.norm(x))   # (B, N, C*P*P)
        x = rearrange(
            x,
            "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
            gh=self.grid_size,
            gw=self.grid_size,
            ph=self.config.image_patch_size,
            pw=self.config.image_patch_size,
            c=self.config.image_channels,
        )
        return x


# ---------------------------------------------------------------------------
# Video Decoder  (latent tokens → video frames)
# ---------------------------------------------------------------------------
class VideoDecoder(nn.Module):
    """Mirror of VideoEncoder: latent tokens → video tensor."""

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.video_channels
        TP = config.video_temporal_patch
        SP = config.video_patch_size
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, C * TP * SP * SP, bias=True)
        self.spatial_grid = config.video_size // SP

    def forward(
        self, x: torch.Tensor, num_temporal_tokens: int | None = None
    ) -> torch.Tensor:
        """x: (B, N_tokens, D) → video: (B, C, T_frames, H, W).

        N_tokens = T_temporal * spatial_grid^2.
        """
        B = x.shape[0]
        gs = self.spatial_grid
        if num_temporal_tokens is None:
            num_temporal_tokens = x.shape[1] // (gs * gs)

        x = self.head(self.norm(x))   # (B, N, C*TP*SP*SP)

        C = self.config.video_channels
        TP = self.config.video_temporal_patch
        SP = self.config.video_patch_size
        x = rearrange(
            x,
            "b (gt gh gw) (c tp sp1 sp2) -> b c (gt tp) (gh sp1) (gw sp2)",
            gt=num_temporal_tokens,
            gh=gs,
            gw=gs,
            c=C,
            tp=TP,
            sp1=SP,
            sp2=SP,
        )
        return x
