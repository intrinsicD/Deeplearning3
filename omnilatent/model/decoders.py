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
    into the full image.  The number of 2x upsampling stages is derived
    from ``image_patch_size`` so it works for any power-of-2 patch size.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim
        P = config.image_patch_size
        self.grid_size = config.image_size // P
        self.norm = RMSNorm(D)

        # Number of 2x upsampling stages = log2(patch_size)
        n_upsample = int(math.log2(P))
        if 2 ** n_upsample != P:
            raise ValueError(
                f"image_patch_size must be a power of 2, got {P}"
            )

        # Build transposed-conv stack: each stage does 2x spatial upsampling
        # and halves channels until the final stage outputs image_channels.
        layers: list[nn.Module] = []
        ch_in = D
        for i in range(n_upsample):
            is_last = i == n_upsample - 1
            ch_out = config.image_channels if is_last else max(ch_in // 2, config.image_channels)
            layers.append(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
            )
            if not is_last:
                layers.append(nn.SiLU())
            ch_in = ch_out
        self.upconv_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        # Reshape 1D sequence to 2D spatial grid: (B, N, D) -> (B, D, G, G)
        x = rearrange(x, "b (gh gw) d -> b d gh gw", gh=self.grid_size, gw=self.grid_size)
        # Apply deconvolutions to reconstruct the image
        return self.upconv_stack(x)


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
