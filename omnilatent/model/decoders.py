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
        self.decoder_arch = getattr(config, "image_decoder", "deconv")

        if self.decoder_arch == "patch":
            self.patch_size = P
            self.patch_head = nn.Linear(D, config.image_channels * P * P)
            return
        if self.decoder_arch == "gaussian":
            self.gaussians_per_token = config.image_gaussians_per_token
            self.gaussian_chunk_size = config.image_gaussian_chunk_size
            if self.gaussians_per_token <= 0:
                raise ValueError(
                    f"image_gaussians_per_token must be > 0, got {self.gaussians_per_token}"
                )
            if self.gaussian_chunk_size <= 0:
                raise ValueError(
                    f"image_gaussian_chunk_size must be > 0, got {self.gaussian_chunk_size}"
                )
            self.gaussian_head = nn.Linear(D, 9 * self.gaussians_per_token)
            self.gaussian_min_scale = config.image_gaussian_min_scale
            self.gaussian_max_scale = config.image_gaussian_max_scale
            self.gaussian_offset_scale = config.image_gaussian_offset_scale
            self.gaussian_anchor_jitter = config.image_gaussian_anchor_jitter
            if not (0 < self.gaussian_min_scale <= self.gaussian_max_scale):
                raise ValueError(
                    "image_gaussian_min_scale must be > 0 and <= image_gaussian_max_scale"
                )
            if self.gaussian_offset_scale < 0:
                raise ValueError("image_gaussian_offset_scale must be >= 0")
            if self.gaussian_anchor_jitter < 0:
                raise ValueError("image_gaussian_anchor_jitter must be >= 0")
            ys, xs = torch.meshgrid(
                torch.linspace(-1.0, 1.0, config.image_size),
                torch.linspace(-1.0, 1.0, config.image_size),
                indexing="ij",
            )
            self.register_buffer("render_x", xs.view(1, 1, config.image_size, config.image_size), persistent=False)
            self.register_buffer("render_y", ys.view(1, 1, config.image_size, config.image_size), persistent=False)
            gy, gx = torch.meshgrid(
                torch.arange(self.grid_size, dtype=torch.float32),
                torch.arange(self.grid_size, dtype=torch.float32),
                indexing="ij",
            )
            anchors_x = ((gx + 0.5) / self.grid_size) * 2.0 - 1.0
            anchors_y = ((gy + 0.5) / self.grid_size) * 2.0 - 1.0
            token_anchors = torch.stack([anchors_x, anchors_y], dim=-1).view(1, -1, 2)
            anchors = self._expand_subcell_anchors(token_anchors)
            self.register_buffer("gaussian_anchors", anchors, persistent=False)
            self.background = nn.Parameter(torch.zeros(1, config.image_channels, 1, 1))
            return
        if self.decoder_arch != "deconv":
            raise ValueError(f"Unknown image_decoder {self.decoder_arch!r}")

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
        if self.decoder_arch == "patch":
            patches = torch.sigmoid(self.patch_head(x))
            p = self.patch_size
            return rearrange(
                patches,
                "b (gh gw) (c p1 p2) -> b c (gh p1) (gw p2)",
                gh=self.grid_size,
                gw=self.grid_size,
                c=self.config.image_channels,
                p1=p,
                p2=p,
            )
        if self.decoder_arch == "gaussian":
            return self._forward_gaussian(x)

        # Reshape 1D sequence to 2D spatial grid: (B, N, D) -> (B, D, G, G)
        x = rearrange(x, "b (gh gw) d -> b d gh gw", gh=self.grid_size, gw=self.grid_size)
        # Apply deconvolutions to reconstruct the image
        return self.upconv_stack(x)

    def _forward_gaussian(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        raw = self.gaussian_head(x).view(B, x.shape[1], self.gaussians_per_token, 9)
        raw = raw.reshape(B, x.shape[1] * self.gaussians_per_token, 9)
        subgrid_size = math.ceil(math.sqrt(self.gaussians_per_token))
        offset_limit = self.gaussian_offset_scale / (self.grid_size * subgrid_size)
        centers = self.gaussian_anchors.to(dtype=x.dtype) + torch.tanh(raw[..., :2]) * offset_limit
        if self.training and self.gaussian_anchor_jitter > 0:
            jitter = (torch.rand_like(centers) * 2.0 - 1.0) * offset_limit * self.gaussian_anchor_jitter
            centers = centers + jitter
        centers = centers.clamp(-1.0, 1.0)
        scales = self.gaussian_min_scale + torch.sigmoid(raw[..., 2:4]) * (
            self.gaussian_max_scale - self.gaussian_min_scale
        )
        angles = torch.tanh(raw[..., 4:5]) * math.pi
        colors = torch.sigmoid(raw[..., 5:8])
        opacity = torch.sigmoid(raw[..., 8:9])

        grid_x = self.render_x.to(device=x.device, dtype=x.dtype)
        grid_y = self.render_y.to(device=x.device, dtype=x.dtype)

        H = self.config.image_size
        W = self.config.image_size
        color_sum = torch.zeros(B, self.config.image_channels, H, W, device=x.device, dtype=x.dtype)
        weight_sum = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        for start in range(0, centers.shape[1], self.gaussian_chunk_size):
            end = min(start + self.gaussian_chunk_size, centers.shape[1])
            center_chunk = centers[:, start:end]
            scale_chunk = scales[:, start:end]
            color_chunk = colors[:, start:end]
            opacity_chunk = opacity[:, start:end]

            cx = center_chunk[..., 0].unsqueeze(-1).unsqueeze(-1)
            cy = center_chunk[..., 1].unsqueeze(-1).unsqueeze(-1)
            sx = scale_chunk[..., 0].unsqueeze(-1).unsqueeze(-1)
            sy = scale_chunk[..., 1].unsqueeze(-1).unsqueeze(-1)
            angle_chunk = angles[:, start:end]
            cos_t = torch.cos(angle_chunk).unsqueeze(-1)
            sin_t = torch.sin(angle_chunk).unsqueeze(-1)
            dx = grid_x - cx
            dy = grid_y - cy
            xr = cos_t * dx + sin_t * dy
            yr = -sin_t * dx + cos_t * dy
            exponent = -0.5 * ((xr / sx).square() + (yr / sy).square())
            weights = torch.exp(exponent) * opacity_chunk.squeeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight_sum = weight_sum + weights.sum(dim=1, keepdim=True)
            color_sum = color_sum + torch.einsum("bnhw,bnc->bchw", weights, color_chunk)

        avg_color = color_sum / weight_sum.clamp_min(1e-6)
        alpha = 1.0 - torch.exp(-weight_sum)
        background = torch.sigmoid(self.background).to(device=x.device, dtype=x.dtype)
        return avg_color * alpha + background * (1.0 - alpha)

    def _expand_subcell_anchors(self, token_anchors: torch.Tensor) -> torch.Tensor:
        """Place multiple Gaussian anchors on a regular sub-grid per image token."""
        if self.gaussians_per_token == 1:
            return token_anchors
        subgrid_size = math.ceil(math.sqrt(self.gaussians_per_token))
        idx = torch.arange(self.gaussians_per_token, dtype=torch.float32)
        sub_y = torch.div(idx, subgrid_size, rounding_mode="floor")
        sub_x = idx.remainder(subgrid_size)
        cell_width = 2.0 / self.grid_size
        local_x = ((sub_x + 0.5) / subgrid_size - 0.5) * cell_width
        local_y = ((sub_y + 0.5) / subgrid_size - 0.5) * cell_width
        local_offsets = torch.stack([local_x, local_y], dim=-1).view(1, 1, -1, 2)
        anchors = token_anchors.unsqueeze(2) + local_offsets
        return anchors.view(1, -1, 2).clamp(-1.0, 1.0)


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
