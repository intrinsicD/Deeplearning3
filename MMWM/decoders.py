"""Decoder heads: text (causal autoregressive) and vector reconstruction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .containers import LatentState
from .helpers import MLP, RMSNorm
from .interfaces import DECODERS, IDecoder


@DECODERS.register("text_autoregressive_head")
class TextAutoregressiveHead(IDecoder):
    """Causally-masked autoregressive text decoder conditioned on a latent state.

    The latent is injected as a prefix token in the sequence. A causal
    self-attention transformer then predicts next-token logits at every
    position for autoregressive generation.
    """
    required_context_keys = ("prefix_tokens",)

    def __init__(
        self,
        vocab_size: int = 32000,
        latent_dim: int = 128,
        text_embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, text_embed_dim)
        self.token_proj = nn.Linear(text_embed_dim, hidden_dim) if text_embed_dim != hidden_dim else nn.Identity()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        if context is None or "prefix_tokens" not in context:
            raise ValueError("TextAutoregressiveHead requires context['prefix_tokens']")
        prefix_tokens: torch.Tensor = context["prefix_tokens"]  # [B, T]
        B, T = prefix_tokens.shape
        device = prefix_tokens.device
        max_prefix = self.max_seq_len - 1
        if T > max_prefix:
            raise ValueError(
                f"TextAutoregressiveHead prefix length {T} exceeds valid max {max_prefix} "
                f"(max_seq_len={self.max_seq_len}, one slot reserved for latent token)."
            )

        # Latent becomes position 0; token embeddings fill positions 1..T
        latent_token = self.latent_proj(latent.z_sem).unsqueeze(1)  # [B, 1, D]
        token_emb = self.token_proj(self.token_embed(prefix_tokens))  # [B, T, D]
        seq = torch.cat([latent_token, token_emb], dim=1)  # [B, 1+T, D]
        if seq.shape[1] > self.max_seq_len:
            raise ValueError(
                f"TextAutoregressiveHead received sequence length {seq.shape[1]} "
                f"(prefix={T} + 1 latent token) but pos_embed only has "
                f"{self.max_seq_len} slots. Increase max_seq_len or truncate inputs."
            )
        positions = torch.arange(seq.shape[1], device=device).unsqueeze(0)
        seq = seq + self.pos_embed(positions)

        causal_mask = self._causal_mask(seq.shape[1], device)
        h = self.transformer(seq, mask=causal_mask)
        h = self.norm(h)

        # Logits at positions 1..T predict tokens at positions 1..T
        logits = self.lm_head(h[:, 1:, :])  # [B, T, vocab]
        return {"text_logits": logits}


@DECODERS.register("vector_reconstruction")
class VectorReconstructionHead(IDecoder):
    def __init__(self, latent_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.head = MLP([latent_dim, latent_dim * 2, output_dim])

    def forward(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        return {"vector_recon": self.head(latent.z_sem)}


class ResidualUpsampleBlock(nn.Module):
    """Residual block with transposed convolution for spatial upsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.upsample_skip = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        h = h + self.conv(h)
        skip = self.upsample_skip(self.skip(x))
        return h + skip


@DECODERS.register("image_reconstruction")
class ImageDecoderHead(IDecoder):
    """Latent-conditioned convolutional image decoder.

    Projects latent to a spatial grid, then uses residual upsampling blocks
    with transposed convolutions to reconstruct the image.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 128,
        output_channels: int = 3,
        output_size: int = 64,
        init_spatial: int = 4,
    ) -> None:
        super().__init__()
        self.init_spatial = init_spatial
        self.output_size = output_size
        self.latent_to_spatial = nn.Linear(latent_dim, base_channels * init_spatial * init_spatial)
        self.norm = nn.GroupNorm(min(8, base_channels), base_channels)

        num_upsample = 0
        s = init_spatial
        while s < output_size:
            s *= 2
            num_upsample += 1

        blocks = []
        ch = base_channels
        for i in range(num_upsample):
            out_ch = max(ch // 2, 32)
            blocks.append(ResidualUpsampleBlock(ch, out_ch))
            ch = out_ch
        self.decoder_blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        z = latent.primary()
        spatial = self.latent_to_spatial(z)
        B = z.shape[0]
        ch = spatial.shape[-1] // (self.init_spatial * self.init_spatial)
        spatial = spatial.view(B, ch, self.init_spatial, self.init_spatial)
        spatial = self.norm(spatial)
        h = self.decoder_blocks(spatial)
        image = self.to_rgb(h)
        if image.shape[-1] != self.output_size:
            image = torch.nn.functional.interpolate(image, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return {"image_recon": image}


class ResidualUpsample1DBlock(nn.Module):
    """Residual block with 1D transposed convolution for temporal upsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.upsample_skip = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        h = h + self.conv(h)
        skip = self.upsample_skip(self.skip(x))
        return h + skip


@DECODERS.register("audio_reconstruction")
class AudioDecoderHead(IDecoder):
    """Latent-conditioned 1D convolutional audio decoder.

    Projects latent to temporal features, then uses 1D transposed convolutions
    to upsample to a spectrogram or waveform representation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 128,
        output_channels: int = 1,
        output_length: int = 1024,
        init_length: int = 8,
    ) -> None:
        super().__init__()
        self.init_length = init_length
        self.output_length = output_length
        self.latent_to_temporal = nn.Linear(latent_dim, base_channels * init_length)
        self.norm = nn.GroupNorm(min(8, base_channels), base_channels)

        num_upsample = 0
        s = init_length
        while s < output_length:
            s *= 2
            num_upsample += 1

        blocks = []
        ch = base_channels
        for i in range(num_upsample):
            out_ch = max(ch // 2, 16)
            blocks.append(ResidualUpsample1DBlock(ch, out_ch))
            ch = out_ch
        self.decoder_blocks = nn.Sequential(*blocks)
        self.to_audio = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
            nn.Conv1d(ch, output_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        z = latent.primary()
        temporal = self.latent_to_temporal(z)
        B = z.shape[0]
        ch = temporal.shape[-1] // self.init_length
        temporal = temporal.view(B, ch, self.init_length)
        temporal = self.norm(temporal)
        h = self.decoder_blocks(temporal)
        audio = self.to_audio(h)
        if audio.shape[-1] != self.output_length:
            audio = torch.nn.functional.interpolate(audio, size=self.output_length, mode="linear", align_corners=False)
        return {"audio_recon": audio}
