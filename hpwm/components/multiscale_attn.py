"""
Component 5: Hierarchical Multi-Scale Attention.

Phase -1: 2 scales (fast + slow), 4+6 layers, 512 token budget.
Full spec: 3 scales, 12+18+24 layers, 2K token budget.

Uses a Perceiver-like architecture: fixed-size learned latent arrays
cross-attend to variable-length temporal features, followed by
self-attention within each scale and cross-scale bridge attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries attend to key-value pairs from another source."""

    def __init__(self, d_model: int, n_heads: int = 4, d_kv: int | None = None):
        super().__init__()
        d_kv = d_kv or d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_kv)

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_kv, d_model, bias=False)
        self.to_v = nn.Linear(d_kv, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

        self.scale = self.d_head ** -0.5

    def forward(
        self, queries: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, N_q, D]
            context: [B, N_kv, D_kv]

        Returns:
            [B, N_q, D]
        """
        q = self.to_q(self.norm_q(queries))
        k = self.to_k(self.norm_kv(context))
        v = self.to_v(self.norm_kv(context))

        B, N_q, _ = q.shape
        N_kv = k.shape[1]

        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return queries + self.to_out(out)


class SelfAttentionBlock(nn.Module):
    """Standard self-attention with pre-norm."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        h = self.norm(x)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.n_heads) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return x + self.to_out(out)


class FFNBlock(nn.Module):
    """Feed-forward network with pre-norm."""

    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttentionScale(nn.Module):
    """
    Single attention scale with Perceiver-style cross-attention intake
    followed by self-attention layers.

    Learned latent tokens cross-attend to temporal features,
    then refine through self-attention.
    """

    def __init__(
        self,
        d_model: int,
        d_input: int,
        n_latents: int,
        n_layers: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents

        # Learned latent tokens
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)

        # Cross-attention intake: latents attend to temporal features
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, d_kv=d_input)

        # Self-attention refinement layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                SelfAttentionBlock(d_model, n_heads),
                FFNBlock(d_model),
            ]))

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: [B, T, D_input] from temporal state

        Returns:
            latents: [B, N_latents, D_model] refined scale representations
        """
        B = temporal_features.shape[0]

        # Expand latents for batch
        x = self.latents.expand(B, -1, -1)

        # Cross-attend to temporal features
        x = self.cross_attn(x, temporal_features)

        # Self-attention refinement
        for self_attn, ffn in self.layers:
            x = self_attn(x)
            x = ffn(x)

        return self.norm_out(x)


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention with cross-scale bridge.

    Phase -1: 2 scales (fast + slow).
    - Fast scale: operates on recent temporal context
    - Slow scale: operates on full temporal context
    - Cross-scale bridge: information flows between scales

    Full spec: 3 scales with Phi_S backed by Mamba persistent state.
    """

    def __init__(
        self,
        d_input: int = 1024,     # n_slots * d_slot
        d_fast: int = 128,
        d_slow: int = 256,
        n_latents: int = 512,    # token budget per scale
        n_layers_fast: int = 4,
        n_layers_slow: int = 6,
        n_heads: int = 4,
        fast_window_ratio: float = 0.25,  # fast scale sees last 25% of frames
    ):
        super().__init__()
        self.fast_window_ratio = fast_window_ratio

        # Fast scale
        self.fast_scale = AttentionScale(
            d_model=d_fast,
            d_input=d_input,
            n_latents=n_latents,
            n_layers=n_layers_fast,
            n_heads=n_heads,
        )

        # Slow scale
        self.slow_scale = AttentionScale(
            d_model=d_slow,
            d_input=d_input,
            n_latents=n_latents,
            n_layers=n_layers_slow,
            n_heads=n_heads,
        )

        # Cross-scale bridge: fast <-> slow
        self.bridge_fast_to_slow = CrossAttentionBlock(
            d_model=d_slow, n_heads=n_heads, d_kv=d_fast,
        )
        self.bridge_slow_to_fast = CrossAttentionBlock(
            d_model=d_fast, n_heads=n_heads, d_kv=d_slow,
        )

        # Final projection to unified dim
        self.proj_fast = nn.Linear(d_fast, d_input)
        self.proj_slow = nn.Linear(d_slow, d_input)

    def forward(
        self, temporal_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            temporal_features: [B, T, D_input] from temporal state

        Returns:
            dict with:
                "fast": [B, N_latents, D_input] fast scale output
                "slow": [B, N_latents, D_input] slow scale output
                "combined": [B, N_latents, D_input] bridge-integrated output
        """
        B, T, D = temporal_features.shape

        # Split temporal context for fast/slow scales
        fast_window = max(1, int(T * self.fast_window_ratio))
        fast_input = temporal_features[:, -fast_window:]  # recent context
        slow_input = temporal_features  # full context

        # Process each scale
        fast_latents = self.fast_scale(fast_input)   # [B, N_lat, D_fast]
        slow_latents = self.slow_scale(slow_input)   # [B, N_lat, D_slow]

        # Cross-scale bridge
        slow_updated = self.bridge_fast_to_slow(slow_latents, fast_latents)
        fast_updated = self.bridge_slow_to_fast(fast_latents, slow_latents)

        # Project to unified dimension
        fast_out = self.proj_fast(fast_updated)   # [B, N_lat, D_input]
        slow_out = self.proj_slow(slow_updated)   # [B, N_lat, D_input]

        # Combined: average of both scales
        combined = (fast_out + slow_out) / 2.0

        return {
            "fast": fast_out,
            "slow": slow_out,
            "combined": combined,
        }
