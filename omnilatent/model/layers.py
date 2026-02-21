"""Custom layers: RMSNorm, SwiGLU, Rotary Position Embeddings, QK-Norm.

All layers are designed for stable mixed-precision training with full
gradient flow.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# RMSNorm  (Zhang & Sennrich, 2019)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization -- faster than LayerNorm and
    works well with mixed precision."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for the norm computation → numerical stability
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# SwiGLU  (Shazeer, 2020 -- used in PaLM, LLaMA)
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: effective hidden dim is 2/3 of mlp_dim
    because one projection is used as the gate."""

    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, mlp_dim, bias=False)  # gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Rotary Position Embeddings  (Su et al., 2021)
# ---------------------------------------------------------------------------
def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10_000.0) -> torch.Tensor:
    """Precompute complex-valued rotation frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_freqs: torch.Tensor,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys.

    Args:
        q, k: (batch, heads, seq_len, head_dim)
        rope_freqs: precomputed complex frequencies
        offset: position offset (useful for modality-specific offsets)
    """
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    # Reshape to complex pairs
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # Slice frequencies for the current sequence
    freqs = rope_freqs[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    # Only use as many frequency dims as we have
    freqs = freqs[..., : q_complex.shape[-1]]

    q_out = torch.view_as_real(q_complex * freqs).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


# ---------------------------------------------------------------------------
# QK-Norm  (Dehghani et al., 2023 -- stabilizes attention)
# ---------------------------------------------------------------------------
class QKNorm(nn.Module):
    """Per-head L2-normalization of queries and keys before attention.
    Prevents attention logits from growing with head dimension."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(head_dim) * (head_dim ** -0.25))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(q.float(), dim=-1).type_as(q) * self.scale
        k = F.normalize(k.float(), dim=-1).type_as(k) * self.scale
        return q, k


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with RoPE + QK-Norm
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.qk_norm = QKNorm(self.head_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        rope_offset: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, H, D)
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # QK-Norm for stable attention
        q, k = self.qk_norm(q, k)

        # Rotary position embeddings
        if rope_freqs is not None:
            q, k = apply_rope(q, k, rope_freqs, offset=rope_offset)

        # Scaled dot-product attention (uses FlashAttention when available)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=1.0,  # already scaled by QKNorm
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm with RMSNorm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim, dropout)
        # LayerScale (Touvron et al., 2021 — CaiT / DeiT-III):
        # learnable per-channel scaling on residual branches, initialized
        # small so early layers don't dominate the residual stream.
        self.ls1 = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.ls2 = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        rope_offset: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x), rope_freqs, rope_offset, attn_mask)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x
