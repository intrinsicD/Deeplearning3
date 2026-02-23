"""Temporal context modules for video understanding.

Implements two complementary approaches for learning long-range temporal
structure from video:

Approach 2 — Hierarchical Clip-then-Sequence Model:
    A lightweight causal transformer that operates on clip-level latent
    vectors.  The main encoder processes each clip independently (with
    no_grad for memory efficiency), producing a 768-dim latent per clip.
    The temporal transformer then models the sequence of clip latents,
    enabling next-clip prediction, clip infilling, and scene boundary
    detection over minute-scale timespans.

    Memory cost: N_clips x 768 dims = ~60 tokens for a minute of video.
    This is tiny compared to the 2048-token backbone sequence limit.

Approach 3 — Recurrent Memory Tokens:
    A small set of learnable memory tokens (default 8) that persist across
    consecutive clips from the same video.  Memory tokens are prepended to
    the input sequence, participate in self-attention, and their updated
    states carry forward to the next clip via truncated BPTT.

    Memory cost: 8 extra tokens per forward pass — negligible.

Both approaches are designed for 8GB VRAM single-GPU training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.model.layers import RMSNorm, SwiGLU, precompute_rope_freqs, apply_rope


# =========================================================================
# Approach 2: Temporal Sequence Transformer
# =========================================================================

class TemporalAttention(nn.Module):
    """Multi-head causal self-attention for temporal clip sequences."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rope_freqs is not None:
            q, k = apply_rope(q, k, rope_freqs)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=attn_mask is None,  # default to causal if no mask
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class TemporalTransformerBlock(nn.Module):
    """Pre-norm transformer block for temporal sequence modeling."""

    def __init__(
        self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = TemporalAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalSequenceTransformer(nn.Module):
    """Hierarchical temporal transformer operating on clip-level latents.

    Input: sequence of clip latent vectors (B, N_clips, D)
    Output: predicted next-clip latents (B, N_clips, D)

    The transformer uses causal attention so each position can only attend
    to previous clips.  This enables autoregressive next-clip prediction.

    Training workflow:
        1. Encode N clips independently through the main model (no_grad)
        2. Collect latent vectors: z_0, z_1, ..., z_{N-1}
        3. Feed through this transformer
        4. Train on next-clip prediction: predict z_{t+1} from z_{0..t}

    This covers ~1 minute of video (60 clips x 2s each) in a single
    training step using only ~60 tokens worth of memory.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim
        num_layers = config.temporal_seq_layers
        num_heads = config.temporal_seq_heads
        mlp_dim = int(D * config.mlp_ratio)
        dropout = config.temporal_seq_dropout

        # Input projection (clip latents may need adaptation)
        self.input_proj = nn.Linear(D, D, bias=False)
        self.input_norm = RMSNorm(D)

        # Learned positional embedding for clip positions
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.temporal_seq_max_clips, D) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(D, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(D)

        # Output projection for next-clip prediction
        self.next_clip_head = nn.Linear(D, D, bias=False)

        # Scene boundary detection head (binary per clip)
        self.scene_boundary_head = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.SiLU(),
            nn.Linear(D // 4, 1),
        )

        # RoPE for temporal positions
        rope = precompute_rope_freqs(D // num_heads, config.temporal_seq_max_clips)
        self.register_buffer("rope_freqs", rope, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the prediction head for stable start
        nn.init.zeros_(self.next_clip_head.weight)

    def forward(
        self,
        clip_latents: torch.Tensor,
        clip_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the temporal transformer.

        Args:
            clip_latents: (B, N, D) sequence of clip latent vectors.
            clip_mask: (B, N) bool mask — True for valid clips.

        Returns dict with:
            next_clip_pred: (B, N, D) predicted next clip latents
            scene_boundary: (B, N, 1) scene boundary logits
            temporal_latent: (B, N, D) contextual clip representations
        """
        B, N, D = clip_latents.shape

        # Project and normalize input
        x = self.input_norm(self.input_proj(clip_latents))

        # Add positional embeddings
        x = x + self.pos_embed[:, :N]

        # Build causal attention mask
        causal_mask = torch.tril(
            torch.ones(N, N, dtype=torch.bool, device=x.device)
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        # Apply padding mask if provided
        if clip_mask is not None:
            # Expand mask: (B, 1, 1, N) — mask out padding positions
            pad_mask = clip_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            causal_mask = causal_mask & pad_mask

        # Transformer layers
        for layer in self.layers:
            x = layer(x, self.rope_freqs, causal_mask)

        x = self.final_norm(x)

        return {
            "next_clip_pred": self.next_clip_head(x),      # (B, N, D)
            "scene_boundary": self.scene_boundary_head(x),  # (B, N, 1)
            "temporal_latent": x,                            # (B, N, D)
        }

    def predict_next_clip(
        self,
        clip_latents: torch.Tensor,
        clip_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convenience: return only next-clip predictions."""
        return self.forward(clip_latents, clip_mask)["next_clip_pred"]


# =========================================================================
# Approach 3: Recurrent Memory Tokens
# =========================================================================

class RecurrentMemory(nn.Module):
    """Recurrent memory tokens that persist across video clips.

    Implements the Recurrent Memory Transformer (RMT) pattern:
    memory tokens are prepended to each clip's input sequence,
    participate in self-attention, and their updated states carry
    forward to the next clip.

    Usage in training:
        memory = RecurrentMemory(config)
        mem_state = memory.init_state(batch_size)  # or None for first clip

        for clip_data in video_clips:
            # Prepend memory to input
            x_with_mem, mem_len = memory.prepend(clip_tokens, mem_state)

            # Run through backbone (memory tokens participate in attention)
            output = backbone(x_with_mem, ...)

            # Extract updated memory and content
            mem_state, content = memory.extract(output, mem_len)

            # mem_state carries forward to next clip
            # Backprop only through current clip (truncated BPTT)
            if truncate:
                mem_state = memory.detach_state(mem_state)

    Memory cost: num_tokens (8) extra tokens per forward pass.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config
        self.num_tokens = config.memory_num_tokens
        self.dim = config.hidden_dim

        # Learnable initial memory state
        self.init_tokens = nn.Parameter(
            torch.randn(1, config.memory_num_tokens, config.hidden_dim) * 0.02
        )

        # Per-layer gating: controls how much memory influences attention
        # Starts nearly silent (like hooks)
        self.gate = nn.Parameter(
            torch.tensor(config.memory_gate_bias_init)
        )

        # Memory state evolution network (between clips)
        self.state_transform = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False),
        )

        # Read/write projections for memory interaction
        self.write_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=False),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init the state transform so it starts as identity
        nn.init.zeros_(self.state_transform[-1].weight)
        # Initialize write gate to pass through ~50%
        nn.init.zeros_(self.write_gate[0].weight)

    @property
    def device(self) -> torch.device:
        return self.init_tokens.device

    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize memory state for a new video sequence.

        Returns: (B, num_tokens, D) initial memory state.
        """
        return self.init_tokens.expand(batch_size, -1, -1)

    def gate_value(self) -> torch.Tensor:
        """Current gate value (sigmoid-scaled)."""
        return torch.sigmoid(self.gate)

    def prepend(
        self,
        content_tokens: torch.Tensor,
        memory_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Prepend gated memory tokens to the content sequence.

        Args:
            content_tokens: (B, N, D) content sequence.
            memory_state: (B, M, D) current memory state, or None for init.

        Returns:
            combined: (B, M + N, D) memory + content sequence.
            mem_len: number of memory tokens prepended.
        """
        B = content_tokens.shape[0]

        if memory_state is None:
            memory_state = self.init_state(B)

        # Apply gate to control memory influence
        gated_memory = memory_state * self.gate_value()

        combined = torch.cat([gated_memory, content_tokens], dim=1)
        return combined, self.num_tokens

    def extract(
        self,
        output: torch.Tensor,
        mem_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract updated memory state and content from backbone output.

        Args:
            output: (B, M + N, D) backbone output with memory prefix.
            mem_len: number of memory tokens at the start.

        Returns:
            new_memory: (B, M, D) updated memory state.
            content: (B, N, D) content output (without memory tokens).
        """
        raw_memory = output[:, :mem_len]
        content = output[:, mem_len:]

        # Update memory through state transform (residual) and write gate
        transformed = raw_memory + self.state_transform(raw_memory)
        write_g = self.write_gate(transformed)
        new_memory = write_g * transformed + (1 - write_g) * raw_memory

        return new_memory, content

    def detach_state(self, memory_state: torch.Tensor) -> torch.Tensor:
        """Detach memory state for truncated BPTT.

        Call this between clips to prevent backprop through the full
        video sequence (which would exceed VRAM).
        """
        return memory_state.detach()

    def expand_attention_mask(
        self,
        attn_mask: torch.Tensor,
        mem_len: int,
    ) -> torch.Tensor:
        """Expand an attention mask to account for prepended memory tokens.

        Delegates to centralized masking module.

        Args:
            attn_mask: (1, 1, N, N) or (B, 1, N, N) existing mask.
            mem_len: number of memory tokens prepended.

        Returns: expanded mask (*, M+N, M+N).
        """
        from omnilatent.model.masking import expand_mask_for_memory
        return expand_mask_for_memory(attn_mask, mem_len)
