"""Latent Reasoning Module — Chain of Continuous Thought.

Adds internal "thinking" to the model through learnable thought tokens
processed by dedicated reasoning layers.  Unlike chain-of-thought prompting
(which generates explicit text tokens), latent reasoning operates entirely
in the model's hidden space — the thoughts are never decoded to any
modality.

Architecture:
  1. N learnable thought tokens (default 16) are initialized as parameters.
  2. They are concatenated with the encoded source tokens.
  3. A small dedicated transformer stack (R layers, default 4) processes
     the combined sequence with bidirectional attention.
  4. The refined thought tokens are extracted and returned.
  5. In the main model, thought tokens become part of the prefix:
     [source_tokens, thought_tokens, target_queries]
     so the decoder can attend to the model's "reasoning state".

Key properties:
  * **Gated activation** — a sigmoid gate (initialized near-zero) scales
    thought token magnitude so they start nearly silent and don't
    destabilize early training.
  * **Dedicated layers** — reasoning uses its own transformer layers
    (not backbone layers), so the backbone's learned representations
    aren't disrupted.
  * **Parameter efficient** — with 4 layers at 768-dim, the reasoning
    module adds ~15M params (~10% of the 140M backbone).
  * **Optional** — controlled by config.reasoning_enabled; when disabled,
    the model behaves exactly as before.
  * **Trainable end-to-end** — thought tokens receive gradients through
    the backbone and decoder via the target reconstruction loss.
  * **Auxiliary bottleneck loss** — optionally trains a projection from
    thought tokens back to source latent space, encouraging thoughts to
    compress/distill useful information.

Inspired by:
  - "Training Large Language Models to Reason in a Continuous Latent Space"
    (Hao et al., 2024 — Coconut)
  - "Think before you speak: Training Language Models With Pause Tokens"
    (Goyal et al., 2024)
  - Perceiver / Perceiver IO latent arrays (Jaegle et al., 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.model.layers import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_freqs,
)


class LatentReasoningModule(nn.Module):
    """Chain of Continuous Thought via dedicated reasoning transformer.

    Processes source tokens + learnable thought tokens through a small
    transformer stack, producing refined thought tokens that capture
    the model's internal reasoning state.

    Args:
        config: OmniLatentConfig with reasoning parameters.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim
        self.num_thoughts = config.reasoning_num_thoughts

        # Learnable thought tokens
        self.thought_tokens = nn.Parameter(
            torch.randn(1, config.reasoning_num_thoughts, D) * 0.02
        )

        # Thought-specific positional embedding
        self.thought_pos = nn.Parameter(
            torch.randn(1, config.reasoning_num_thoughts, D) * 0.02
        )

        # Dedicated reasoning transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=D,
                num_heads=config.reasoning_num_heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
            )
            for _ in range(config.reasoning_num_layers)
        ])
        self.norm = RMSNorm(D)

        # Pre-compute RoPE for reasoning layers
        max_reason_len = config.max_seq_len + config.reasoning_num_thoughts
        rope = precompute_rope_freqs(
            D // config.reasoning_num_heads,
            max_reason_len,
        )
        self.register_buffer("rope_freqs", rope, persistent=False)

        # Sigmoid gate — starts near-zero for stable training
        self.gate_bias = nn.Parameter(
            torch.tensor(config.reasoning_gate_bias_init)
        )

        # Bottleneck projection for auxiliary loss:
        # thought_tokens → reconstructed source summary
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(D, D, bias=False),
            nn.SiLU(),
            nn.Linear(D, D, bias=False),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-init bottleneck output so aux loss starts at zero."""
        nn.init.zeros_(self.bottleneck_proj[-1].weight)

    @property
    def gate(self) -> torch.Tensor:
        """Current gate value (0 to 1)."""
        return torch.sigmoid(self.gate_bias)

    def forward(
        self,
        src_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run latent reasoning on source tokens.

        Args:
            src_tokens: (B, S, D) encoded source tokens.

        Returns:
            thought_out: (B, N, D) refined thought tokens (gated).
            bottleneck_pred: (B, D) predicted source summary from thoughts,
                used for auxiliary bottleneck loss.
        """
        B, S, D = src_tokens.shape

        # Expand thought tokens to batch size + add positional encoding
        thoughts = self.thought_tokens.expand(B, -1, -1) + self.thought_pos

        # Concatenate: [source_tokens, thought_tokens]
        # Source tokens are frozen context; thoughts are the reasoning workspace
        combined = torch.cat([src_tokens, thoughts], dim=1)

        # Bidirectional attention: everything sees everything
        # (source provides context, thoughts develop reasoning)
        for layer in self.layers:
            combined = layer(combined, self.rope_freqs)

        combined = self.norm(combined)

        # Extract refined thought tokens
        thought_out = combined[:, S:]  # (B, N, D)

        # Apply gate for gradual activation
        thought_out = thought_out * self.gate

        # Bottleneck: can we recover source summary from thoughts?
        # Mean-pool thoughts and project
        thought_summary = thought_out.mean(dim=1)  # (B, D)
        bottleneck_pred = self.bottleneck_proj(thought_summary)  # (B, D)

        return thought_out, bottleneck_pred


class ReasoningBottleneckLoss(nn.Module):
    """Auxiliary loss encouraging thought tokens to compress source information.

    Compares the reasoning module's bottleneck prediction against the
    mean-pooled source latent.  This forces thoughts to actually encode
    useful information rather than being ignored.

    Uses MSE + cosine similarity for robust latent-space matching.
    """

    def __init__(self, cosine_weight: float = 0.5) -> None:
        super().__init__()
        self.cosine_weight = cosine_weight

    def forward(
        self,
        bottleneck_pred: torch.Tensor,
        source_summary: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bottleneck reconstruction loss.

        Args:
            bottleneck_pred: (B, D) prediction from thought tokens.
            source_summary: (B, D) mean-pooled source latent (target).

        Returns scalar loss.
        """
        mse = F.mse_loss(bottleneck_pred, source_summary.detach())
        cos_sim = F.cosine_similarity(
            bottleneck_pred, source_summary.detach(), dim=-1
        )
        cos_loss = (1.0 - cos_sim).mean()
        return mse + self.cosine_weight * cos_loss
