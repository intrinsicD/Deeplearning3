"""Unified Transformer backbone.

This is the shared core that processes all modalities in a common latent
space.  It is deliberately modality-agnostic: it receives a sequence of
latent tokens and returns a sequence of latent tokens.  Modality encoders
project raw signals into this space; modality decoders project them back.

Key design choices for 8 GB single-GPU training:
  * Pre-norm (RMSNorm) for training stability
  * SwiGLU MLP for parameter efficiency
  * RoPE for flexible sequence lengths
  * QK-Norm to prevent attention entropy collapse
  * Gradient checkpointing per layer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from omnilatent.config import OmniLatentConfig
from omnilatent.model.layers import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_freqs,
)
from omnilatent.model.masking import HookPolicy, expand_mask_for_hooks


class UnifiedTransformer(nn.Module):
    """The shared transformer backbone operating in the unified latent space.

    The backbone does NOT know about modalities.  It takes in token
    embeddings (B, N, D) and returns transformed embeddings (B, N, D).
    Latent Neural Hooks interact with this backbone at the layer level.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_dim)

        # Pre-compute RoPE frequencies (registered as buffer → moves with .to())
        rope = precompute_rope_freqs(
            config.hidden_dim // config.num_heads,
            config.max_seq_len,
        )
        self.register_buffer("rope_freqs", rope, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_offset: int = 0,
        attn_mask: torch.Tensor | None = None,
        hook_manager: "HookManager | None" = None,
        prefix_len: int | None = None,
        hook_policy: HookPolicy = HookPolicy.BIDIRECTIONAL_WITH_PREFIX,
    ) -> torch.Tensor:
        """Forward pass through all transformer layers.

        Args:
            x: (B, N, D) latent token sequence.
            rope_offset: optional position offset for RoPE.
            attn_mask: optional attention mask.
            hook_manager: optional HookManager that injects Latent Neural
                Hooks at designated layers.
            prefix_len: number of prefix tokens for mask expansion policy.
                Required when hook_manager is provided with a mask.
            hook_policy: how hook tokens interact with the attention mask.
        """
        content_len = x.shape[1]

        for layer_idx, layer in enumerate(self.layers):
            # --- Latent Neural Hook injection (before layer) ---
            if hook_manager is not None:
                x = hook_manager.pre_layer(layer_idx, x)

            # Expand attention mask for hook tokens using centralized logic
            layer_mask = attn_mask
            if layer_mask is not None and layer_mask.shape[-1] != x.shape[1]:
                n_hook = x.shape[1] - layer_mask.shape[-1]
                effective_prefix = prefix_len if prefix_len is not None else content_len
                layer_mask = expand_mask_for_hooks(
                    layer_mask,
                    n_hook_tokens=n_hook,
                    prefix_len=effective_prefix,
                    policy=hook_policy,
                )

            # --- Transformer layer (with optional gradient checkpointing) ---
            if self.config.gradient_checkpointing and self.training:
                x = cp.checkpoint(
                    layer,
                    x,
                    self.rope_freqs,
                    rope_offset,
                    layer_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, self.rope_freqs, rope_offset, layer_mask)

            # --- Latent Neural Hook extraction (after layer) ---
            if hook_manager is not None:
                x = hook_manager.post_layer(layer_idx, x)

        return self.final_norm(x)
