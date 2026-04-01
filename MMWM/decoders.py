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
    position, enabling proper autoregressive generation with KV caching.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        latent_dim: int = 128,
        text_embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, text_embed_dim)
        self.token_proj = nn.Linear(text_embed_dim, hidden_dim) if text_embed_dim != hidden_dim else nn.Identity()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(2048, hidden_dim)
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

        # Latent becomes position 0; token embeddings fill positions 1..T
        latent_token = self.latent_proj(latent.z_sem).unsqueeze(1)  # [B, 1, D]
        token_emb = self.token_proj(self.token_embed(prefix_tokens))  # [B, T, D]
        seq = torch.cat([latent_token, token_emb], dim=1)  # [B, 1+T, D]
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
