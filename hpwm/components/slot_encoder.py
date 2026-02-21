"""
Component 2: DINO-Bootstrapped Slot Encoder.

DINOv2 blocks 1-8: FROZEN (Phase -1: fully frozen ViT-S/14)
Slot attention groups patch features into object-level entity tokens.
Token Merging collapses redundant slots for efficiency.

Phase -1: N=8 slots, D_slot=128. Full spec: N=32, D_slot=768.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SlotAttention(nn.Module):
    """
    Iterative slot attention mechanism.

    Adapted from "Object-Centric Learning with Slot Attention" (Locatello et al.)
    Slots compete for patch features via softmax over the slot dimension.
    """

    def __init__(
        self,
        n_slots: int = 8,
        d_input: int = 384,
        d_slot: int = 128,
        n_iters: int = 3,
        mlp_hidden: int = 256,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.n_iters = n_iters
        self.eps = eps

        # Input projection
        self.norm_input = nn.LayerNorm(d_input)
        self.proj_k = nn.Linear(d_input, d_slot, bias=False)
        self.proj_v = nn.Linear(d_input, d_slot, bias=False)

        # Slot projections
        self.proj_q = nn.Linear(d_slot, d_slot, bias=False)
        self.norm_slots = nn.LayerNorm(d_slot)

        # Slot update GRU
        self.gru = nn.GRUCell(d_slot, d_slot)

        # Slot MLP refinement
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_slot),
            nn.Linear(d_slot, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, d_slot),
        )

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, 1, d_slot) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, d_slot))

        self.scale = d_slot ** -0.5

    def _init_slots(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        """Initialize slot embeddings with learned Gaussian."""
        mu = self.slot_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.n_slots, -1)
        if self.training:
            return mu + sigma * torch.randn_like(mu)
        return mu

    def forward(
        self, features: torch.Tensor, prev_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, N_patches, D_input] patch features from DINO + MoD
            prev_slots: [B, N_slots, D_slot] previous frame's slots for
                temporal continuity (optional)

        Returns:
            slots: [B, N_slots, D_slot] refined slot embeddings
            attn_weights: [B, N_slots, N_patches] assignment weights
        """
        B, N, _ = features.shape

        # Project inputs
        features = self.norm_input(features)
        k = self.proj_k(features)  # [B, N, D_slot]
        v = self.proj_v(features)  # [B, N, D_slot]

        # Initialize slots (from previous frame or learned prior)
        if prev_slots is not None:
            slots = prev_slots
        else:
            slots = self._init_slots(B, features.device, features.dtype)

        # Iterative attention
        attn_weights = None
        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.proj_q(slots)  # [B, N_slots, D_slot]

            # Attention: softmax over slots (competition)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(attn_logits, dim=1)  # [B, N_slots, N]

            # Weighted mean of values (normalize by attention sum)
            attn_norm = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.bmm(attn_norm, v)  # [B, N_slots, D_slot]

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.d_slot),
                slots_prev.reshape(-1, self.d_slot),
            ).reshape(B, self.n_slots, self.d_slot)

            # MLP refinement
            slots = slots + self.mlp(slots)

        return slots, attn_weights


class TokenMerger(nn.Module):
    """
    Token Merging: collapse redundant slots based on cosine similarity.

    When two slots are too similar (cosine sim > threshold), merge them
    and pad back to N_slots for static tensor shapes.
    """

    def __init__(self, threshold: float = 0.9):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def get_merge_map(self, slots: torch.Tensor) -> torch.Tensor:
        """Compute which slots should be merged.

        Args:
            slots: [B, N_slots, D_slot]

        Returns:
            merge_target: [B, N_slots] index of the target slot.
                merge_target[i] = i means slot i is kept.
                merge_target[i] = j (j != i) means slot i is merged into j.
        """
        B, N, D = slots.shape
        # Cosine similarity matrix
        slots_norm = F.normalize(slots, dim=-1)
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, N, N]

        # Zero diagonal (don't merge with self)
        sim = sim - torch.eye(N, device=sim.device).unsqueeze(0) * 2.0

        merge_target = torch.arange(N, device=slots.device).unsqueeze(0).expand(B, -1).clone()

        # Greedy merging: merge highest-similarity pairs
        for _ in range(N):
            max_sim, max_idx = sim.reshape(B, -1).max(dim=-1)
            row = max_idx // N
            col = max_idx % N

            for b in range(B):
                if max_sim[b] > self.threshold:
                    # Merge col into row (keep the earlier index)
                    src = max(row[b].item(), col[b].item())
                    tgt = min(row[b].item(), col[b].item())
                    merge_target[b, src] = tgt
                    # Disable this pair
                    sim[b, row[b], col[b]] = -2.0
                    sim[b, col[b], row[b]] = -2.0
                    # Disable merged slot
                    sim[b, src, :] = -2.0
                    sim[b, :, src] = -2.0

        return merge_target

    def forward(
        self, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: [B, N_slots, D_slot]

        Returns:
            merged: [B, N_slots, D_slot] (same shape, merged slots are averaged)
            merge_map: [B, N_slots] merge target indices
        """
        merge_map = self.get_merge_map(slots)
        B, N, D = slots.shape

        merged = torch.zeros_like(slots)
        counts = torch.zeros(B, N, 1, device=slots.device)

        for b in range(B):
            for i in range(N):
                tgt = merge_map[b, i].item()
                merged[b, tgt] += slots[b, i]
                counts[b, tgt] += 1

        merged = merged / counts.clamp(min=1)
        return merged, merge_map


class SlotEncoder(nn.Module):
    """
    Full slot encoder: DINO features -> slot attention -> token merging.

    Produces object-level entity tokens from patch-level features.
    Supports temporal continuity by accepting previous frame's slots.
    """

    def __init__(
        self,
        d_input: int = 384,
        n_slots: int = 8,
        d_slot: int = 128,
        n_iters: int = 3,
        mlp_hidden: int = 256,
        tome_threshold: float = 0.9,
    ):
        super().__init__()

        self.slot_attention = SlotAttention(
            n_slots=n_slots,
            d_input=d_input,
            d_slot=d_slot,
            n_iters=n_iters,
            mlp_hidden=mlp_hidden,
        )
        self.token_merger = TokenMerger(threshold=tome_threshold)

        self.n_slots = n_slots
        self.d_slot = d_slot

    def forward(
        self,
        features: torch.Tensor,
        prev_slots: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: [B, N_patches, D_input] routed patch features
            prev_slots: [B, N_slots, D_slot] previous frame's slots (optional)

        Returns:
            dict with:
                "slots": [B, N_slots, D_slot] entity tokens
                "attn_weights": [B, N_slots, N_patches] slot assignments
                "merge_map": [B, N_slots] merge targets
        """
        slots, attn_weights = self.slot_attention(features, prev_slots)
        merged_slots, merge_map = self.token_merger(slots)

        return {
            "slots": merged_slots,
            "attn_weights": attn_weights,
            "merge_map": merge_map,
            "raw_slots": slots,
        }
