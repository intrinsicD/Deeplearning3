"""
Component 1: MoD Surprise Router + Forward World Model (FWM).

Mixture-of-Depths routing replaces hard frame dropping.
All patches pass through the network; only top-K% are routed to
expensive Transformer blocks. Static tensor shapes throughout.

FWM warm-up schedule: K_ratio = 1.0, anneal to 0.3 over 5K steps.

v0.4: CfC adoption for downstream components is gated on
MoD sparsity criterion (measured here).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FWM(nn.Module):
    """
    Forward World Model: 3-layer ConvNet predicting next-frame DINO features.

    Operates on the patch grid (e.g., 9x9 for 128px/14px patches).
    Surprise = L2 distance between predicted and actual next-frame features.
    """

    def __init__(self, d_features: int = 384, n_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Conv2d(d_features, d_features, 3, padding=1),
                nn.GroupNorm(8, d_features),
                nn.SiLU() if i < n_layers - 1 else nn.Identity(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict next-frame features from current features.

        Args:
            features: [B, D, H_grid, W_grid] current frame DINO features
                reshaped to spatial grid

        Returns:
            predicted: [B, D, H_grid, W_grid] predicted next-frame features
        """
        return self.net(features)


class HeavyBlock(nn.Module):
    """Small Transformer block for heavy-path patches."""

    def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.proj_out = nn.Linear(d_model, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, D_in] selected top-K patches
        Returns:
            [B, K, D_in] processed patches
        """
        h = self.proj_in(x)
        h = self.transformer(h)
        return self.proj_out(h)


class LightBlock(nn.Module):
    """Lightweight linear projection for non-routed patches."""

    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.SiLU(),
            nn.Linear(d_in, d_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MoDSurpriseRouter(nn.Module):
    """
    Mixture-of-Depths routing based on FWM surprise scores.

    1. FWM predicts next-frame features
    2. Surprise = prediction error (per-patch)
    3. Top-K% patches routed to heavy Transformer blocks
    4. Remaining patches through light projection
    5. Static tensor shapes maintained via gather/scatter
    """

    def __init__(
        self,
        d_features: int = 384,
        patch_grid: int = 9,
        n_heavy_layers: int = 2,
        d_heavy: int = 128,
        fwm_layers: int = 3,
        k_ratio_init: float = 1.0,
        k_ratio_final: float = 0.3,
        k_ratio_warmup_steps: int = 5000,
    ):
        super().__init__()

        self.d_features = d_features
        self.patch_grid = patch_grid
        self.n_patches = patch_grid * patch_grid
        self.k_ratio_init = k_ratio_init
        self.k_ratio_final = k_ratio_final
        self.k_ratio_warmup_steps = k_ratio_warmup_steps

        # Forward World Model
        self.fwm = FWM(d_features, n_layers=fwm_layers)

        # Heavy path: small Transformer
        self.heavy_block = HeavyBlock(
            d_in=d_features, d_model=d_heavy, n_layers=n_heavy_layers,
        )

        # Light path: linear projection
        self.light_block = LightBlock(d_features)

        # Track training step for K-ratio annealing
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    @property
    def current_k_ratio(self) -> float:
        """Current K ratio based on training step (annealing schedule)."""
        step = self._step.item()
        if step >= self.k_ratio_warmup_steps:
            return self.k_ratio_final
        progress = step / self.k_ratio_warmup_steps
        return self.k_ratio_init - progress * (self.k_ratio_init - self.k_ratio_final)

    @property
    def current_k(self) -> int:
        """Number of patches to route to heavy path."""
        return max(1, int(self.current_k_ratio * self.n_patches))

    def compute_surprise(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-patch surprise scores.

        Args:
            features_current: [B, N_patches, D] current frame features
            features_next: [B, N_patches, D] actual next-frame features

        Returns:
            surprise: [B, N_patches] per-patch surprise scores
            fwm_loss: scalar FWM prediction loss
        """
        B, N, D = features_current.shape
        H = W = self.patch_grid

        # Reshape to spatial grid for ConvNet
        current_grid = rearrange(
            features_current, "b (h w) d -> b d h w", h=H, w=W,
        )
        predicted_grid = self.fwm(current_grid)
        predicted = rearrange(predicted_grid, "b d h w -> b (h w) d")

        # Per-patch surprise: L2 distance
        surprise = (predicted - features_next.detach()).pow(2).mean(dim=-1)

        # FWM training loss
        fwm_loss = F.mse_loss(predicted, features_next.detach())

        return surprise, fwm_loss

    def route_patches(
        self,
        features: torch.Tensor,
        surprise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route patches based on surprise scores.

        Uses gather/scatter for static tensor shapes.

        Args:
            features: [B, N_patches, D]
            surprise: [B, N_patches]

        Returns:
            routed: [B, N_patches, D] features after routing
            routing_mask: [B, N_patches] binary mask (1 = heavy path)
        """
        B, N, D = features.shape
        K = self.current_k

        # Get top-K indices by surprise score
        _, topk_idx = surprise.topk(K, dim=-1)  # [B, K]

        # Build binary mask
        routing_mask = torch.zeros(B, N, device=features.device)
        routing_mask.scatter_(1, topk_idx, 1.0)

        # Light path: all patches
        light_out = self.light_block(features)  # [B, N, D]

        # Heavy path: selected patches only
        topk_features = features.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # [B, K, D]
        heavy_out = self.heavy_block(topk_features)  # [B, K, D]

        # Scatter heavy outputs back into light output
        routed = light_out.clone()
        routed.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, D), heavy_out)

        return routed, routing_mask

    def forward(
        self,
        features_current: torch.Tensor,
        features_next: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full MoD routing step.

        Args:
            features_current: [B, N_patches, D] current frame DINO features
            features_next: [B, N_patches, D] next frame features (for FWM training).
                If None, surprise is computed as feature variance (inference mode).

        Returns:
            dict with keys:
                "routed_features": [B, N_patches, D]
                "surprise": [B, N_patches]
                "routing_mask": [B, N_patches]
                "fwm_loss": scalar (0 if features_next is None)
        """
        if features_next is not None:
            surprise, fwm_loss = self.compute_surprise(
                features_current, features_next,
            )
        else:
            # Inference: use feature magnitude as proxy for surprise
            surprise = features_current.pow(2).mean(dim=-1)
            fwm_loss = torch.tensor(0.0, device=features_current.device)

        routed, routing_mask = self.route_patches(features_current, surprise)

        return {
            "routed_features": routed,
            "surprise": surprise,
            "routing_mask": routing_mask,
            "fwm_loss": fwm_loss,
        }

    def step(self):
        """Increment training step counter for K-ratio annealing."""
        self._step += 1

    def get_sparsity_metrics(
        self, routing_mask: torch.Tensor
    ) -> dict[str, float]:
        """Compute sparsity metrics for the MoD/CfC dependency gate.

        Args:
            routing_mask: [B, N_patches] or [B, T, N_patches]

        Returns:
            dict with sparsity measurements
        """
        heavy_ratio = routing_mask.float().mean().item()
        return {
            "heavy_ratio": heavy_ratio,
            "light_ratio": 1.0 - heavy_ratio,
            "k_ratio": self.current_k_ratio,
            "routing_entropy": self._compute_entropy(routing_mask),
        }

    @staticmethod
    def _compute_entropy(routing_mask: torch.Tensor) -> float:
        """Compute entropy of routing distribution over spatial patches."""
        # Average routing probability per spatial position
        if routing_mask.dim() == 3:
            probs = routing_mask.float().mean(dim=(0, 1))  # [N_patches]
        else:
            probs = routing_mask.float().mean(dim=0)  # [N_patches]

        probs = probs.clamp(min=1e-6, max=1 - 1e-6)
        entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log()).mean()
        return entropy.item()
