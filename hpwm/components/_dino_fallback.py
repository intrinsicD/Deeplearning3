"""
Fallback DINOv2-S/14 implementation for environments without internet access.

Creates a randomly-initialized ViT-S/14 with the same architecture as DINOv2-S/14.
Only for testing - features will be random without pretrained weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 14, d_model: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), "b d h w -> b (h w) d")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv2Fallback(nn.Module):
    """Minimal ViT-S/14 matching DINOv2-S architecture."""

    def __init__(self, d_model: int = 384, n_layers: int = 12, n_heads: int = 6):
        super().__init__()
        self.patch_embed = PatchEmbedding(14, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 82, d_model) * 0.02)  # 81 patches + 1 CLS
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.shape[0]
        patches = self.patch_embed(x)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)

        # Interpolate pos_embed if needed
        if x.shape[1] != self.pos_embed.shape[1]:
            x = x + F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
            ).transpose(1, 2)
        else:
            x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        return {"x_norm_patchtokens": x[:, 1:]}  # exclude CLS


def create_dino_fallback() -> DINOv2Fallback:
    """Create a fallback DINOv2-S/14 model."""
    print("[WARN] Using random DINOv2 fallback - no pretrained weights available.")
    print("       Download DINOv2-S/14 for meaningful features.")
    return DINOv2Fallback()
