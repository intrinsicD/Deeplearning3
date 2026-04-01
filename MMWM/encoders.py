"""Multimodal encoders with pluggable per-modality sub-encoders."""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .containers import ObservationPacket
from .helpers import MLP, RMSNorm
from .interfaces import ENCODERS, IEncoder


class ModalitySubEncoder(nn.Module, abc.ABC):
    """Base class for per-modality sub-encoders."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a single modality tensor to [B, hidden_dim]."""
        raise NotImplementedError


class TextSubEncoder(ModalitySubEncoder):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = MLP([embed_dim, hidden_dim])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(x)  # [B, T, D]
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            pooled = (emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        else:
            pooled = emb.mean(dim=1)
        return self.proj(pooled)


class VectorSubEncoder(ModalitySubEncoder):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = MLP([input_dim, hidden_dim, hidden_dim])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.proj(x)


class ImageSubEncoder(ModalitySubEncoder):
    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = MLP([128, hidden_dim, hidden_dim])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.proj(self.conv(x).flatten(1))


@ENCODERS.register("simple_multimodal")
class SimpleMultimodalEncoder(IEncoder):
    """Flexible multimodal encoder that fuses an arbitrary set of modalities.

    Sub-encoders are registered by name. At forward time, only present modalities
    are encoded and fused via learned gating + summation, so adding or removing a
    modality requires no changes to this class.
    """

    def __init__(
        self,
        text_vocab_size: int = 32000,
        text_embed_dim: int = 256,
        vector_input_dim: int = 128,
        image_channels: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sub_encoders = nn.ModuleDict({
            "text": TextSubEncoder(text_vocab_size, text_embed_dim, hidden_dim),
            "vector": VectorSubEncoder(vector_input_dim, hidden_dim),
            "image": ImageSubEncoder(image_channels, hidden_dim),
        })
        self.modality_gates = nn.ParameterDict({
            name: nn.Parameter(torch.ones(hidden_dim)) for name in self.sub_encoders
        })
        self.fuse_norm = RMSNorm(hidden_dim)
        self.fuse_proj = MLP([hidden_dim, hidden_dim * 2, hidden_dim])

    def add_modality(self, name: str, sub_encoder: ModalitySubEncoder) -> None:
        """Register a new modality at runtime."""
        self.sub_encoders[name] = sub_encoder
        self.modality_gates[name] = nn.Parameter(
            torch.ones(self.hidden_dim, device=next(self.parameters()).device)
        )

    def forward(self, obs: ObservationPacket) -> Dict[str, torch.Tensor]:
        device = obs.device()
        batch_size = next(iter(obs.modalities.values())).shape[0]

        per_modality: Dict[str, torch.Tensor] = {}
        fused = torch.zeros(batch_size, self.hidden_dim, device=device)
        num_present = 0

        for name, sub_enc in self.sub_encoders.items():
            if name not in obs.modalities:
                continue
            feat = sub_enc(obs.modalities[name], obs.masks.get(name))
            gate = torch.sigmoid(self.modality_gates[name])
            per_modality[f"{name}_feat"] = feat
            fused = fused + gate * feat
            num_present += 1

        if num_present > 0:
            fused = fused / num_present
        fused = self.fuse_proj(self.fuse_norm(fused))
        per_modality["fused"] = fused
        return per_modality
