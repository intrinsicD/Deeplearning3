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


class SlotAttentionBlock(nn.Module):
    def __init__(self, n_slots: int, d_input: int, d_slot: int, n_iters: int = 3) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.n_iters = n_iters
        self.norm_input = nn.LayerNorm(d_input)
        self.proj_k = nn.Linear(d_input, d_slot, bias=False)
        self.proj_v = nn.Linear(d_input, d_slot, bias=False)
        self.proj_q = nn.Linear(d_slot, d_slot, bias=False)
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_slot),
            nn.Linear(d_slot, d_slot * 2),
            nn.GELU(),
            nn.Linear(d_slot * 2, d_slot),
        )
        self.slot_mu = nn.Parameter(torch.randn(1, n_slots, d_slot) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, n_slots, d_slot))
        self.scale = d_slot ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, _, _ = tokens.shape
        x = self.norm_input(tokens)
        k = self.proj_k(x)
        v = self.proj_v(x)
        slots = self.slot_mu.expand(b, -1, -1)
        if self.training:
            slots = slots + self.slot_log_sigma.exp() * torch.randn_like(slots)
        for _ in range(self.n_iters):
            prev = slots
            q = self.proj_q(slots)
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=1)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            updates = torch.bmm(attn, v)
            slots = self.gru(updates.reshape(-1, self.d_slot), prev.reshape(-1, self.d_slot)).reshape(b, self.n_slots, self.d_slot)
            slots = slots + self.mlp(slots)
        return slots


class TokenMerger(nn.Module):
    def __init__(self, threshold: float = 0.9) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        sim = torch.bmm(torch.nn.functional.normalize(slots, dim=-1), torch.nn.functional.normalize(slots, dim=-1).transpose(1, 2))
        max_sim, partner = sim.topk(k=2, dim=-1)
        merge_mask = (max_sim[..., 1] > self.threshold).float().unsqueeze(-1)
        partner_slots = slots.gather(1, partner[..., 1].unsqueeze(-1).expand(-1, -1, slots.shape[-1]))
        return slots * (1.0 - merge_mask) + 0.5 * (slots + partner_slots) * merge_mask


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
        if not obs.modalities:
            raise ValueError(
                "SimpleMultimodalEncoder.forward received an ObservationPacket "
                "with no modalities; cannot infer batch size or produce features."
            )
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

        # Defensive: every modality in obs may still be unknown to this encoder
        # (caller passed e.g. {"audio": ...} but no audio sub-encoder is registered).
        if num_present == 0:
            known = sorted(self.sub_encoders.keys())
            got = sorted(obs.modalities.keys())
            raise ValueError(
                f"SimpleMultimodalEncoder received modalities {got} but none "
                f"match its registered sub-encoders {known}."
            )
        fused = fused / num_present
        fused = self.fuse_proj(self.fuse_norm(fused))
        per_modality["fused"] = fused
        return per_modality


class TextTransformerSubEncoder(ModalitySubEncoder):
    """Token-level causal transformer for structured text features."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 2, nhead: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(2048, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = RMSNorm(hidden_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = x.shape
        emb = self.embedding(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        emb = self.input_proj(emb + self.pos_embed(positions))
        causal_mask = self._causal_mask(T, x.device)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        h = self.transformer(emb, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            return (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return h.mean(dim=1)


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion replacing naive gated averaging.

    Each modality attends to all other modalities, producing a richer fused
    representation that captures inter-modal relationships.
    """

    def __init__(self, hidden_dim: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.mha = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = MLP([hidden_dim, hidden_dim * 2, hidden_dim])

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(modality_features) == 0:
            raise ValueError("CrossModalAttentionFusion requires at least one modality")
        if len(modality_features) == 1:
            return next(iter(modality_features.values()))
        tokens = torch.stack(list(modality_features.values()), dim=1)  # [B, M, D]
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.mean(dim=1)  # [B, D]


@ENCODERS.register("structured_multimodal")
class StructuredMultimodalEncoder(IEncoder):
    """Multimodal encoder with modality-specific structuring + cross-modal attention.

    Text gets a causal transformer, images get slot attention, vectors get MLP.
    Fusion uses cross-modal attention instead of naive gated averaging.
    """

    def __init__(
        self,
        text_vocab_size: int = 32000,
        text_embed_dim: int = 256,
        text_transformer_layers: int = 2,
        text_nhead: int = 4,
        vector_input_dim: int = 128,
        image_channels: int = 3,
        hidden_dim: int = 256,
        n_slots: int = 8,
        slot_dim: int = 128,
        slot_iters: int = 3,
        merge_threshold: float = 0.9,
        fusion_nhead: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_encoder = TextTransformerSubEncoder(
            text_vocab_size, text_embed_dim, hidden_dim,
            num_layers=text_transformer_layers, nhead=text_nhead,
        )
        self.vector_encoder = VectorSubEncoder(vector_input_dim, hidden_dim)
        self.image_encoder = ImageSubEncoder(image_channels, hidden_dim)

        self.image_tokenizer = nn.Sequential(
            nn.Conv2d(image_channels, slot_dim, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(slot_dim, slot_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.slot_attention = SlotAttentionBlock(n_slots=n_slots, d_input=slot_dim, d_slot=slot_dim, n_iters=slot_iters)
        self.token_merger = TokenMerger(threshold=merge_threshold)
        self.slot_proj = MLP([n_slots * slot_dim, hidden_dim, hidden_dim])

        self.cross_modal_fusion = CrossModalAttentionFusion(hidden_dim, nhead=fusion_nhead)
        self.fuse_norm = RMSNorm(hidden_dim)
        self.fuse_proj = MLP([hidden_dim, hidden_dim * 2, hidden_dim])

    def forward(self, obs: ObservationPacket) -> Dict[str, torch.Tensor]:
        if not obs.modalities:
            raise ValueError(
                "StructuredMultimodalEncoder.forward received an ObservationPacket "
                "with no modalities; cannot infer batch size or produce features."
            )
        device = obs.device()
        per_modality: Dict[str, torch.Tensor] = {}
        fusion_inputs: Dict[str, torch.Tensor] = {}

        if "text" in obs.modalities:
            feat = self.text_encoder(obs.modalities["text"], obs.masks.get("text"))
            per_modality["text_feat"] = feat
            fusion_inputs["text"] = feat

        if "vector" in obs.modalities:
            feat = self.vector_encoder(obs.modalities["vector"], obs.masks.get("vector"))
            per_modality["vector_feat"] = feat
            fusion_inputs["vector"] = feat

        if "image" in obs.modalities:
            image = obs.modalities["image"]
            cnn_feat = self.image_encoder(image)
            per_modality["image_feat"] = cnn_feat

            tokens = self.image_tokenizer(image).flatten(2).transpose(1, 2)
            slots = self.slot_attention(tokens)
            slots = self.token_merger(slots)
            slot_feat = self.slot_proj(slots.flatten(1))
            per_modality["image_slots"] = slot_feat
            fusion_inputs["image"] = 0.5 * cnn_feat + 0.5 * slot_feat

        if len(fusion_inputs) > 0:
            fused = self.cross_modal_fusion(fusion_inputs)
        else:
            known = ["text", "vector", "image"]
            got = sorted(obs.modalities.keys())
            raise ValueError(
                f"StructuredMultimodalEncoder received modalities {got} but none "
                f"match supported modalities {known}."
            )

        fused = self.fuse_proj(self.fuse_norm(fused))
        per_modality["fused"] = fused
        return per_modality


@ENCODERS.register("slot_multimodal")
class SlotMultimodalEncoder(SimpleMultimodalEncoder):
    """SimpleMultimodalEncoder + object-centric slot path for images."""

    def __init__(
        self,
        text_vocab_size: int = 32000,
        text_embed_dim: int = 256,
        vector_input_dim: int = 128,
        image_channels: int = 3,
        hidden_dim: int = 256,
        n_slots: int = 8,
        slot_dim: int = 128,
        slot_iters: int = 3,
        merge_threshold: float = 0.9,
    ) -> None:
        super().__init__(text_vocab_size, text_embed_dim, vector_input_dim, image_channels, hidden_dim)
        self.image_tokenizer = nn.Sequential(
            nn.Conv2d(image_channels, slot_dim, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(slot_dim, slot_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.slot_attention = SlotAttentionBlock(n_slots=n_slots, d_input=slot_dim, d_slot=slot_dim, n_iters=slot_iters)
        self.token_merger = TokenMerger(threshold=merge_threshold)
        self.slot_proj = MLP([n_slots * slot_dim, hidden_dim, hidden_dim])

    def forward(self, obs: ObservationPacket) -> Dict[str, torch.Tensor]:
        out = super().forward(obs)
        if "image" not in obs.modalities:
            return out
        image = obs.modalities["image"]
        tokens = self.image_tokenizer(image).flatten(2).transpose(1, 2)
        slots = self.slot_attention(tokens)
        slots = self.token_merger(slots)
        slot_feat = self.slot_proj(slots.flatten(1))
        out["image_slots"] = slot_feat
        out["fused"] = self.fuse_proj(self.fuse_norm(0.5 * out["fused"] + 0.5 * slot_feat))
        return out
