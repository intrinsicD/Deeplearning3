"""
Suggested package layout for runnable lazy/binary tool loading:

latent_tool_ecosystem/
  __init__.py
  model.py                 # core world model and trainer
  tools/
    __init__.py
    memory_tools.py        # build_memory_read_tool / build_memory_write_tool
    transition_tools.py    # build_internal_transition_tool
    decoder_tools.py       # build_text_decoder_tool
  runtime/
    __init__.py
    registry.py            # loader, registry, promoter, judge
    router.py              # latent router / execution engine
  scripts/
    train_demo.py          # training entry point
    tool_demo.py           # tool ecosystem demo

If you keep this as a single file, update ToolDescriptor.entry_point to the actual
importable module path of the file. Example:
  "latent_tool_ecosystem.tools.memory_tools:build_memory_read_tool"
"""

from __future__ import annotations

import abc
import dataclasses
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# Multimodal Latent World Model (modular research skeleton)
# ------------------------------------------------------------
# Core idea:
#   observation -> encoder -> latent
#   latent + action + memory -> transition -> next latent
#   next latent -> decoders / heads / planner
#
# This file is intentionally modular so encoders, transition cores,
# regularizers, decoders, memory modules, and conditioning methods can be
# swapped without changing the top-level training loop.
#
# Browser monitoring:
#   Uses TensorBoard via torch.utils.tensorboard.SummaryWriter
#   Launch locally with:
#       tensorboard --logdir runs
#
# This implementation is a practical starting point, not a final SOTA model.
# ============================================================


# ============================================================
# Data containers
# ============================================================


@dataclass
class ObservationPacket:
    """Unified container for multimodal observations.

    Typical keys:
      modalities: {
         "text": LongTensor[B, T]
         "image": FloatTensor[B, C, H, W]
         "audio": FloatTensor[B, C, S]
         "vector": FloatTensor[B, D]
      }
      mask: optional masks per modality
      meta: arbitrary metadata
    """

    modalities: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def device(self) -> torch.device:
        for value in self.modalities.values():
            return value.device
        return torch.device("cpu")


@dataclass
class LatentState:
    """Role-based latent container.

    All tensors are expected to be shaped [B, D].
    """

    z_sem: torch.Tensor
    z_dyn: Optional[torch.Tensor] = None
    z_ctrl: Optional[torch.Tensor] = None
    z_mem: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def primary(self) -> torch.Tensor:
        parts = [self.z_sem]
        if self.z_dyn is not None:
            parts.append(self.z_dyn)
        if self.z_ctrl is not None:
            parts.append(self.z_ctrl)
        if self.z_mem is not None:
            parts.append(self.z_mem)
        return torch.cat(parts, dim=-1)

    def detach(self) -> "LatentState":
        return LatentState(
            z_sem=self.z_sem.detach(),
            z_dyn=None if self.z_dyn is None else self.z_dyn.detach(),
            z_ctrl=None if self.z_ctrl is None else self.z_ctrl.detach(),
            z_mem=None if self.z_mem is None else self.z_mem.detach(),
            extras={k: v.detach() for k, v in self.extras.items()},
        )


@dataclass
class MemoryState:
    context: Optional[torch.Tensor] = None
    hidden: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionOutput:
    next_latent: LatentState
    next_memory: MemoryState
    uncertainty: Optional[torch.Tensor] = None
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ModelOutput:
    current_latent: LatentState
    predicted_next_latent: LatentState
    target_next_latent: Optional[LatentState]
    next_memory: MemoryState
    decoder_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


# ============================================================
# Registry / factory
# ============================================================


class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        def decorator(obj: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            if name in self._items:
                raise ValueError(f"Duplicate registration: {name}")
            self._items[name] = obj
            return obj

        return decorator

    def build(self, name: str, **kwargs: Any) -> nn.Module:
        if name not in self._items:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._items)}")
        return self._items[name](**kwargs)

    def names(self) -> List[str]:
        return sorted(self._items.keys())


ENCODERS = Registry()
LATENT_PROJECTORS = Registry()
MEMORIES = Registry()
ACTION_ENCODERS = Registry()
CONDITIONERS = Registry()
TRANSITION_CORES = Registry()
PREDICTION_HEADS = Registry()
REGULARIZERS = Registry()
DECODERS = Registry()


# ============================================================
# Interfaces
# ============================================================


class IEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, obs: ObservationPacket) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class ILatentProjector(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, encoded: Dict[str, torch.Tensor]) -> LatentState:
        raise NotImplementedError


class IMemory(nn.Module, abc.ABC):
    @abc.abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, latent: LatentState, action_repr: torch.Tensor, state: MemoryState) -> MemoryState:
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, state: MemoryState) -> torch.Tensor:
        raise NotImplementedError


class IActionEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IConditioner(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, core_input: torch.Tensor, action_repr: torch.Tensor, memory_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class ITransitionCore(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, MemoryState, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class IPredictionHead(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, hidden: torch.Tensor, reference: LatentState) -> LatentState:
        raise NotImplementedError


class IRegularizer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, latent: LatentState) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class IDecoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


# ============================================================
# Helpers
# ============================================================


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0, activation: Callable[[], nn.Module] = nn.GELU) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class BlockAttentionResidual(nn.Module):
    """Practical block-level Attention Residuals (AttnRes-style) over depth.

    This approximates the 2026 Attention Residuals idea by replacing fixed
    additive residual accumulation across layers with learned attention over
    previous block states. It is most applicable to deep Transformer-like stacks.

    Design choice here:
      - within a block: standard residual updates
      - across blocks: learned attention over prior block summaries
    """

    def __init__(self, dim: int, attn_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim or dim
        self.q_proj = nn.Linear(dim, self.attn_dim)
        self.k_proj = nn.Linear(dim, self.attn_dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.attn_dim ** -0.5

    def forward(self, current: torch.Tensor, history: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(history) == 0:
            return current
        h = torch.stack(list(history), dim=1)
        q = self.q_proj(current).unsqueeze(1)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = scores.softmax(dim=-1)
        mixed = torch.matmul(weights, v).squeeze(1)
        return current + self.out_proj(mixed)


class LowRankHyperAdapter(nn.Module):
    """Generates lightweight per-step low-rank modulation instead of full weights.

    This is the practical "simulate more depth / specialization" path added on top
    of recurrent latent iteration. A hypernetwork generates rank-r factors from the
    current hidden state, and the factors induce a low-rank residual update.
    """

    def __init__(self, dim: int, rank: int = 8, hyper_hidden: int = 256) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.hyper = MLP([dim, hyper_hidden, hyper_hidden, 2 * dim * rank])
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.hyper(x)
        a, b = params.split(self.dim * self.rank, dim=-1)
        a = a.view(x.shape[0], self.dim, self.rank)
        b = b.view(x.shape[0], self.rank, self.dim)
        delta = torch.bmm(torch.bmm(x.unsqueeze(1), a), b).squeeze(1)
        return x + self.scale * delta


class AdaptiveHaltingHead(nn.Module):
    """Predicts whether recurrent latent iteration should stop."""

    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = MLP([dim, hidden_dim, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Encoders
# ============================================================


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
        # Learned per-modality importance gate
        self.modality_gates = nn.ParameterDict({
            name: nn.Parameter(torch.ones(hidden_dim)) for name in self.sub_encoders
        })
        self.fuse_norm = RMSNorm(hidden_dim)
        self.fuse_proj = MLP([hidden_dim, hidden_dim * 2, hidden_dim])

    def add_modality(self, name: str, sub_encoder: ModalitySubEncoder) -> None:
        """Register a new modality at runtime (no architecture change needed)."""
        self.sub_encoders[name] = sub_encoder
        self.modality_gates[name] = nn.Parameter(torch.ones(self.hidden_dim, device=next(self.parameters()).device))

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


# ============================================================
# Latent projectors
# ============================================================


@LATENT_PROJECTORS.register("role_split_mlp")
class RoleSplitLatentProjector(ILatentProjector):
    def __init__(self, input_dim: int = 256, latent_dim: int = 128, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.sem = nn.Linear(input_dim, latent_dim)
        self.dyn = nn.Linear(input_dim, latent_dim)
        self.ctrl = nn.Linear(input_dim, latent_dim)
        self.mem = nn.Linear(input_dim, latent_dim)
        self.bn = nn.BatchNorm1d(input_dim) if use_batchnorm else nn.Identity()

    def forward(self, encoded: Dict[str, torch.Tensor]) -> LatentState:
        fused = self.bn(encoded["fused"])
        return LatentState(
            z_sem=self.sem(fused),
            z_dyn=self.dyn(fused),
            z_ctrl=self.ctrl(fused),
            z_mem=self.mem(fused),
            extras={k: v for k, v in encoded.items()},
        )


# ============================================================
# Memory modules
# ============================================================


@MEMORIES.register("identity")
class IdentityMemory(IMemory):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        return MemoryState(context=torch.zeros(batch_size, self.latent_dim, device=device), hidden=None)

    def update(self, latent: LatentState, action_repr: torch.Tensor, state: MemoryState) -> MemoryState:
        return MemoryState(context=latent.z_mem if latent.z_mem is not None else latent.z_sem, hidden=state.hidden)

    def read(self, state: MemoryState) -> torch.Tensor:
        assert state.context is not None
        return state.context


@MEMORIES.register("gru")
class GRUMemory(IMemory):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        return MemoryState(context=hidden, hidden=hidden)

    def update(self, latent: LatentState, action_repr: torch.Tensor, state: MemoryState) -> MemoryState:
        mem_part = latent.z_mem if latent.z_mem is not None else latent.z_sem
        x = torch.cat([mem_part, action_repr], dim=-1)
        prev = state.hidden
        if prev is None:
            raise RuntimeError("GRUMemory requires state.hidden")
        hidden = self.cell(x, prev)
        return MemoryState(context=hidden, hidden=hidden)

    def read(self, state: MemoryState) -> torch.Tensor:
        assert state.context is not None
        return state.context


# ============================================================
# Action encoders
# ============================================================


@ACTION_ENCODERS.register("mlp")
class MLPActionEncoder(IActionEncoder):
    def __init__(self, action_dim: int = 32, action_embed_dim: int = 128) -> None:
        super().__init__()
        self.net = MLP([action_dim, action_embed_dim, action_embed_dim])

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


@ACTION_ENCODERS.register("embedding")
class DiscreteActionEncoder(IActionEncoder):
    def __init__(self, num_actions: int = 64, action_embed_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_actions, action_embed_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.dtype != torch.long:
            action = action.long()
        return self.embedding(action.squeeze(-1) if action.ndim > 1 and action.shape[-1] == 1 else action)


# ============================================================
# Conditioning adapters
# ============================================================


@CONDITIONERS.register("concat_mlp")
class ConcatConditioner(IConditioner):
    def __init__(self, latent_dim: int = 512, action_dim: int = 128, memory_dim: int = 128, out_dim: int = 512) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = MLP([latent_dim + action_dim + memory_dim, out_dim, out_dim])

    def forward(self, core_input: torch.Tensor, action_repr: torch.Tensor, memory_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        if memory_ctx is None:
            memory_ctx = torch.zeros(core_input.shape[0], 0, device=core_input.device, dtype=core_input.dtype)
        return self.net(torch.cat([core_input, action_repr, memory_ctx], dim=-1))


@CONDITIONERS.register("film")
class FiLMConditioner(IConditioner):
    def __init__(self, latent_dim: int = 512, action_dim: int = 128, memory_dim: int = 128) -> None:
        super().__init__()
        self.gamma = nn.Linear(action_dim + memory_dim, latent_dim)
        self.beta = nn.Linear(action_dim + memory_dim, latent_dim)

    def forward(self, core_input: torch.Tensor, action_repr: torch.Tensor, memory_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        if memory_ctx is None:
            memory_ctx = torch.zeros(core_input.shape[0], 0, device=core_input.device, dtype=core_input.dtype)
        cond = torch.cat([action_repr, memory_ctx], dim=-1)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return core_input * (1.0 + gamma) + beta


# ============================================================
# Transition cores
# ============================================================


@TRANSITION_CORES.register("mlp")
class MLPTransitionCore(ITransitionCore):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, depth: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * depth
        self.net = MLP(dims, dropout=dropout)

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, MemoryState, Dict[str, torch.Tensor]]:
        hidden = self.net(conditioned_input)
        return hidden, memory_state, {}


@TRANSITION_CORES.register("gru")
class GRUTransitionCore(ITransitionCore):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, MemoryState, Dict[str, torch.Tensor]]:
        if memory_state.hidden is None:
            hidden_prev = torch.zeros(conditioned_input.shape[0], self.hidden_dim, device=conditioned_input.device)
        else:
            hidden_prev = memory_state.hidden
        hidden = self.cell(conditioned_input, hidden_prev)
        next_memory = MemoryState(context=hidden, hidden=hidden, extras=memory_state.extras)
        return hidden, next_memory, {}


@TRANSITION_CORES.register("transformer")
class TransformerTransitionCore(ITransitionCore):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, num_layers: int = 4, nhead: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, MemoryState, Dict[str, torch.Tensor]]:
        x = self.input_proj(conditioned_input).unsqueeze(1)
        h = self.encoder(x).squeeze(1)
        h = self.norm(h)
        next_memory = MemoryState(context=h, hidden=memory_state.hidden, extras=memory_state.extras)
        return h, next_memory, {}


@TRANSITION_CORES.register("attnres_transformer")
class AttnResTransformerTransitionCore(ITransitionCore):
    """Transformer-like transition core with block Attention Residuals.

    Best fit in this codebase:
      - deep latent transition stacks
    Less applicable here:
      - GRU / MLP transition cores
      - current small non-transformer decoder head
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        block_size: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])
        self.block_residual = BlockAttentionResidual(hidden_dim)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, MemoryState, Dict[str, torch.Tensor]]:
        x = self.input_proj(conditioned_input).unsqueeze(1)
        block_history: List[torch.Tensor] = []
        block_states: List[torch.Tensor] = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            block_states.append(x.squeeze(1))
            is_block_end = ((idx + 1) % self.block_size == 0) or (idx == len(self.layers) - 1)
            if is_block_end:
                block_summary = torch.stack(block_states, dim=1).mean(dim=1)
                mixed = self.block_residual(block_summary, block_history)
                x = mixed.unsqueeze(1)
                block_history.append(mixed)
                block_states = []

        h = self.norm(x.squeeze(1))
        next_memory = MemoryState(context=h, hidden=memory_state.hidden, extras=memory_state.extras)
        return h, next_memory, {
            "attnres_num_blocks": torch.tensor(float(len(block_history)), device=h.device),
            "attnres_hidden_norm": h.norm(dim=-1).mean(),
        }


@TRANSITION_CORES.register("recurrent_attnres_transformer")
class RecurrentAttnResTransformerTransitionCore(ITransitionCore):
    """Simulated-depth latent transition core.

    Mechanisms:
      - shared recurrent latent depth (reuse the same deep block multiple times)
      - adaptive halting (variable compute per example)
      - hypernetwork-generated low-rank per-step modulation

    This is the most direct implementation of the user's "simulate depth" idea.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        block_size: int = 2,
        recurrent_steps: int = 4,
        halt_threshold: float = 0.5,
        adapter_rank: int = 8,
    ) -> None:
        super().__init__()
        self.core = AttnResTransformerTransitionCore(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
            block_size=block_size,
        )
        self.recurrent_steps = recurrent_steps
        self.halt_threshold = halt_threshold
        self.hyper_adapter = LowRankHyperAdapter(hidden_dim, rank=adapter_rank)
        self.halting = AdaptiveHaltingHead(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.silent_thought_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, MemoryState, Dict[str, torch.Tensor]]:
        current = self.input_proj(conditioned_input)
        halted_mask = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
        halting_steps = torch.zeros(current.shape[0], device=current.device)
        last_aux: Dict[str, torch.Tensor] = {}

        for step in range(self.recurrent_steps):
            core_hidden, _, aux = self.core(current, memory_state)
            core_hidden = self.hyper_adapter(core_hidden)
            core_hidden = current + self.silent_thought_scale * (core_hidden - current)

            halt_logit = self.halting(core_hidden).squeeze(-1)
            should_halt = torch.sigmoid(halt_logit) > self.halt_threshold
            newly_halted = should_halt & (~halted_mask)
            halting_steps = torch.where(newly_halted, torch.full_like(halting_steps, float(step + 1)), halting_steps)
            halted_mask = halted_mask | should_halt
            current = torch.where(halted_mask.unsqueeze(-1), current, core_hidden)
            last_aux = aux
            if bool(halted_mask.all()):
                break

        halting_steps = torch.where(halting_steps == 0, torch.full_like(halting_steps, float(self.recurrent_steps)), halting_steps)
        next_memory = MemoryState(context=current, hidden=memory_state.hidden, extras=memory_state.extras)
        result_aux = dict(last_aux)
        result_aux.update({
            "recurrent_steps_mean": halting_steps.mean(),
            "recurrent_steps_max": halting_steps.max(),
            "recurrent_hidden_norm": current.norm(dim=-1).mean(),
        })
        return current, next_memory, result_aux


# ============================================================
# Prediction heads
# ============================================================


@PREDICTION_HEADS.register("role_split")
class RoleSplitPredictionHead(IPredictionHead):
    def __init__(self, hidden_dim: int = 512, latent_dim: int = 128) -> None:
        super().__init__()
        self.sem = nn.Linear(hidden_dim, latent_dim)
        self.dyn = nn.Linear(hidden_dim, latent_dim)
        self.ctrl = nn.Linear(hidden_dim, latent_dim)
        self.mem = nn.Linear(hidden_dim, latent_dim)
        self.uncertainty = nn.Linear(hidden_dim, latent_dim)

    def forward(self, hidden: torch.Tensor, reference: LatentState) -> LatentState:
        extras = dict(reference.extras)
        extras["predicted_logvar"] = self.uncertainty(hidden)
        return LatentState(
            z_sem=self.sem(hidden),
            z_dyn=self.dyn(hidden) if reference.z_dyn is not None else None,
            z_ctrl=self.ctrl(hidden) if reference.z_ctrl is not None else None,
            z_mem=self.mem(hidden) if reference.z_mem is not None else None,
            extras=extras,
        )


# ============================================================
# Regularizers
# ============================================================


@REGULARIZERS.register("none")
class NoRegularizer(IRegularizer):
    def forward(self, latent: LatentState) -> Dict[str, torch.Tensor]:
        device = latent.z_sem.device
        return {"regularizer_total": torch.zeros((), device=device)}


@REGULARIZERS.register("sigreg_like")
class SIGRegLike(IRegularizer):
    """A lightweight anti-collapse regularizer inspired by the paper.

    This is not a verbatim reproduction of SIGReg. It is a practical, modular
    placeholder that encourages variance and discourages covariance collapse.
    """

    def __init__(self, variance_weight: float = 1.0, covariance_weight: float = 0.04, target_std: float = 1.0) -> None:
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.target_std = target_std

    @staticmethod
    def _cov_offdiag(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / max(x.shape[0] - 1, 1)
        offdiag = cov - torch.diag(torch.diag(cov))
        return offdiag.pow(2).mean()

    def _apply(self, z: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = F.relu(self.target_std - std).mean()
        covariance_loss = self._cov_offdiag(z)
        total = self.variance_weight * variance_loss + self.covariance_weight * covariance_loss
        return {
            f"{prefix}_variance_loss": variance_loss,
            f"{prefix}_covariance_loss": covariance_loss,
            f"{prefix}_reg_total": total,
        }

    def forward(self, latent: LatentState) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=latent.z_sem.device)
        for name, tensor in [("z_sem", latent.z_sem), ("z_dyn", latent.z_dyn), ("z_ctrl", latent.z_ctrl), ("z_mem", latent.z_mem)]:
            if tensor is None:
                continue
            part = self._apply(tensor, name)
            losses.update(part)
            total = total + part[f"{name}_reg_total"]
        losses["regularizer_total"] = total
        return losses


# ============================================================
# Decoders
# ============================================================


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
        # KV cache for generation
        self._cache: Optional[torch.Tensor] = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def clear_cache(self) -> None:
        self._cache = None

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
        # (position 0 is the latent conditioning token)
        logits = self.lm_head(h[:, 1:, :])  # [B, T, vocab]
        return {"text_logits": logits}


@DECODERS.register("vector_reconstruction")
class VectorReconstructionHead(IDecoder):
    def __init__(self, latent_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.head = MLP([latent_dim, latent_dim * 2, output_dim])

    def forward(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        return {"vector_recon": self.head(latent.z_sem)}


# ============================================================
# Top-level model
# ============================================================


class ModularLatentWorldModel(nn.Module):
    def __init__(
        self,
        encoder: IEncoder,
        latent_projector: ILatentProjector,
        memory: IMemory,
        action_encoder: IActionEncoder,
        conditioner: IConditioner,
        transition_core: ITransitionCore,
        prediction_head: IPredictionHead,
        regularizer: IRegularizer,
        decoders: Optional[Dict[str, IDecoder]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_projector = latent_projector
        self.memory = memory
        self.action_encoder = action_encoder
        self.conditioner = conditioner
        self.transition_core = transition_core
        self.prediction_head = prediction_head
        self.regularizer = regularizer
        self.decoders = nn.ModuleDict(decoders or {})

    def encode(self, obs: ObservationPacket) -> LatentState:
        encoded = self.encoder(obs)
        latent = self.latent_projector(encoded)
        return latent

    def transition(self, latent: LatentState, action: torch.Tensor, memory_state: MemoryState) -> TransitionOutput:
        action_repr = self.action_encoder(action)
        memory_ctx = self.memory.read(memory_state)
        core_input = latent.primary()
        conditioned = self.conditioner(core_input, action_repr, memory_ctx)
        hidden, next_memory_from_core, aux = self.transition_core(conditioned, memory_state)
        predicted_next = self.prediction_head(hidden, reference=latent)
        updated_memory = self.memory.update(predicted_next, action_repr, next_memory_from_core)
        uncertainty = predicted_next.extras.get("predicted_logvar")
        aux["action_norm"] = action_repr.norm(dim=-1).mean()
        return TransitionOutput(next_latent=predicted_next, next_memory=updated_memory, uncertainty=uncertainty, aux=aux)

    def decode(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name, decoder in self.decoders.items():
            dec_out = decoder(latent, context=context)
            for k, v in dec_out.items():
                outputs[f"{name}.{k}"] = v
        return outputs

    def forward(
        self,
        obs_t: ObservationPacket,
        action_t: torch.Tensor,
        obs_tp1: Optional[ObservationPacket] = None,
        memory_state: Optional[MemoryState] = None,
        decoder_context: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        current_latent = self.encode(obs_t)
        batch_size = current_latent.z_sem.shape[0]
        if memory_state is None:
            memory_state = self.memory.init_state(batch_size=batch_size, device=current_latent.z_sem.device)

        transition = self.transition(current_latent, action_t, memory_state)
        target_next_latent = self.encode(obs_tp1) if obs_tp1 is not None else None
        decoder_outputs = self.decode(transition.next_latent, context=decoder_context)

        aux = dict(transition.aux)
        aux.update(self.regularizer(current_latent))
        aux["latent_norm"] = current_latent.z_sem.norm(dim=-1).mean()
        return ModelOutput(
            current_latent=current_latent,
            predicted_next_latent=transition.next_latent,
            target_next_latent=target_next_latent,
            next_memory=transition.next_memory,
            decoder_outputs=decoder_outputs,
            aux=aux,
        )


# ============================================================
# Losses
# ============================================================


@dataclass
class LossWeights:
    latent_sem: float = 1.0
    latent_dyn: float = 1.0
    latent_ctrl: float = 0.25
    latent_mem: float = 0.25
    regularizer: float = 1.0
    text_ce: float = 1.0
    vector_recon: float = 0.0


class WorldModelLoss(nn.Module):
    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    @staticmethod
    def _mse(pred: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> torch.Tensor:
        if pred is None or target is None:
            device = pred.device if pred is not None else target.device if target is not None else torch.device("cpu")
            return torch.zeros((), device=device)
        return F.mse_loss(pred, target)

    def forward(self, output: ModelOutput, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if output.target_next_latent is None:
            raise ValueError("WorldModelLoss requires target_next_latent. Provide obs_tp1 during forward().")

        losses: Dict[str, torch.Tensor] = {}
        losses["latent_sem_loss"] = self._mse(output.predicted_next_latent.z_sem, output.target_next_latent.z_sem)
        losses["latent_dyn_loss"] = self._mse(output.predicted_next_latent.z_dyn, output.target_next_latent.z_dyn)
        losses["latent_ctrl_loss"] = self._mse(output.predicted_next_latent.z_ctrl, output.target_next_latent.z_ctrl)
        losses["latent_mem_loss"] = self._mse(output.predicted_next_latent.z_mem, output.target_next_latent.z_mem)

        losses["regularizer_loss"] = output.aux["regularizer_total"]

        total = (
            self.weights.latent_sem * losses["latent_sem_loss"]
            + self.weights.latent_dyn * losses["latent_dyn_loss"]
            + self.weights.latent_ctrl * losses["latent_ctrl_loss"]
            + self.weights.latent_mem * losses["latent_mem_loss"]
            + self.weights.regularizer * losses["regularizer_loss"]
        )

        if "text_target" in batch and any(key.endswith("text_logits") for key in output.decoder_outputs):
            text_logits = next(v for k, v in output.decoder_outputs.items() if k.endswith("text_logits"))
            text_target = batch["text_target"]
            # text_logits is [B, T, V] from causal decoder; text_target is [B, T]
            if text_logits.ndim == 3:
                losses["text_ce_loss"] = F.cross_entropy(
                    text_logits.reshape(-1, text_logits.size(-1)), text_target.reshape(-1)
                )
            else:
                losses["text_ce_loss"] = F.cross_entropy(text_logits, text_target)
            total = total + self.weights.text_ce * losses["text_ce_loss"]
        else:
            device = total.device
            losses["text_ce_loss"] = torch.zeros((), device=device)

        if "vector_target" in batch and any(key.endswith("vector_recon") for key in output.decoder_outputs):
            vector_pred = next(v for k, v in output.decoder_outputs.items() if k.endswith("vector_recon"))
            vector_target = batch["vector_target"]
            losses["vector_recon_loss"] = F.mse_loss(vector_pred, vector_target)
            total = total + self.weights.vector_recon * losses["vector_recon_loss"]
        else:
            device = total.device
            losses["vector_recon_loss"] = torch.zeros((), device=device)

        losses["total_loss"] = total
        return losses


# ============================================================
# TensorBoard hooks and monitoring
# ============================================================


class HookManager:
    """Central place for train/inference observability.

    Everything here is browser-visible through TensorBoard.
    """

    def __init__(self, writer: SummaryWriter, histogram_every: int = 200, embedding_every: int = 500) -> None:
        self.writer = writer
        self.histogram_every = histogram_every
        self.embedding_every = embedding_every

    def log_losses(self, losses: Mapping[str, torch.Tensor], step: int, split: str) -> None:
        for name, value in losses.items():
            self.writer.add_scalar(f"{split}/loss/{name}", float(value.detach().cpu().item()), step)

    def log_aux(self, aux: Mapping[str, torch.Tensor], step: int, split: str) -> None:
        for name, value in aux.items():
            if value.ndim == 0:
                self.writer.add_scalar(f"{split}/aux/{name}", float(value.detach().cpu().item()), step)
            else:
                self.writer.add_scalar(f"{split}/aux/{name}_mean", float(value.detach().mean().cpu().item()), step)

    def log_latents(self, output: ModelOutput, step: int, split: str) -> None:
        current = output.current_latent
        pred = output.predicted_next_latent
        for tag, latent in [("current", current), ("predicted", pred)]:
            self.writer.add_scalar(f"{split}/latent/{tag}_z_sem_norm", float(latent.z_sem.norm(dim=-1).mean().detach().cpu().item()), step)
            self.writer.add_scalar(f"{split}/latent/{tag}_z_sem_std", float(latent.z_sem.std(dim=0).mean().detach().cpu().item()), step)
            if latent.z_dyn is not None:
                self.writer.add_scalar(f"{split}/latent/{tag}_z_dyn_norm", float(latent.z_dyn.norm(dim=-1).mean().detach().cpu().item()), step)
            if latent.z_ctrl is not None:
                self.writer.add_scalar(f"{split}/latent/{tag}_z_ctrl_norm", float(latent.z_ctrl.norm(dim=-1).mean().detach().cpu().item()), step)
            if latent.z_mem is not None:
                self.writer.add_scalar(f"{split}/latent/{tag}_z_mem_norm", float(latent.z_mem.norm(dim=-1).mean().detach().cpu().item()), step)

        if step % self.histogram_every == 0:
            self.writer.add_histogram(f"{split}/hist/current_z_sem", current.z_sem.detach().cpu(), step)
            self.writer.add_histogram(f"{split}/hist/predicted_z_sem", pred.z_sem.detach().cpu(), step)

    def log_embeddings(self, output: ModelOutput, step: int, split: str, metadata: Optional[List[str]] = None) -> None:
        if step % self.embedding_every != 0:
            return
        z = output.current_latent.z_sem.detach().cpu()
        max_points = min(256, z.shape[0])
        self.writer.add_embedding(z[:max_points], metadata=metadata[:max_points] if metadata else None, tag=f"{split}/embeddings/z_sem", global_step=step)

    def log_text_predictions(self, output: ModelOutput, batch: Mapping[str, Any], step: int, split: str, id_to_token: Optional[Callable[[int], str]] = None) -> None:
        text_key = next((k for k in output.decoder_outputs if k.endswith("text_logits")), None)
        if text_key is None or "text_target" not in batch:
            return
        logits = output.decoder_outputs[text_key].detach()
        pred_ids = logits.argmax(dim=-1)  # [B, T] or [B]
        target_ids = batch["text_target"].detach()

        def tok(i: int) -> str:
            return id_to_token(i) if id_to_token is not None else str(i)

        preview = []
        # Handle both [B, T] sequence logits and legacy [B] single-token logits
        if pred_ids.ndim == 2:
            for p_seq, t_seq in zip(pred_ids[:4].tolist(), target_ids[:4].tolist()):
                pred_str = " ".join(tok(int(x)) for x in p_seq[:8])
                tgt_str = " ".join(tok(int(x)) for x in (t_seq[:8] if isinstance(t_seq, list) else [t_seq]))
                preview.append(f"pred=[{pred_str}] | target=[{tgt_str}]")
        else:
            for p, t in zip(pred_ids[:8].tolist(), target_ids[:8].tolist()):
                preview.append(f"pred={tok(int(p))} | target={tok(int(t))}")
        text = "\n".join(preview)
        self.writer.add_text(f"{split}/text_predictions", text, step)

    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        for idx, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"train/lr/group_{idx}", group["lr"], step)

    def log_gradient_norms(self, model: nn.Module, step: int) -> None:
        total_sq = 0.0
        count = 0
        for param in model.parameters():
            if param.grad is None:
                continue
            g = param.grad.detach().float()
            total_sq += float(g.norm().item() ** 2)
            count += 1
        total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
        self.writer.add_scalar("train/grad/total_norm", total_norm, step)
        self.writer.add_scalar("train/grad/num_tensors", count, step)

    def log_inference_trace(self, predicted_tokens: Sequence[int], step: int, split: str, id_to_token: Optional[Callable[[int], str]] = None) -> None:
        if id_to_token is None:
            rendered = " ".join(str(int(x)) for x in predicted_tokens)
        else:
            rendered = " ".join(id_to_token(int(x)) for x in predicted_tokens)
        self.writer.add_text(f"{split}/inference/generated", rendered, step)

    def log_depth_metrics(self, aux: Mapping[str, torch.Tensor], step: int, split: str) -> None:
        for key in ["recurrent_steps_mean", "recurrent_steps_max", "recurrent_hidden_norm", "attnres_num_blocks", "attnres_hidden_norm"]:
            if key in aux:
                value = aux[key]
                scalar = float(value.detach().mean().cpu().item()) if value.ndim > 0 else float(value.detach().cpu().item())
                self.writer.add_scalar(f"{split}/depth/{key}", scalar, step)


# ============================================================
# Trainer
# ============================================================


class Trainer:
    def __init__(
        self,
        model: ModularLatentWorldModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: WorldModelLoss,
        device: torch.device,
        run_dir: str = "runs/modular_world_model",
        grad_clip_norm: Optional[float] = 1.0,
        mixed_precision: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")
        self.writer = SummaryWriter(log_dir=run_dir)
        self.hooks = HookManager(self.writer)
        self.global_step = 0

    def _to_packet(self, batch: Mapping[str, Any], suffix: str) -> ObservationPacket:
        modalities: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for name in ["text", "vector", "image"]:
            key = f"{name}_{suffix}"
            if key in batch:
                modalities[name] = batch[key].to(self.device)
            mask_key = f"{name}_mask_{suffix}"
            if mask_key in batch:
                masks[name] = batch[mask_key].to(self.device)
        return ObservationPacket(modalities=modalities, masks=masks, meta={})

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self.model.train()
        obs_t = self._to_packet(batch, "t")
        obs_tp1 = self._to_packet(batch, "tp1")
        action = batch["action"].to(self.device)

        decoder_context: Dict[str, Any] = {}
        if "prefix_tokens" in batch:
            decoder_context["prefix_tokens"] = batch["prefix_tokens"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            output = self.model(obs_t, action, obs_tp1=obs_tp1, memory_state=None, decoder_context=decoder_context)
            loss_inputs: Dict[str, Any] = {}
            if "text_target" in batch:
                loss_inputs["text_target"] = batch["text_target"].to(self.device)
            if "vector_target" in batch:
                loss_inputs["vector_target"] = batch["vector_target"].to(self.device)
            losses = self.loss_fn(output, loss_inputs)
            total_loss = losses["total_loss"]

        self.scaler.scale(total_loss).backward()
        if self.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.hooks.log_losses(losses, self.global_step, split="train")
        self.hooks.log_aux(output.aux, self.global_step, split="train")
        self.hooks.log_depth_metrics(output.aux, self.global_step, split="train")
        self.hooks.log_latents(output, self.global_step, split="train")
        self.hooks.log_gradient_norms(self.model, self.global_step)
        self.hooks.log_learning_rate(self.optimizer, self.global_step)
        if "text_target" in batch:
            self.hooks.log_text_predictions(output, {"text_target": batch["text_target"].to(self.device)}, self.global_step, split="train")
        self.hooks.log_embeddings(output, self.global_step, split="train")

        result = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
        self.global_step += 1
        return result

    @torch.no_grad()
    def eval_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        self.model.eval()
        obs_t = self._to_packet(batch, "t")
        obs_tp1 = self._to_packet(batch, "tp1")
        action = batch["action"].to(self.device)

        decoder_context: Dict[str, Any] = {}
        if "prefix_tokens" in batch:
            decoder_context["prefix_tokens"] = batch["prefix_tokens"].to(self.device)

        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            output = self.model(obs_t, action, obs_tp1=obs_tp1, memory_state=None, decoder_context=decoder_context)
            loss_inputs: Dict[str, Any] = {}
            if "text_target" in batch:
                loss_inputs["text_target"] = batch["text_target"].to(self.device)
            if "vector_target" in batch:
                loss_inputs["vector_target"] = batch["vector_target"].to(self.device)
            losses = self.loss_fn(output, loss_inputs)

        self.hooks.log_losses(losses, self.global_step, split="eval")
        self.hooks.log_aux(output.aux, self.global_step, split="eval")
        self.hooks.log_depth_metrics(output.aux, self.global_step, split="eval")
        self.hooks.log_latents(output, self.global_step, split="eval")
        if "text_target" in batch:
            self.hooks.log_text_predictions(output, {"text_target": batch["text_target"].to(self.device)}, self.global_step, split="eval")
        self.hooks.log_embeddings(output, self.global_step, split="eval")

        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    def fit(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None, epochs: int = 1, eval_every_steps: Optional[int] = None) -> None:
        for epoch in range(epochs):
            for batch in train_loader:
                train_metrics = self.train_step(batch)
                if eval_loader is not None and eval_every_steps is not None and self.global_step % eval_every_steps == 0:
                    eval_batch = next(iter(eval_loader))
                    self.eval_step(eval_batch)
            self.writer.add_text("train/epoch", f"Finished epoch {epoch}", self.global_step)

    @torch.no_grad()
    def generate_next_tokens(
        self,
        obs_t: ObservationPacket,
        action_t: torch.Tensor,
        prefix_tokens: torch.Tensor,
        steps: int = 16,
    ) -> List[int]:
        """Autoregressively generate tokens from the causal text decoder.

        The world model transition is computed once to obtain the predicted
        next-latent, then the text decoder is called step-by-step, appending
        each sampled token to the growing prefix.
        """
        self.model.eval()
        batch_size = prefix_tokens.shape[0]
        if batch_size != 1:
            raise ValueError("generate_next_tokens currently expects batch size 1 for simplicity.")

        obs_t = ObservationPacket(
            modalities={k: v.to(self.device) for k, v in obs_t.modalities.items()},
            masks={k: v.to(self.device) for k, v in obs_t.masks.items()},
            meta=dict(obs_t.meta),
        )
        action_t = action_t.to(self.device)
        prefix_tokens = prefix_tokens.to(self.device)

        latent = self.model.encode(obs_t)
        memory = self.model.memory.init_state(batch_size=1, device=self.device)
        transition = self.model.transition(latent, action_t, memory)
        pred_latent = transition.next_latent
        generated: List[int] = []

        for _ in range(steps):
            outputs = self.model.decode(pred_latent, context={"prefix_tokens": prefix_tokens})
            logits = next(v for k, v in outputs.items() if k.endswith("text_logits"))
            # logits is [B, T, V]; take last position for next-token prediction
            next_token = int(logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], device=self.device, dtype=prefix_tokens.dtype)
            prefix_tokens = torch.cat([prefix_tokens, next_token_tensor], dim=1)

        self.hooks.log_inference_trace(generated, self.global_step, split="inference")
        return generated


# ============================================================
# Notes on Attention Residuals applicability
# ------------------------------------------------------------
# The 2026 Attention Residuals paper replaces fixed additive residual
# accumulation across depth with learned attention over prior layer/block states.
# In this codebase, the best fit is the deep Transformer-style latent transition
# core, because that is where we currently have a residual stack over depth.
# It is much less applicable to:
#   - GRU transition cores
#   - plain MLP tools
#   - the current tiny text decoder head
# If you later add a full transformer decoder or a deeper transformer encoder,
# the same BlockAttentionResidual module can be reused there as well.
# ============================================================
# Tool ecosystem: binary / lazy-loaded neural tools
# ============================================================


@dataclass
class LatentPacket:
    """Stable protocol passed between tools.

    Tools should read/write only through this packet so they remain swappable.
    """

    state: LatentState
    confidence: Optional[torch.Tensor] = None
    source_tool: Optional[str] = None
    timestamp: Optional[int] = None
    trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    memory_state: Optional[MemoryState] = None
    observation: Optional[ObservationPacket] = None
    action: Optional[torch.Tensor] = None
    raw_inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    packet: Optional[LatentPacket]
    raw_output: Optional[Any] = None
    confidence: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    side_effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDescriptor:
    tool_id: str
    kind: str
    version: str
    path: Optional[str] = None
    entry_point: Optional[str] = None
    input_schema: str = "LatentPacket"
    output_schema: str = "ToolResult"
    reads_slots: Tuple[str, ...] = tuple()
    writes_slots: Tuple[str, ...] = tuple()
    benchmark_id: Optional[str] = None
    estimated_cost: float = 1.0
    estimated_latency_ms: float = 1.0
    memory_mb: float = 0.0
    tags: Tuple[str, ...] = tuple()
    enabled: bool = True
    binary: bool = True
    lazy_load: bool = True
    unload_after_call: bool = False
    device_preference: str = "auto"
    config: Dict[str, Any] = field(default_factory=dict)


class BaseTool(nn.Module, abc.ABC):
    descriptor: ToolDescriptor

    def __init__(self, descriptor: ToolDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor

    @abc.abstractmethod
    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        raise NotImplementedError


class BinaryToolLoader:
    """Loads neural tools only when needed.

    Expected packaging:
      - a Python module path in descriptor.entry_point, or
      - a serialized torch checkpoint in descriptor.path plus a known builder.

    For the first implementation, we support entry_point strings formatted as:
      "module.submodule:build_tool"
    where build_tool(descriptor) -> BaseTool
    """

    def __init__(self, device: torch.device, keep_hot: int = 2) -> None:
        self.device = device
        self.keep_hot = keep_hot
        self._loaded: Dict[str, BaseTool] = {}
        self._lru: Deque[str] = deque()

    def _touch(self, tool_id: str) -> None:
        try:
            self._lru.remove(tool_id)
        except ValueError:
            pass
        self._lru.append(tool_id)
        while len(self._lru) > self.keep_hot:
            old = self._lru.popleft()
            self.unload(old)

    def load(self, descriptor: ToolDescriptor) -> BaseTool:
        if descriptor.tool_id in self._loaded:
            self._touch(descriptor.tool_id)
            return self._loaded[descriptor.tool_id]

        if descriptor.entry_point is None:
            raise ValueError(f"Tool {descriptor.tool_id} has no entry_point")

        module_name, func_name = descriptor.entry_point.split(":", maxsplit=1)
        mod = __import__(module_name, fromlist=[func_name])
        builder = getattr(mod, func_name)
        tool = builder(descriptor)
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Builder for {descriptor.tool_id} did not return BaseTool")
        tool.to(self.device)
        tool.eval()
        self._loaded[descriptor.tool_id] = tool
        self._touch(descriptor.tool_id)
        return tool

    def unload(self, tool_id: str) -> None:
        tool = self._loaded.pop(tool_id, None)
        if tool is None:
            return
        del tool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            self._lru.remove(tool_id)
        except ValueError:
            pass

    def call(self, descriptor: ToolDescriptor, packet: LatentPacket, context: ToolContext) -> ToolResult:
        tool = self.load(descriptor)
        with torch.no_grad():
            result = tool.run(packet, context)
        if descriptor.unload_after_call:
            self.unload(descriptor.tool_id)
        return result


class ToolRegistrySystem:
    def __init__(self, loader: BinaryToolLoader) -> None:
        self.loader = loader
        self._descriptors: Dict[str, ToolDescriptor] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, descriptor: ToolDescriptor, alias: Optional[str] = None) -> None:
        self._descriptors[descriptor.tool_id] = descriptor
        if alias is not None:
            self._aliases[alias] = descriptor.tool_id

    def get(self, tool_id_or_alias: str) -> ToolDescriptor:
        tool_id = self._aliases.get(tool_id_or_alias, tool_id_or_alias)
        return self._descriptors[tool_id]

    def list_by_kind(self, kind: str) -> List[ToolDescriptor]:
        return [d for d in self._descriptors.values() if d.kind == kind and d.enabled]

    def call(self, tool_id_or_alias: str, packet: LatentPacket, context: ToolContext) -> ToolResult:
        descriptor = self.get(tool_id_or_alias)
        return self.loader.call(descriptor, packet, context)

    def replace(self, old_tool_id: str, new_descriptor: ToolDescriptor) -> None:
        aliases_to_move = [alias for alias, target in self._aliases.items() if target == old_tool_id]
        self._descriptors[new_descriptor.tool_id] = new_descriptor
        for alias in aliases_to_move:
            self._aliases[alias] = new_descriptor.tool_id


class CriticalExperienceBuffer:
    def __init__(self) -> None:
        self.examples: List[Dict[str, Any]] = []

    def add(self, example: Dict[str, Any]) -> None:
        self.examples.append(example)

    def sample(self, benchmark_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = [ex for ex in self.examples if benchmark_id is None or benchmark_id in ex.get("benchmarks", [])]
        return items if limit is None else items[:limit]


class ToolJudge:
    def __init__(self, critical_buffer: CriticalExperienceBuffer) -> None:
        self.critical_buffer = critical_buffer

    def score_candidate(
        self,
        candidate: ToolDescriptor,
        incumbent: Optional[ToolDescriptor],
        local_metrics: Dict[str, float],
        regression_metrics: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        score = 0.0
        score += local_metrics.get("quality", 0.0)
        score -= 0.1 * local_metrics.get("latency_ms", candidate.estimated_latency_ms)
        score -= 0.01 * local_metrics.get("memory_mb", candidate.memory_mb)
        regressions = regression_metrics.get("critical_failures", 0.0)
        score -= 100.0 * regressions
        accepted = regressions <= 0.0 and score > 0.0
        return accepted, {
            "score": score,
            "critical_failures": regressions,
            **local_metrics,
            **regression_metrics,
        }


class CandidatePromoter:
    def __init__(self, registry: ToolRegistrySystem, judge: ToolJudge) -> None:
        self.registry = registry
        self.judge = judge

    def maybe_promote(
        self,
        old_tool_id: str,
        candidate: ToolDescriptor,
        local_metrics: Dict[str, float],
        regression_metrics: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        incumbent = self.registry.get(old_tool_id)
        accepted, report = self.judge.score_candidate(candidate, incumbent, local_metrics, regression_metrics)
        if accepted:
            self.registry.replace(old_tool_id, candidate)
        return accepted, report


class LatentRouter(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.policy = MLP([latent_dim, hidden_dim, hidden_dim, num_actions])
        self.stop_head = MLP([latent_dim, hidden_dim, 1])

    def forward(self, packet: LatentPacket) -> Dict[str, torch.Tensor]:
        x = packet.state.primary()
        return {
            "action_logits": self.policy(x),
            "stop_logit": self.stop_head(x),
        }


class InternalTransitionTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, transition_model: ModularLatentWorldModel) -> None:
        super().__init__(descriptor)
        self.transition_model = transition_model

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        if context.action is None:
            raise ValueError("InternalTransitionTool requires context.action")
        memory_state = context.memory_state
        if memory_state is None:
            memory_state = self.transition_model.memory.init_state(packet.state.z_sem.shape[0], packet.state.z_sem.device)
        transition = self.transition_model.transition(packet.state, context.action, memory_state)
        next_packet = LatentPacket(
            state=transition.next_latent,
            confidence=transition.uncertainty,
            source_tool=self.descriptor.tool_id,
            trace=packet.trace + [self.descriptor.tool_id],
            metadata={**packet.metadata, **transition.aux},
        )
        return ToolResult(packet=next_packet, confidence=transition.uncertainty)


class TextDecoderTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, decoder: IDecoder) -> None:
        super().__init__(descriptor)
        self.decoder = decoder

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        outputs = self.decoder(packet.state, context.raw_inputs)
        confidence = None
        if "text_logits" in outputs:
            probs = outputs["text_logits"].softmax(dim=-1)
            confidence = probs.max(dim=-1).values.mean()
        return ToolResult(packet=packet, raw_output=outputs, confidence=confidence)


class MemoryReadTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, memory_bank: Dict[str, torch.Tensor]) -> None:
        super().__init__(descriptor)
        self.memory_bank = memory_bank

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        key = context.raw_inputs.get("memory_key", "default")
        value = self.memory_bank.get(key)
        if value is None:
            value = torch.zeros_like(packet.state.z_sem)
        next_state = LatentState(
            z_sem=packet.state.z_sem,
            z_dyn=packet.state.z_dyn,
            z_ctrl=packet.state.z_ctrl,
            z_mem=value,
            extras=dict(packet.state.extras),
        )
        next_packet = LatentPacket(
            state=next_state,
            source_tool=self.descriptor.tool_id,
            trace=packet.trace + [self.descriptor.tool_id],
            metadata=dict(packet.metadata),
        )
        return ToolResult(packet=next_packet)


class MemoryWriteTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, memory_bank: Dict[str, torch.Tensor]) -> None:
        super().__init__(descriptor)
        self.memory_bank = memory_bank

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        key = context.raw_inputs.get("memory_key", "default")
        self.memory_bank[key] = packet.state.z_mem if packet.state.z_mem is not None else packet.state.z_sem
        return ToolResult(packet=packet, side_effects={"memory_written": key})


class ToolExecutionEngine:
    def __init__(
        self,
        registry: ToolRegistrySystem,
        router: LatentRouter,
        action_id_to_tool: Dict[int, str],
        device: torch.device,
    ) -> None:
        self.registry = registry
        self.router = router.to(device)
        self.action_id_to_tool = action_id_to_tool
        self.device = device

    @torch.no_grad()
    def iterate(
        self,
        packet: LatentPacket,
        context: ToolContext,
        max_steps: int = 8,
    ) -> Tuple[LatentPacket, List[Dict[str, Any]]]:
        trace: List[Dict[str, Any]] = []
        current = packet
        for step in range(max_steps):
            route = self.router(current)
            stop_prob = torch.sigmoid(route["stop_logit"]).mean().item()
            action_id = int(route["action_logits"].argmax(dim=-1)[0].item())
            trace.append({"step": step, "stop_prob": stop_prob, "action_id": action_id})
            if stop_prob > 0.5:
                break
            tool_name = self.action_id_to_tool[action_id]
            result = self.registry.call(tool_name, current, context)
            if result.packet is not None:
                current = result.packet
            trace[-1]["tool"] = tool_name
            trace[-1]["confidence"] = None if result.confidence is None else float(result.confidence.detach().mean().cpu().item())
        return current, trace


# Entry-point builders for external lazy/binary tools.
# In a real package, move these into dedicated modules, e.g.:
#   latent_tool_ecosystem.tools.memory_tools
# and point ToolDescriptor.entry_point there.
def build_memory_read_tool(descriptor: ToolDescriptor) -> BaseTool:
    bank: Dict[str, torch.Tensor] = descriptor.config.setdefault("memory_bank", {})
    return MemoryReadTool(descriptor, bank)


def build_memory_write_tool(descriptor: ToolDescriptor) -> BaseTool:
    bank: Dict[str, torch.Tensor] = descriptor.config.setdefault("memory_bank", {})
    return MemoryWriteTool(descriptor, bank)


def build_internal_transition_tool(descriptor: ToolDescriptor) -> BaseTool:
    model = descriptor.config["transition_model"]
    return InternalTransitionTool(descriptor, model)


def build_text_decoder_tool(descriptor: ToolDescriptor) -> BaseTool:
    decoder = descriptor.config["decoder"]
    return TextDecoderTool(descriptor, decoder)


# ============================================================
# Example config and factory
# ============================================================


@dataclass
class ModelConfig:
    encoder_name: str = "simple_multimodal"
    latent_projector_name: str = "role_split_mlp"
    memory_name: str = "gru"
    action_encoder_name: str = "mlp"
    conditioner_name: str = "concat_mlp"
    transition_core_name: str = "recurrent_attnres_transformer"
    prediction_head_name: str = "role_split"
    regularizer_name: str = "sigreg_like"
    decoder_names: Tuple[str, ...] = ("text_autoregressive_head",)

    text_vocab_size: int = 32000
    text_embed_dim: int = 256
    vector_input_dim: int = 128
    image_channels: int = 3
    hidden_dim: int = 256
    latent_dim: int = 128
    action_dim: int = 32
    action_embed_dim: int = 128
    transition_hidden_dim: int = 512


def build_model(cfg: ModelConfig) -> ModularLatentWorldModel:
    encoder = ENCODERS.build(
        cfg.encoder_name,
        text_vocab_size=cfg.text_vocab_size,
        text_embed_dim=cfg.text_embed_dim,
        vector_input_dim=cfg.vector_input_dim,
        image_channels=cfg.image_channels,
        hidden_dim=cfg.hidden_dim,
    )
    latent_projector = LATENT_PROJECTORS.build(
        cfg.latent_projector_name,
        input_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        use_batchnorm=True,
    )
    memory = MEMORIES.build(
        cfg.memory_name,
        input_dim=cfg.latent_dim + cfg.action_embed_dim,
        hidden_dim=cfg.latent_dim,
    ) if cfg.memory_name == "gru" else MEMORIES.build(cfg.memory_name, latent_dim=cfg.latent_dim)
    action_encoder = ACTION_ENCODERS.build(
        cfg.action_encoder_name,
        action_dim=cfg.action_dim,
        action_embed_dim=cfg.action_embed_dim,
    ) if cfg.action_encoder_name == "mlp" else ACTION_ENCODERS.build(cfg.action_encoder_name, num_actions=cfg.action_dim, action_embed_dim=cfg.action_embed_dim)

    primary_latent_dim = cfg.latent_dim * 4
    conditioner = CONDITIONERS.build(
        cfg.conditioner_name,
        latent_dim=primary_latent_dim,
        action_dim=cfg.action_embed_dim,
        memory_dim=cfg.latent_dim,
        out_dim=cfg.transition_hidden_dim,
    ) if cfg.conditioner_name == "concat_mlp" else CONDITIONERS.build(
        cfg.conditioner_name,
        latent_dim=primary_latent_dim,
        action_dim=cfg.action_embed_dim,
        memory_dim=cfg.latent_dim,
    )

    transition_core = TRANSITION_CORES.build(
        cfg.transition_core_name,
        input_dim=cfg.transition_hidden_dim,
        hidden_dim=cfg.transition_hidden_dim,
    )
    prediction_head = PREDICTION_HEADS.build(
        cfg.prediction_head_name,
        hidden_dim=cfg.transition_hidden_dim,
        latent_dim=cfg.latent_dim,
    )
    regularizer = REGULARIZERS.build(cfg.regularizer_name)

    decoders: Dict[str, IDecoder] = {}
    for name in cfg.decoder_names:
        if name == "text_autoregressive_head":
            decoders[name] = DECODERS.build(name, vocab_size=cfg.text_vocab_size, latent_dim=cfg.latent_dim, text_embed_dim=cfg.text_embed_dim, hidden_dim=cfg.hidden_dim)
        elif name == "vector_reconstruction":
            decoders[name] = DECODERS.build(name, latent_dim=cfg.latent_dim, output_dim=cfg.vector_input_dim)
        else:
            raise KeyError(f"Unsupported decoder config: {name}")

    return ModularLatentWorldModel(
        encoder=encoder,
        latent_projector=latent_projector,
        memory=memory,
        action_encoder=action_encoder,
        conditioner=conditioner,
        transition_core=transition_core,
        prediction_head=prediction_head,
        regularizer=regularizer,
        decoders=decoders,
    )


def build_tool_ecosystem(
    model: ModularLatentWorldModel,
    device: torch.device,
    keep_hot_tools: int = 2,
    entrypoint_prefix: str = "__main__",
) -> Tuple[ToolRegistrySystem, ToolExecutionEngine, CriticalExperienceBuffer, CandidatePromoter]:
    """Build a binary/lazy-loaded tool ecosystem around the world model.

    Set entrypoint_prefix to the real importable module path when using package files,
    for example:
      entrypoint_prefix="latent_tool_ecosystem.tools.memory_tools"
    for memory tools, or split builders across modules as shown in the header note.
    """
    loader = BinaryToolLoader(device=device, keep_hot=keep_hot_tools)
    registry = ToolRegistrySystem(loader)

    shared_memory_bank: Dict[str, torch.Tensor] = {}

    registry.register(
        ToolDescriptor(
            tool_id="memory.read.v1",
            kind="memory.read",
            version="1.0",
            entry_point=f"{entrypoint_prefix}:build_memory_read_tool",
            benchmark_id="memory_read_basic",
            estimated_latency_ms=0.2,
            memory_mb=1.0,
            config={"memory_bank": shared_memory_bank},
        ),
        alias="memory.read",
    )
    registry.register(
        ToolDescriptor(
            tool_id="memory.write.v1",
            kind="memory.write",
            version="1.0",
            entry_point=f"{entrypoint_prefix}:build_memory_write_tool",
            benchmark_id="memory_write_basic",
            estimated_latency_ms=0.2,
            memory_mb=1.0,
            config={"memory_bank": shared_memory_bank},
        ),
        alias="memory.write",
    )

    registry.register(
        ToolDescriptor(
            tool_id="transition.internal.v1",
            kind="transition.latent",
            version="1.0",
            entry_point=f"{entrypoint_prefix}:build_internal_transition_tool",
            benchmark_id="transition_rollout_basic",
            estimated_latency_ms=1.0,
            memory_mb=64.0,
            config={"transition_model": model},
            binary=True,
            lazy_load=True,
        ),
        alias="transition.internal",
    )

    has_text_decoder = "text_autoregressive_head" in model.decoders
    if has_text_decoder:
        registry.register(
            ToolDescriptor(
                tool_id="decoder.text.v1",
                kind="decoder.text",
                version="1.0",
                entry_point=f"{entrypoint_prefix}:build_text_decoder_tool",
                benchmark_id="decoder_text_basic",
                estimated_latency_ms=0.5,
                memory_mb=16.0,
                config={"decoder": model.decoders["text_autoregressive_head"]},
                binary=True,
                lazy_load=True,
            ),
            alias="decoder.text",
        )

    primary_dim = model.latent_projector.sem.out_features * 4
    num_actions = 4 if has_text_decoder else 3
    router = LatentRouter(latent_dim=primary_dim, num_actions=num_actions)
    action_id_to_tool = {
        0: "transition.internal",
        1: "memory.read",
        2: "memory.write",
    }
    if has_text_decoder:
        action_id_to_tool[3] = "decoder.text"

    engine = ToolExecutionEngine(registry, router, action_id_to_tool, device=device)
    critical_buffer = CriticalExperienceBuffer()
    judge = ToolJudge(critical_buffer)
    promoter = CandidatePromoter(registry, judge)
    return registry, engine, critical_buffer, promoter


# ============================================================
# Minimal synthetic demo batch
# ============================================================


def make_dummy_batch(
    batch_size: int = 8,
    text_len: int = 16,
    vocab_size: int = 32000,
    vector_dim: int = 128,
    action_dim: int = 32,
    image_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    return {
        "text_t": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "text_tp1": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "vector_t": torch.randn(batch_size, vector_dim, device=device),
        "vector_tp1": torch.randn(batch_size, vector_dim, device=device),
        "image_t": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "image_tp1": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "action": torch.randn(batch_size, action_dim, device=device),
        "prefix_tokens": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "text_target": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "vector_target": torch.randn(batch_size, vector_dim, device=device),
    }


# ============================================================
# Example main
# ============================================================


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = WorldModelLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir="runs/modular_world_model",
        grad_clip_norm=1.0,
        mixed_precision=True,
    )

    for _ in range(10):
        batch = make_dummy_batch(device=device)
        metrics = trainer.train_step(batch)
        print({k: round(v, 4) for k, v in metrics.items()})

    registry, engine, critical_buffer, promoter = build_tool_ecosystem(
        model,
        device=device,
        keep_hot_tools=2,
        entrypoint_prefix="__main__",  # replace with package module path after splitting files
    )
    critical_buffer.add({
        "id": "decoder_regression_001",
        "benchmarks": ["decoder_text_basic"],
        "description": "Text decoder should remain callable after tool routing.",
        "weight": 10.0,
    })

    obs = ObservationPacket(
        modalities={
            "text": torch.randint(0, cfg.text_vocab_size, (1, 16), device=device),
            "vector": torch.randn(1, cfg.vector_input_dim, device=device),
            "image": torch.randn(1, 3, 64, 64, device=device),
        }
    )
    initial_state = model.encode(obs)
    packet = LatentPacket(state=initial_state, trace=["encode"])
    context = ToolContext(
        memory_state=model.memory.init_state(batch_size=1, device=device),
        observation=obs,
        action=torch.randn(1, cfg.action_dim, device=device),
        raw_inputs={
            "prefix_tokens": torch.randint(0, cfg.text_vocab_size, (1, 16), device=device),
            "memory_key": "default",
        },
    )
    final_packet, trace = engine.iterate(packet, context, max_steps=4)
    print("Tool iteration trace:", trace)
    print("Final packet trace:", final_packet.trace)

    candidate = ToolDescriptor(
        tool_id="memory.read.v2",
        kind="memory.read",
        version="2.0",
        entry_point="__main__:build_memory_read_tool",
        benchmark_id="memory_read_basic",
        estimated_latency_ms=0.15,
        memory_mb=1.2,
        config={"memory_bank": registry.get("memory.read").config["memory_bank"]},
    )
    accepted, report = promoter.maybe_promote(
        old_tool_id="memory.read.v1",
        candidate=candidate,
        local_metrics={"quality": 0.2, "latency_ms": 0.15, "memory_mb": 1.2},
        regression_metrics={"critical_failures": 0.0},
    )
    print("Candidate accepted:", accepted)
    print("Promotion report:", report)

    generated = trainer.generate_next_tokens(
        obs,
        torch.randn(1, cfg.action_dim, device=device),
        torch.randint(0, cfg.text_vocab_size, (1, 16), device=device),
        steps=8,
    )
    print("Generated token ids:", generated)
    print("Open TensorBoard in a browser with: tensorboard --logdir runs")


if __name__ == "__main__":
    main()

