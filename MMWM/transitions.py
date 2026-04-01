"""Transition cores: MLP, GRU, Transformer, AttnRes, and Recurrent variants.

IMPORTANT: Transition cores do NOT own or update MemoryState. They receive
the conditioned input (which already incorporates memory context via the
conditioner) and return (hidden, aux). Memory updates happen in the IMemory
module, called by ModularLatentWorldModel.transition().
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .containers import MemoryState
from .helpers import (
    MLP,
    AdaptiveHaltingHead,
    BlockAttentionResidual,
    LowRankHyperAdapter,
    RMSNorm,
)
from .interfaces import TRANSITION_CORES, ITransitionCore


@TRANSITION_CORES.register("mlp")
class MLPTransitionCore(ITransitionCore):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, depth: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * depth
        self.net = MLP(dims, dropout=dropout)

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden = self.net(conditioned_input)
        return hidden, {}


@TRANSITION_CORES.register("gru")
class GRUTransitionCore(ITransitionCore):
    """GRU-based transition core with internal recurrent hidden state.

    Note: this core maintains its own recurrent hidden state separate from
    the IMemory module. The IMemory module handles long-term episodic memory,
    while this hidden state is the transition core's internal computation state.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Use transition-specific hidden state stored in memory_state.extras
        hidden_prev = memory_state.extras.get("transition_hidden")
        if hidden_prev is None:
            hidden_prev = torch.zeros(conditioned_input.shape[0], self.hidden_dim, device=conditioned_input.device)
        hidden = self.cell(conditioned_input, hidden_prev)
        # Store updated transition hidden for next step via aux
        return hidden, {"_transition_hidden": hidden}


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

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.input_proj(conditioned_input).unsqueeze(1)
        h = self.encoder(x).squeeze(1)
        h = self.norm(h)
        return h, {}


@TRANSITION_CORES.register("attnres_transformer")
class AttnResTransformerTransitionCore(ITransitionCore):
    """Transformer transition core with block Attention Residuals."""

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

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        return h, {
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

    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        current = self.input_proj(conditioned_input)
        halted_mask = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
        halting_steps = torch.zeros(current.shape[0], device=current.device)
        last_aux: Dict[str, torch.Tensor] = {}

        for step in range(self.recurrent_steps):
            core_hidden, aux = self.core(current, memory_state)
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
        result_aux = dict(last_aux)
        result_aux.update({
            "recurrent_steps_mean": halting_steps.mean(),
            "recurrent_steps_max": halting_steps.max(),
            "recurrent_hidden_norm": current.norm(dim=-1).mean(),
        })
        return current, result_aux
