"""Latent Neural Hook system.

Latent Neural Hooks (LNH) are a novel extensibility mechanism that injects
learnable latent vectors *directly into the transformer's attention
computation*.  Unlike adapters (which bolt modules onto outputs) or LoRA
(which modifies weights), hooks participate in self-attention as first-class
tokens -- they both read from and write to the model's internal
representations.

Key properties:
  1. **Participatory** -- hook tokens join the regular token sequence during
     attention, so they can attend to content tokens *and* be attended to.
  2. **Gated** -- a per-layer learnable gate controls influence strength.
     Initialized near zero for stable training (new hooks don't immediately
     perturb a trained model).
  3. **Persistent state** -- hook tokens carry state across layers, building
     up a representation of the information they extract/inject.
  4. **Composable** -- multiple hooks can be active simultaneously; they see
     each other in the attention window.
  5. **Transform network** -- an optional small MLP transforms hook states
     between layers, giving hooks their own internal processing.
  6. **Zero-cost removal** -- removing a hook is instant; the base model is
     never modified.

Typical use-cases:
  * Add a new modality (e.g. 3D point clouds) without touching existing code
  * Inject task-specific behaviour (e.g. style control)
  * Probe/monitor internal representations for interpretability
  * Inject retrieved knowledge directly into attention
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentNeuralHook(nn.Module):
    """A single Latent Neural Hook.

    Args:
        name: human-readable identifier.
        num_tokens: how many latent tokens this hook injects.
        dim: hidden dimension of the backbone (must match).
        target_layers: which transformer layer indices to participate in.
        gate_bias_init: initial bias for the sigmoid gate.  A large negative
            value (e.g. -4) means the hook starts nearly silent.
        use_transform: if True, a small 2-layer MLP transforms hook states
            between layers (gives the hook its own internal processing).
    """

    def __init__(
        self,
        name: str,
        num_tokens: int,
        dim: int,
        target_layers: Sequence[int],
        gate_bias_init: float = -4.0,
        use_transform: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.num_tokens = num_tokens
        self.dim = dim
        self.target_layers = set(target_layers)

        # Learnable hook latent tokens
        self.hook_tokens = nn.Parameter(
            torch.randn(1, num_tokens, dim) * 0.02
        )

        # Per-target-layer gating (starts near-zero)
        self.gates = nn.ParameterDict({
            str(l): nn.Parameter(torch.tensor(gate_bias_init))
            for l in target_layers
        })

        # Per-target-layer projection: maps the influence of hook tokens
        # back into the main sequence space
        self.influence_projs = nn.ModuleDict({
            str(l): nn.Linear(dim, dim, bias=False)
            for l in target_layers
        })

        # Optional inter-layer transform for hook state evolution
        self.use_transform = use_transform
        if use_transform:
            self.transforms = nn.ModuleDict({
                str(l): nn.Sequential(
                    nn.Linear(dim, dim * 2, bias=False),
                    nn.SiLU(),
                    nn.Linear(dim * 2, dim, bias=False),
                )
                for l in target_layers
            })

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in self.influence_projs.values():
            nn.init.zeros_(proj.weight)  # start with zero influence
        if self.use_transform:
            for t in self.transforms.values():
                # Zero-init the last linear so transform is identity at start
                nn.init.zeros_(t[-1].weight)

    @property
    def device(self) -> torch.device:
        return self.hook_tokens.device

    def get_hook_tokens(self, batch_size: int) -> torch.Tensor:
        """Return hook tokens expanded to batch size."""
        return self.hook_tokens.expand(batch_size, -1, -1)

    def gate_value(self, layer_idx: int) -> torch.Tensor:
        """Sigmoid-gated influence strength at a given layer."""
        return torch.sigmoid(self.gates[str(layer_idx)])

    def transform_state(
        self, layer_idx: int, hook_state: torch.Tensor
    ) -> torch.Tensor:
        """Evolve hook state between layers."""
        if self.use_transform and str(layer_idx) in self.transforms:
            return hook_state + self.transforms[str(layer_idx)](hook_state)
        return hook_state

    def compute_influence(
        self, layer_idx: int, hook_state: torch.Tensor
    ) -> torch.Tensor:
        """Compute the influence vector that will be added to main tokens.

        Returns a (B, 1, D) vector -- the mean-pooled, projected, gated
        hook representation.
        """
        proj = self.influence_projs[str(layer_idx)]
        gate = self.gate_value(layer_idx)
        # Mean-pool across hook tokens → (B, D)
        pooled = hook_state.mean(dim=1)
        return (gate * proj(pooled)).unsqueeze(1)  # (B, 1, D)


class HookManager(nn.Module):
    """Manages multiple active Latent Neural Hooks during a forward pass.

    The manager is responsible for:
      * Injecting hook tokens into the sequence before each targeted layer
      * Extracting hook tokens after each targeted layer
      * Accumulating hook state across layers
      * Computing and applying gated influence on main tokens
    """

    def __init__(self) -> None:
        super().__init__()
        self.hooks: nn.ModuleDict = nn.ModuleDict()
        # Runtime state (set per forward pass)
        self._hook_states: dict[str, torch.Tensor] = {}
        self._batch_size: int = 0

    def register_hook(self, hook: LatentNeuralHook) -> None:
        self.hooks[hook.name] = hook

    def remove_hook(self, name: str) -> LatentNeuralHook | None:
        if name in self.hooks:
            hook = self.hooks[name]
            del self.hooks[name]
            if name in self._hook_states:
                del self._hook_states[name]
            return hook
        return None

    def has_hooks(self) -> bool:
        return len(self.hooks) > 0

    def begin_forward(self, batch_size: int) -> None:
        """Reset hook states for a new forward pass."""
        self._batch_size = batch_size
        self._hook_states = {}
        for name, hook in self.hooks.items():
            self._hook_states[name] = hook.get_hook_tokens(batch_size)

    def pre_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Inject hook tokens into the sequence before a transformer layer.

        Hook tokens are concatenated at the end of the sequence so that
        they participate in full self-attention with all content tokens.
        """
        parts = [x]
        for name, hook in self.hooks.items():
            if layer_idx in hook.target_layers:
                state = self._hook_states[name]
                parts.append(state)
        if len(parts) == 1:
            return x
        return torch.cat(parts, dim=1)

    def post_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Extract hook tokens and apply gated influence after a layer.

        1. Separate content tokens from hook tokens.
        2. Update hook states (with optional transform).
        3. Compute gated influence and add to content tokens.
        """
        # Count how many hook tokens were injected at this layer
        total_hook_tokens = sum(
            hook.num_tokens
            for hook in self.hooks.values()
            if layer_idx in hook.target_layers
        )
        if total_hook_tokens == 0:
            return x

        # Split content and hook tokens
        content_len = x.shape[1] - total_hook_tokens
        content = x[:, :content_len]
        hook_region = x[:, content_len:]

        # Distribute hook tokens back to their owners and compute influence
        offset = 0
        for name, hook in self.hooks.items():
            if layer_idx in hook.target_layers:
                n = hook.num_tokens
                hook_out = hook_region[:, offset : offset + n]
                offset += n

                # Update persistent hook state
                hook_out = hook.transform_state(layer_idx, hook_out)
                self._hook_states[name] = hook_out

                # Gated influence on content tokens
                influence = hook.compute_influence(layer_idx, hook_out)
                content = content + influence

        return content
