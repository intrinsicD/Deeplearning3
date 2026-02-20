"""Latent Neural Hook system.

Latent Neural Hooks (LNH) inject learnable latent vectors *directly into
the transformer's attention computation*.  Hook tokens participate in
self-attention as first-class tokens — they attend to content tokens
and are attended to by them.

Key properties:
  1. **Participatory** — hook tokens join the sequence during attention.
  2. **Gated** — a per-layer sigmoid gate scales hook token magnitude
     before injection.  Initialized near zero for stable training.
  3. **Persistent state** — hook tokens carry state across layers.
  4. **Composable** — multiple hooks can be active simultaneously.
  5. **Transform network** — optional small MLP evolves hook states
     between layers.
  6. **Zero-cost removal** — removing a hook is instant; the base model
     is never modified.

Unlike the previous design, hooks influence content *purely through
attention* — there is no mean-pooled broadcast bias added to content
tokens.  This eliminates a global-bias artifact and lets the attention
mechanism properly route information.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


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
        """Evolve hook state between layers (residual connection)."""
        if self.use_transform and str(layer_idx) in self.transforms:
            return hook_state + self.transforms[str(layer_idx)](hook_state)
        return hook_state


class HookManager(nn.Module):
    """Manages multiple active Latent Neural Hooks during a forward pass.

    The manager:
      * pre_layer: concatenates gated hook tokens to the sequence
      * post_layer: strips hook tokens, updates their state via transform

    Hook influence on content happens purely through attention — no
    broadcasting.
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
        """Inject gated hook tokens at the end of the sequence.

        Gate is applied to hook tokens BEFORE concatenation, controlling
        their magnitude in attention.
        """
        parts = [x]
        for name, hook in self.hooks.items():
            if layer_idx in hook.target_layers:
                gate = hook.gate_value(layer_idx)
                parts.append(self._hook_states[name] * gate)
        if len(parts) == 1:
            return x
        return torch.cat(parts, dim=1)

    def post_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Strip hook tokens from the sequence and update their state.

        Hook influence on content happens purely through attention in the
        layer — no broadcasting bias is added here.
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

        # Distribute hook tokens back to owners and update state
        offset = 0
        for name, hook in self.hooks.items():
            if layer_idx in hook.target_layers:
                n = hook.num_tokens
                hook_out = hook_region[:, offset : offset + n]
                offset += n
                # Update persistent hook state (with optional transform)
                self._hook_states[name] = hook.transform_state(
                    layer_idx, hook_out
                )

        return content
