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

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, cast

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NeuralPortSpec:
    """Manifest for a differentiable latent extension.

    ``NeuralPortSpec`` is the typed replacement for ad-hoc hook construction.
    It keeps the old hook parameters (name, token count, latent dim, target
    layers) and adds extension metadata used by the agent/kernel runtime.
    """

    name: str
    kind: str
    version: str = "1"
    latent_dim: int = 512
    hook_tokens: int = 8
    target_layers: Sequence[int] = field(default_factory=tuple)
    reads: Sequence[str] = field(default_factory=tuple)
    writes: Sequence[str] = field(default_factory=tuple)
    side_effects: bool = False
    trainable: Sequence[str] = field(default_factory=lambda: ("hook_tokens", "gates", "transforms"))
    compatibility_loss: Dict[str, bool] = field(default_factory=dict)
    gate_bias_init: float = -4.0
    use_transform: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("NeuralPortSpec.name must not be empty")
        if not self.kind:
            raise ValueError("NeuralPortSpec.kind must not be empty")
        if not self.version:
            raise ValueError("NeuralPortSpec.version must not be empty")
        if self.latent_dim <= 0:
            raise ValueError("NeuralPortSpec.latent_dim must be > 0")
        if self.hook_tokens < 0:
            raise ValueError("NeuralPortSpec.hook_tokens must be >= 0")
        for layer in self.target_layers:
            if layer < 0:
                raise ValueError("NeuralPortSpec.target_layers must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "latent_dim": self.latent_dim,
            "hook_tokens": self.hook_tokens,
            "target_layers": list(self.target_layers),
            "reads": list(self.reads),
            "writes": list(self.writes),
            "side_effects": self.side_effects,
            "trainable": list(self.trainable),
            "compatibility_loss": dict(self.compatibility_loss),
            "gate_bias_init": self.gate_bias_init,
            "use_transform": self.use_transform,
            "metadata": dict(self.metadata),
        }


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

    def freeze(self) -> None:
        """Disable gradient updates for this hook/port."""
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self) -> None:
        """Enable gradient updates for this hook/port."""
        for param in self.parameters():
            param.requires_grad_(True)

    def gate_values(self) -> Dict[int, torch.Tensor]:
        """Return sigmoid gate values for all target layers."""
        return {int(layer): torch.sigmoid(gate) for layer, gate in self.gates.items()}


class NeuralPort(LatentNeuralHook):
    """Differentiable extension implemented as attention-participating tokens.

    This class intentionally subclasses :class:`LatentNeuralHook` so existing
    backbone injection/extraction behavior remains unchanged.
    """

    def __init__(self, spec: NeuralPortSpec) -> None:
        spec.validate()
        self.spec = spec
        super().__init__(
            name=spec.name,
            num_tokens=spec.hook_tokens,
            dim=spec.latent_dim,
            target_layers=spec.target_layers,
            gate_bias_init=spec.gate_bias_init,
            use_transform=spec.use_transform,
        )

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def version(self) -> str:
        return self.spec.version


class NeuralPortManager(nn.Module):
    """Manages active differentiable neural ports during a forward pass.

    The manager:
      * pre_layer: concatenates gated hook tokens to the sequence
      * post_layer: strips hook tokens, updates their state via transform

    Hook influence on content happens purely through attention — no
    broadcasting.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``hooks`` is the legacy public name and remains the canonical module
        # registry so existing checkpoints/tests keep working.
        self.hooks: nn.ModuleDict = nn.ModuleDict()
        self.specs: dict[str, NeuralPortSpec] = {}
        # Runtime state (set per forward pass)
        self._hook_states: dict[str, torch.Tensor] = {}
        self._batch_size: int = 0
        self._last_gate_log: dict[str, float] = {}

    def register_hook(self, hook: LatentNeuralHook) -> None:
        self.hooks[hook.name] = hook
        if isinstance(hook, NeuralPort):
            self.specs[hook.name] = hook.spec

    def register_port(self, port: NeuralPort | NeuralPortSpec) -> NeuralPort:
        """Register a neural port or construct one from its spec."""
        if isinstance(port, NeuralPortSpec):
            port = NeuralPort(port)
        self.register_hook(port)
        return port

    def remove_hook(self, name: str) -> LatentNeuralHook | None:
        if name in self.hooks:
            hook_module = self.hooks[name]
            if not isinstance(hook_module, LatentNeuralHook):
                raise TypeError(f"Registered module {name!r} is not a LatentNeuralHook")
            hook = cast(LatentNeuralHook, hook_module)
            del self.hooks[name]
            self.specs.pop(name, None)
            if name in self._hook_states:
                del self._hook_states[name]
            keys_to_remove = [k for k in self._last_gate_log if k.startswith(f"{name}.")]
            for key in keys_to_remove:
                del self._last_gate_log[key]
            return cast(LatentNeuralHook, hook)
        return None

    def remove_port(self, name: str) -> NeuralPort | LatentNeuralHook | None:
        """Remove a named port. Alias for legacy ``remove_hook``."""
        return self.remove_hook(name)

    def has_hooks(self) -> bool:
        return len(self.hooks) > 0

    def has_ports(self) -> bool:
        return self.has_hooks()

    def freeze_port(self, name: str) -> None:
        if name not in self.hooks:
            raise KeyError(name)
        cast(LatentNeuralHook, self.hooks[name]).freeze()

    def unfreeze_port(self, name: str) -> None:
        if name not in self.hooks:
            raise KeyError(name)
        cast(LatentNeuralHook, self.hooks[name]).unfreeze()

    def freeze_all(self) -> None:
        for hook in self.hooks.values():
            cast(LatentNeuralHook, hook).freeze()

    def unfreeze_all(self) -> None:
        for hook in self.hooks.values():
            cast(LatentNeuralHook, hook).unfreeze()

    def gate_log(self) -> dict[str, float]:
        """Return latest observed gate values as ``name.layer`` -> float."""
        return dict(self._last_gate_log)

    def gate_values(self) -> dict[str, dict[int, float]]:
        """Return current gate values for all registered ports/hooks."""
        values: dict[str, dict[int, float]] = {}
        for name, hook in self.hooks.items():
            hook = cast(LatentNeuralHook, hook)
            values[name] = {layer: float(value.detach().cpu().item()) for layer, value in hook.gate_values().items()}
        return values

    def begin_forward(self, batch_size: int) -> None:
        """Reset hook states for a new forward pass."""
        self._batch_size = batch_size
        self._hook_states = {}
        self._last_gate_log = {}
        for name, hook in self.hooks.items():
            hook = cast(LatentNeuralHook, hook)
            self._hook_states[name] = hook.get_hook_tokens(batch_size)

    def pre_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Inject gated hook tokens at the end of the sequence.

        Gate is applied to hook tokens BEFORE concatenation, controlling
        their magnitude in attention.
        """
        parts = [x]
        for name, hook in self.hooks.items():
            hook = cast(LatentNeuralHook, hook)
            if layer_idx in hook.target_layers:
                gate = hook.gate_value(layer_idx)
                self._last_gate_log[f"{name}.{layer_idx}"] = float(gate.detach().cpu().item())
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
        total_hook_tokens = 0
        for hook_module in self.hooks.values():
            hook = cast(LatentNeuralHook, hook_module)
            if layer_idx in hook.target_layers:
                total_hook_tokens += hook.num_tokens
        if total_hook_tokens == 0:
            return x

        # Split content and hook tokens
        content_len = x.shape[1] - total_hook_tokens
        content = x[:, :content_len]
        hook_region = x[:, content_len:]

        # Distribute hook tokens back to owners and update state
        offset = 0
        for name, hook in self.hooks.items():
            hook = cast(LatentNeuralHook, hook)
            if layer_idx in hook.target_layers:
                n = hook.num_tokens
                hook_out = hook_region[:, offset : offset + n]
                offset += n
                # Update persistent hook state (with optional transform)
                self._hook_states[name] = hook.transform_state(
                    layer_idx, hook_out
                )

        return content


class HookManager(NeuralPortManager):
    """Backward-compatible name for :class:`NeuralPortManager`."""


__all__ = [
    "HookManager",
    "LatentNeuralHook",
    "NeuralPort",
    "NeuralPortManager",
    "NeuralPortSpec",
]


