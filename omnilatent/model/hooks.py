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
        # Content-conditioned routing weights (work plan W3.1): per-hook
        # multiplier on the static gate, set by a router before the forward.
        # ``None``/absent ⇒ weight 1.0 (unconditioned, original behaviour).
        # A scalar 0 ⇒ the hook is skipped entirely this forward (its tokens
        # are never injected), giving *exact* recovery of prior behaviour.
        self._route_weights: dict[str, Any] = {}
        # Per-layer per-sample hook-token validity (B, H_layer): False where a
        # hook's per-sample route weight is 0, so those samples don't attend to
        # the hook's (zero-valued) tokens. Built in pre_layer, consumed by the
        # backbone via mask_inactive_hook_positions (work plan bug 2).
        self._hook_validity: dict[int, torch.Tensor] = {}

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

    def set_route_weights(self, weights: dict[str, Any] | None) -> None:
        """Set per-hook content-conditioned routing weights (work plan W3.1).

        ``weights`` maps hook name → multiplier (a Python float or a per-batch
        ``(B,)`` tensor) applied to that hook's static gate. A hook absent from
        the dict keeps weight 1.0. A scalar-0 weight skips the hook entirely.
        Pass ``None`` to clear all routing (back to unconditioned behaviour).

        Weights persist across :meth:`begin_forward` (the router sets them
        before the model's internal forward), so clear them explicitly.
        """
        self._route_weights = dict(weights) if weights else {}

    def _is_active(self, name: str) -> bool:
        """Whether a hook injects tokens this forward (route weight not all-zero)."""
        rw = self._route_weights.get(name)
        if rw is None:
            return True
        if isinstance(rw, torch.Tensor):
            return bool(torch.any(rw != 0))
        return rw != 0

    def _effective_gate(self, name: str, gate: torch.Tensor) -> torch.Tensor:
        """Static sigmoid gate scaled by the hook's route weight (if any)."""
        rw = self._route_weights.get(name)
        if rw is None:
            return gate
        if not isinstance(rw, torch.Tensor):
            rw = torch.as_tensor(rw, dtype=gate.dtype, device=gate.device)
        return gate * rw

    def begin_forward(self, batch_size: int) -> None:
        """Reset hook states for a new forward pass.

        Route weights are intentionally NOT reset here: a router sets them
        before the model's internal ``begin_forward`` call.
        """
        self._batch_size = batch_size
        self._hook_states = {}
        self._last_gate_log = {}
        self._hook_validity = {}
        for name, hook in self.hooks.items():
            hook = cast(LatentNeuralHook, hook)
            self._hook_states[name] = hook.get_hook_tokens(batch_size)

    def pre_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Inject gated hook tokens at the end of the sequence.

        Gate is applied to hook tokens BEFORE concatenation, controlling
        their magnitude in attention.
        """
        parts = [x]
        validity_parts: list[torch.Tensor] = []
        needs_mask = False
        for name, hook in self.hooks.items():
            hook = cast(LatentNeuralHook, hook)
            if layer_idx not in hook.target_layers:
                continue
            # Skip a route-weight-0 hook entirely: not injecting its tokens
            # (rather than injecting zero-valued tokens) gives exact recovery
            # of the no-hook behaviour, since zero tokens still occupy attention
            # positions (work plan W3.1).
            if not self._is_active(name):
                continue
            gate = self._effective_gate(name, hook.gate_value(layer_idx))
            self._last_gate_log[f"{name}.{layer_idx}"] = float(gate.detach().mean().cpu().item())
            state = self._hook_states[name]
            if gate.dim() == 0:
                scaled = state * gate
            else:
                # per-batch gate (B,) → broadcast over (B, tokens, dim)
                scaled = state * gate.view(-1, *([1] * (state.dim() - 1)))
            parts.append(scaled)

            # Per-sample validity for this hook's tokens: a sample whose route
            # weight is 0 must NOT attend to these (zero-valued) tokens, else
            # its output is not equal to the no-hook case (bug 2).
            rw = self._route_weights.get(name)
            if isinstance(rw, torch.Tensor) and rw.dim() > 0:
                valid = (rw != 0)
                needs_mask = needs_mask or bool((~valid).any())
            else:
                valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            validity_parts.append(valid.view(-1, 1).expand(-1, hook.num_tokens))

        if len(parts) == 1:
            return x
        if needs_mask:
            self._hook_validity[layer_idx] = torch.cat(validity_parts, dim=1)  # (B, H_layer)
        return torch.cat(parts, dim=1)

    def mask_inactive_hook_positions(
        self,
        layer_idx: int,
        attn_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Key-mask hook tokens that are inactive for specific samples (bug 2).

        For samples whose per-sample route weight for a hook is 0, mask that
        hook's token positions as **keys** so the samples' content never attends
        to the (zero-valued) tokens — making their output exactly the no-hook
        result. Only the key dimension is masked (not the query): a fully-masked
        query row would produce NaN, and the inactive hook's own output is
        discarded anyway (its next-layer contribution is gated to 0).
        """
        hv = self._hook_validity.get(layer_idx)
        if hv is None or bool(hv.all()):
            return attn_mask
        b, h = hv.shape
        content_len = seq_len - h
        content_valid = torch.ones(b, content_len, dtype=torch.bool, device=device)
        valid = torch.cat([content_valid, hv.to(device)], dim=1)  # (B, seq_len)
        if attn_mask is None:
            base = torch.ones(b, 1, seq_len, seq_len, dtype=torch.bool, device=device)
        else:
            heads = attn_mask.shape[-3] if attn_mask.dim() >= 3 else 1
            base = attn_mask.expand(b, heads, seq_len, seq_len).clone()
        # Mask the KEY dimension only.
        return base & valid[:, None, None, :]

    def post_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Strip hook tokens from the sequence and update their state.

        Hook influence on content happens purely through attention in the
        layer — no broadcasting bias is added here.
        """
        # Count how many hook tokens were injected at this layer. Must match
        # pre_layer exactly: only ACTIVE hooks (route weight != 0) were
        # injected, so only those are stripped here (work plan W3.1).
        total_hook_tokens = 0
        for name, hook_module in self.hooks.items():
            hook = cast(LatentNeuralHook, hook_module)
            if layer_idx in hook.target_layers and self._is_active(name):
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
            if layer_idx in hook.target_layers and self._is_active(name):
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


