"""Expert registry for input-conditioned selection (work plan W2.1).

The registry is the set of *selectable experts* a learned router chooses among
for a given input (Audit.md P1, wish 2 — "identify the right pattern"). An
expert is anything the system can deploy to handle an input:

  * a **hook** — a :class:`~omnilatent.model.hooks.LatentNeuralHook` / neural
    port (a learned skill injected into attention);
  * a **tool** — a callable registered with the agent runtime;
  * a **kb** query — "retrieve relevant memory before deciding".

Every expert carries a learnable *key* vector. A router scores a pooled summary
of the input latent against these keys to produce routing weights. Keys are
deterministically seeded from the expert's id + tags (so a fresh registry with
the same experts starts from the same point), but are trainable thereafter.

This module deliberately holds **no routing logic** — it only owns the experts
and their keys. The router (W2.2) consumes ``keys()`` and ``ids()``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn as nn

EXPERT_KINDS = ("hook", "tool", "kb")


def _seed_from(text: str) -> int:
    """Stable 63-bit seed derived from a string (reproducible across runs)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class ExpertSpec:
    """Immutable description of one registered expert.

    ``action`` is the agent-graph action a router emits when this expert is
    selected. For tools it must be the **dispatch key** under which the tool is
    registered in the runtime (``AgentRuntime.tools``) — emitting a generic
    ``"TOOL_CALL"`` would lose which tool was chosen and the runtime could not
    execute it. Defaults are filled in by :meth:`ExpertRegistry.register`.
    """

    expert_id: str
    kind: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    action: str | None = None

    def __post_init__(self) -> None:
        if not self.expert_id:
            raise ValueError("ExpertSpec.expert_id must not be empty")
        if self.kind not in EXPERT_KINDS:
            raise ValueError(f"ExpertSpec.kind must be one of {EXPERT_KINDS}; got {self.kind!r}")


class ExpertRegistry(nn.Module):
    """A learnable-key registry over hooks ∪ tools ∪ KB-query experts.

    Args:
        key_dim: dimensionality of each expert key (match the router's query
            projection, typically the model hidden dim).
        key_init_scale: stddev of the seeded key initialization.
    """

    def __init__(self, key_dim: int, key_init_scale: float = 0.02) -> None:
        super().__init__()
        if key_dim <= 0:
            raise ValueError("key_dim must be positive")
        self.key_dim = key_dim
        self.key_init_scale = key_init_scale
        self._keys = nn.ParameterDict()
        self._specs: dict[str, ExpertSpec] = {}
        self._order: list[str] = []

    # -- registration ----------------------------------------------------
    def register(
        self,
        expert_id: str,
        kind: str,
        tags: Sequence[str] = (),
        action: str | None = None,
    ) -> ExpertSpec:
        """Register an expert with a deterministically-seeded learnable key.

        For ``kind="tool"`` the dispatch ``action`` defaults to ``expert_id``
        with a leading ``"tool:"`` stripped (so ``"tool:search"`` dispatches as
        ``"search"``); a router emits this action and the runtime looks it up in
        its ``tools`` map.
        """
        if expert_id in self._specs:
            raise ValueError(f"Expert {expert_id!r} already registered")
        if action is None and kind == "tool":
            action = expert_id.split(":", 1)[1] if expert_id.startswith("tool:") else expert_id
        spec = ExpertSpec(expert_id=expert_id, kind=kind, tags=tuple(tags), action=action)
        # Seed the key from id + tags so the same expert always starts the same.
        gen = torch.Generator().manual_seed(_seed_from("|".join((expert_id, *spec.tags))))
        key = torch.randn(self.key_dim, generator=gen) * self.key_init_scale
        self._keys[expert_id] = nn.Parameter(key)
        self._specs[expert_id] = spec
        self._order.append(expert_id)
        return spec

    def unregister(self, expert_id: str) -> bool:
        if expert_id not in self._specs:
            return False
        del self._keys[expert_id]
        del self._specs[expert_id]
        self._order.remove(expert_id)
        return True

    def sync_hooks(self, manager, *, prefix: str = "hook:") -> None:
        """Register a ``hook`` expert for every hook in a NeuralPortManager.

        Idempotent: hooks already present are skipped; hooks that disappeared
        from the manager are unregistered. Tags come from the port spec when
        available. This is how capacity-expansion hooks (work plan W4.1) become
        routable.
        """
        live = set()
        for name in manager.hooks.keys():
            expert_id = f"{prefix}{name}"
            live.add(expert_id)
            if expert_id in self._specs:
                continue
            spec = manager.specs.get(name)
            tags = tuple(spec.tags) if spec is not None else ()
            self.register(expert_id, "hook", tags=tags)
        # Drop experts whose hook was removed.
        for expert_id in [e for e in self._order if e.startswith(prefix) and e not in live]:
            self.unregister(expert_id)

    # -- queries ---------------------------------------------------------
    def ids(self) -> list[str]:
        """Expert ids in stable registration order (rows of ``keys()``)."""
        return list(self._order)

    def specs(self) -> list[ExpertSpec]:
        return [self._specs[e] for e in self._order]

    def spec(self, expert_id: str) -> ExpertSpec:
        return self._specs[expert_id]

    def kind(self, expert_id: str) -> str:
        return self._specs[expert_id].kind

    def ids_of_kind(self, kind: str) -> list[str]:
        return [e for e in self._order if self._specs[e].kind == kind]

    def tool_actions(self) -> dict[str, str]:
        """Map each tool expert's dispatch ``action`` → expert id.

        Use the keys to wire the agent graph (each maps to a ``TOOL_CALL``
        node) and the runtime's ``tools`` map, so a selected tool executes.
        """
        return {
            self._specs[e].action: e
            for e in self._order
            if self._specs[e].kind == "tool" and self._specs[e].action
        }

    def keys(self) -> torch.Tensor:
        """Stacked key matrix ``(num_experts, key_dim)`` in id order.

        Returns an empty ``(0, key_dim)`` tensor when nothing is registered.
        """
        if not self._order:
            return torch.zeros(0, self.key_dim)
        return torch.stack([self._keys[e] for e in self._order], dim=0)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, expert_id: object) -> bool:
        return expert_id in self._specs

    def __iter__(self) -> Iterable[str]:
        return iter(self._order)


__all__ = ["EXPERT_KINDS", "ExpertSpec", "ExpertRegistry"]
