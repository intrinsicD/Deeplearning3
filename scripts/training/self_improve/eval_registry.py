"""Frozen probe sets + the per-component evaluator registry.

Phase 2 ships **deterministic synthetic** probes — fixed-seed batches
that exercise each component's eval path without needing real datasets.
A later phase (alongside the dataset adapters) swaps these for the
production probe sets described in ``docs/self_improvement.md`` §6.1.

Two things matter for the rollback gate:

1. **Determinism.** Running ``plugin.evaluate(probe)`` in eval-mode
   twice with the same weights must produce the same numbers — the
   round-trip test depends on it.
2. **At least two metrics per component** (§6.4: reward-hacking
   resistance). The two metrics must be uncorrelated enough that gaming
   one doesn't trivially game the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import torch


# ---------------------------------------------------------------------------
# ProbeSet
# ---------------------------------------------------------------------------


@dataclass
class ProbeSet:
    """A frozen collection of batches used to score a component.

    ``batches`` is opaque to the registry — its element type is whatever
    the corresponding plugin's ``evaluate`` expects (tensor or batch dict).
    """

    component: str
    batches: list[Any]
    name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


# ---------------------------------------------------------------------------
# Probe builders — deterministic synthetic data per component
# ---------------------------------------------------------------------------


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def build_gaussian_encoder_probe(
    seed: int = 42,
    n_batches: int = 4,
    batch_size: int = 4,
    image_size: int = 28,
    in_ch: int = 1,
) -> ProbeSet:
    g = _seeded_generator(seed)
    batches = [
        torch.rand(batch_size, in_ch, image_size, image_size, generator=g)
        for _ in range(n_batches)
    ]
    return ProbeSet(
        component="gaussian_encoder",
        batches=batches,
        metadata={"seed": seed, "shape": [batch_size, in_ch, image_size, image_size]},
    )


def build_lgq_probe(
    seed: int = 42,
    n_batches: int = 4,
    batch_size: int = 2,
    resolution: int = 32,
    in_channels: int = 3,
) -> ProbeSet:
    g = _seeded_generator(seed)
    batches = [
        torch.rand(batch_size, in_channels, resolution, resolution, generator=g)
        for _ in range(n_batches)
    ]
    return ProbeSet(
        component="lgq",
        batches=batches,
        metadata={"seed": seed, "resolution": resolution},
    )


def build_omnilatent_probe(
    seed: int = 42,
    n_batches: int = 4,
    batch_size: int = 2,
    image_size: int = 32,
    image_channels: int = 3,
) -> ProbeSet:
    g = _seeded_generator(seed)
    batches = [
        {"image": torch.rand(batch_size, image_channels, image_size, image_size, generator=g)}
        for _ in range(n_batches)
    ]
    return ProbeSet(
        component="omnilatent",
        batches=batches,
        metadata={"seed": seed, "modality": "image"},
    )


def build_mmwm_probe(
    seed: int = 42,
    n_batches: int = 2,
    batch_size: int = 2,
    vector_dim: int = 128,
    action_dim: int = 32,
) -> ProbeSet:
    """Synthetic transition batches sourced from
    :class:`MMWM.data.DeterministicTransitionDataset` so that the data
    matches the trainer's expected batch dict exactly.

    Defaults to ``vector_dim=128`` / ``action_dim=32`` to align with the
    plugin's tiny config (``MMWM.config.ModelConfig`` defaults). Phase
    7+ datasets will replace this with real video-derived transitions.
    """
    from MMWM.data import DeterministicTransitionDataset, collate_transition_batch

    dataset = DeterministicTransitionDataset(
        length=batch_size * n_batches,
        vector_dim=vector_dim,
        action_dim=action_dim,
        seed=seed,
    )
    items = [dataset[i] for i in range(batch_size * n_batches)]
    batches = [
        collate_transition_batch(items[i * batch_size : (i + 1) * batch_size])
        for i in range(n_batches)
    ]
    return ProbeSet(
        component="mmwm",
        batches=batches,
        metadata={"seed": seed, "vector_dim": vector_dim, "action_dim": action_dim},
    )


def build_hpwm_probe(
    seed: int = 42,
    n_batches: int = 2,
    batch_size: int = 1,
    n_frames: int = 2,
    resolution: int = 64,
) -> ProbeSet:
    g = _seeded_generator(seed)
    batches = [
        {"frames": torch.rand(batch_size, n_frames, 3, resolution, resolution, generator=g)}
        for _ in range(n_batches)
    ]
    return ProbeSet(
        component="hpwm",
        batches=batches,
        metadata={"seed": seed, "n_frames": n_frames, "resolution": resolution},
    )


_DEFAULT_BUILDERS = {
    "gaussian_encoder": build_gaussian_encoder_probe,
    "lgq": build_lgq_probe,
    "omnilatent": build_omnilatent_probe,
    "mmwm": build_mmwm_probe,
    "hpwm": build_hpwm_probe,
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class EvalRegistry:
    """In-memory store of frozen probe sets keyed by component name."""

    def __init__(self) -> None:
        self._probes: dict[str, ProbeSet] = {}

    def register(self, probe_set: ProbeSet) -> None:
        if probe_set.component in self._probes:
            raise KeyError(
                f"probe set for component {probe_set.component!r} already registered"
            )
        self._probes[probe_set.component] = probe_set

    def get(self, component: str) -> ProbeSet:
        if component not in self._probes:
            raise KeyError(f"no probe set registered for component {component!r}")
        return self._probes[component]

    def has(self, component: str) -> bool:
        return component in self._probes

    def components(self) -> Iterable[str]:
        return self._probes.keys()

    @classmethod
    def with_defaults(cls, components: Iterable[str] | None = None) -> "EvalRegistry":
        """Build a registry pre-populated with synthetic probes."""
        reg = cls()
        for name in components or _DEFAULT_BUILDERS.keys():
            builder = _DEFAULT_BUILDERS.get(name)
            if builder is None:
                raise KeyError(f"no default probe builder for {name!r}")
            reg.register(builder())
        return reg


__all__ = [
    "EvalRegistry",
    "ProbeSet",
    "build_gaussian_encoder_probe",
    "build_hpwm_probe",
    "build_lgq_probe",
    "build_mmwm_probe",
    "build_omnilatent_probe",
]
