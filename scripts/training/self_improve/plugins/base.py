"""Base ``ComponentPlugin`` interface and shared report types.

The orchestrator interacts with every component through this small surface.
Phase 1 implements only the training-time slice; replay, EWC, EMA, and
pseudo-label hooks are stubbed and become required in later phases.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class StepReport:
    """Result of a single training step.

    ``loss`` is the scalar the optimizer stepped on; ``losses`` is a flat
    dict of named sub-loss values (for logging and diagnostics).
    """

    loss: float
    losses: dict[str, float] = field(default_factory=dict)
    grad_norm: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def is_finite(self) -> bool:
        if not math.isfinite(self.loss):
            return False
        return all(math.isfinite(v) for v in self.losses.values())


@dataclass
class EvalReport:
    """Result of an evaluation pass on a frozen probe set.

    ``score`` is a single scalar the scheduler uses for priority and
    rollback decisions; ``metrics`` is the full breakdown.
    ``higher_is_better`` reflects the convention of ``score``.
    """

    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    higher_is_better: bool = True


# ---------------------------------------------------------------------------
# Plugin base class
# ---------------------------------------------------------------------------


class ComponentPlugin(ABC):
    """Abstract base for every component plugin.

    Subclasses are responsible for owning their model, optimizer, loss,
    and (where applicable) scheduler. The orchestrator never touches
    those directly — it only calls the methods declared here.
    """

    #: Stable identifier used in checkpoint paths, configs, and logs.
    name: ClassVar[str] = "unset"

    # ------------------------------------------------------------------
    # Required surface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """The component's primary trainable module."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the model lives on."""

    @abstractmethod
    def make_synthetic_batch(self, batch_size: int = 2) -> Any:
        """Build a tiny batch of synthetic inputs suitable for one step.

        Used by smoke tests in phase 1 and by the orchestrator's dry-run
        mode in later phases. Plugins should make this fast and avoid
        downloading any data.
        """

    @abstractmethod
    def train_step(self, batch: Any) -> StepReport:
        """Run one optimizer step on *batch*."""

    # ------------------------------------------------------------------
    # Optional surface (defaults raise NotImplementedError)
    # ------------------------------------------------------------------

    def evaluate(self, probe_set: Any | None = None) -> EvalReport:
        """Score the component on a frozen probe set.

        Phase 1 plugins may leave this unimplemented; phase 2 wires this
        into the eval registry and vault.
        """
        raise NotImplementedError(f"{self.name}.evaluate is not implemented yet")

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the plugin's mutable state."""
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.model.parameters()

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} device={self.device}>"


__all__ = ["ComponentPlugin", "EvalReport", "StepReport"]
