"""Base ``ComponentPlugin`` interface and shared report types.

The orchestrator interacts with every component through this small surface.
Phase 1 implements only the training-time slice; replay, EWC, EMA, and
pseudo-label hooks are stubbed and become required in later phases.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # avoid import cycle at module-load time
    from scripts.training.self_improve.forgetting.ema import EMATeacher
    from scripts.training.self_improve.forgetting.ewc import EWCSI
    from scripts.training.self_improve.forgetting.replay import (
        ReplayBank,
        ReplayItem,
    )


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
    # Auxiliary forgetting state (phase 3)
    # ------------------------------------------------------------------
    #
    # These attributes are set by ``attach_replay`` / ``attach_ema`` and
    # are ``None`` by default — every plugin works without them. When
    # set, the plugin's ``train_step`` is expected to fold the
    # corresponding auxiliary loss into the gradient and to keep the
    # auxiliary state consistent (insert into the replay bank, step the
    # EMA after the optimizer step).

    replay_bank: "ReplayBank | None" = None
    ema_teacher: "EMATeacher | None" = None
    ewc: "EWCSI | None" = None
    #: Weight on the replay task loss (recompute task loss on a buffered
    #: batch) and the DER++ logits-consistency term. The defaults follow
    #: the DER++ paper's mid-range settings.
    replay_weight: float = 0.5
    der_weight: float = 0.5
    #: Weight on the EMA-teacher distillation loss.
    distill_weight: float = 0.5
    #: Number of replay samples drawn per training step.
    replay_batch_size: int = 1
    #: Weight on the cross-component pseudo-label consistency loss
    #: (§5). The default is conservative — pseudo-labels never
    #: dominate the real-data signal — matching the design-doc 30%
    #: real-data-majority cap (which the orchestrator enforces at the
    #: batch level; ``pseudo_weight`` shapes the loss-level mix).
    pseudo_weight: float = 0.3

    def attach_replay(
        self,
        bank: "ReplayBank",
        *,
        weight: float | None = None,
        der_weight: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Wire a shared :class:`ReplayBank` into this plugin.

        After attachment, ``train_step`` will sample from the bank each
        step, add a replay task loss + DER++ consistency loss, and
        insert the fresh batch back into the bank.
        """
        self.replay_bank = bank
        if weight is not None:
            self.replay_weight = float(weight)
        if der_weight is not None:
            self.der_weight = float(der_weight)
        if batch_size is not None:
            self.replay_batch_size = int(batch_size)

    def attach_ema(
        self,
        *,
        decay: float = 0.999,
        weight: float | None = None,
    ) -> None:
        """Initialize an :class:`EMATeacher` from the current model.

        Must be called *after* the plugin's model has been fully
        materialized (e.g. after HPWM's lazy DINO load). The teacher
        starts bit-identical to the student.
        """
        from scripts.training.self_improve.forgetting.ema import EMATeacher

        self.ema_teacher = EMATeacher(self.model, decay=decay)
        if weight is not None:
            self.distill_weight = float(weight)

    def attach_ewc(
        self,
        *,
        fisher_decay: float = 0.95,
        lam_ewc: float = 1.0,
        lam_si: float = 1.0,
    ) -> None:
        """Initialize the online EWC + SI regularizer from the current model.

        The anchor ``θ*`` starts at the current parameters; ``F`` and
        ``ω`` start at zero. The plugin's ``train_step`` is responsible
        for adding ``ewc.penalty(model)`` to the loss before backward,
        calling ``consolidate(model)`` between backward and step, and
        calling ``post_step(model)`` after the optimizer step.
        """
        from scripts.training.self_improve.forgetting.ewc import EWCSI

        self.ewc = EWCSI(
            self.model,
            fisher_decay=fisher_decay,
            lam_ewc=lam_ewc,
            lam_si=lam_si,
        )

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
        out = {"model": self.model.state_dict()}
        ema_state = self._ema_state()
        if ema_state is not None:
            out["_ema"] = ema_state
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self._load_ema_state(state.get("_ema"))

    def _ema_state(self) -> dict[str, Any] | None:
        """Helper for subclass ``state_dict`` impls.

        Returns the EMA teacher's state when one is attached, ``None``
        otherwise. Subclasses with custom ``state_dict`` should call
        this and merge the result under the ``"_ema"`` key so that
        vault rollback / resume restores teacher weights alongside the
        student — otherwise the teacher drifts away from the restored
        student and distillation loss becomes meaningless.
        """
        if self.ema_teacher is None:
            return None
        return self.ema_teacher.state_dict()

    def _load_ema_state(self, state: Any) -> None:
        """Companion to :meth:`_ema_state` for subclass ``load_state_dict``.

        Tolerates three cases without raising:

        - No EMA in the checkpoint and no teacher attached: no-op.
        - EMA in the checkpoint and a teacher attached: load.
        - EMA in the checkpoint but no teacher: lazily attach (so
          a resumed run with distillation enabled doesn't lose its
          teacher just because attach_ema hadn't yet been called).

        The fourth case (teacher attached but no EMA in the checkpoint)
        leaves the teacher untouched; resuming a non-distillation
        checkpoint into a distillation-enabled plugin is the caller's
        problem.
        """
        if state is None:
            return
        if self.ema_teacher is None:
            # Lazy reattach. Decay is read back from the saved state.
            from scripts.training.self_improve.forgetting.ema import EMATeacher

            self.ema_teacher = EMATeacher(self.model, decay=float(state["decay"]))
        self.ema_teacher.load_state_dict(state)

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
