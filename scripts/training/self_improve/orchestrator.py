"""Main training loop for the self-improvement protocol.

See ``docs/self_improvement.md`` §3 for the pseudocode this implements.
Phase 5 ships the single-component variant: one plugin, scheduler
trivial, eval+snapshot+rollback wired through the vault, replay/EMA/EWC
optional via ``attach_*`` calls on the plugin.

Phase 6 enables multi-component (the scheduler picks among several);
phase 7 adds the pseudo-label broker. The loop body below is shaped so
those phases extend it without restructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401  (kept for type hints in subclasses)
from typing import Any  # noqa: F401

from scripts.training.self_improve.data_stream import DataStream, SyntheticDataStream
from scripts.training.self_improve.eval_registry import EvalRegistry
from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)
from scripts.training.self_improve.scheduler import (
    RoundRobinScheduler,
    Scheduler,
)
from scripts.training.self_improve.vault import Vault

logger = logging.getLogger(__name__)


@dataclass
class RunReport:
    """Outcome of an :meth:`Orchestrator.run` call.

    Used by the CLI for logging, and by tests for assertions. The
    per-component step histories and final eval scores are small —
    storing them in-memory is fine through phase 6's multi-component
    scale (5 components × thousands of steps).
    """

    total_steps: int = 0
    steps_per_component: dict[str, int] = field(default_factory=dict)
    final_evals: dict[str, EvalReport] = field(default_factory=dict)
    rollbacks: dict[str, int] = field(default_factory=dict)
    step_losses: dict[str, list[float]] = field(default_factory=dict)


class Orchestrator:
    """Drives plugin training under a scheduler.

    The orchestrator's responsibilities (per design doc §3):

    1. Pick a component (scheduler).
    2. Fetch a batch (data_stream).
    3. Run one ``train_step`` and record the loss.
    4. Every ``eval_every`` steps on each component, evaluate against
       the frozen probe set, snapshot the result, and roll back if it
       regressed beyond ``rollback_tol``.

    The plugin handles its own forgetting-mitigation (replay, EMA, EWC)
    inside ``train_step``; the orchestrator is composition glue, not
    gradient-aware.
    """

    def __init__(
        self,
        plugins: dict[str, ComponentPlugin],
        *,
        scheduler: Scheduler | None = None,
        data_stream: DataStream | None = None,
        eval_registry: EvalRegistry | None = None,
        vault: Vault | None = None,
        eval_every: int = 100,
        rollback_tol: float = 0.0,
        batch_size: int = 2,
        log_every: int = 10,
        seed: int | None = None,
    ) -> None:
        if not plugins:
            raise ValueError("Orchestrator requires ≥1 plugin")
        self.plugins = dict(plugins)

        if scheduler is None:
            scheduler = RoundRobinScheduler(list(plugins.keys()))
        # Verify the scheduler only references known plugins.
        for c in scheduler.components():
            if c not in self.plugins:
                raise KeyError(
                    f"scheduler references unknown component {c!r}; "
                    f"known: {list(self.plugins)}"
                )
        self.scheduler = scheduler
        self.data_stream = data_stream or SyntheticDataStream(batch_size=batch_size)
        self.eval_registry = eval_registry
        self.vault = vault
        self.eval_every = int(eval_every)
        self.rollback_tol = float(rollback_tol)
        self.batch_size = int(batch_size)
        self.log_every = int(log_every)

        # Note: we do NOT call ``torch.manual_seed`` here, even when
        # ``seed`` is passed. Plugins are constructed by the caller and
        # consume RNG before they reach us; seeding inside __init__
        # would silently desync the orchestrator's trajectory from the
        # equivalent direct-training trajectory (which the CLI and
        # parity tests rely on). The ``seed`` parameter is retained for
        # forward compatibility with the multi-worker data stream in
        # phase 7 but is currently unused.
        self._seed = seed

        # Persistent per-component bookkeeping.
        self._steps_per_component: dict[str, int] = {n: 0 for n in plugins}
        self._rollbacks: dict[str, int] = {n: 0 for n in plugins}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_steps: int) -> RunReport:
        """Execute ``max_steps`` training steps and return a summary.

        The returned :class:`RunReport` reports counts **for this
        invocation only** — staged training that reuses one orchestrator
        across multiple ``run`` calls sees per-call deltas, not the
        cumulative totals. Persistent counters
        (``self._steps_per_component`` / ``self._rollbacks``) keep
        accumulating across calls for the scheduler's benefit, but the
        report is a snapshot of "what happened on this call".
        """
        # Snapshot counters before the loop so we can compute per-call
        # deltas at the end. A naive ``+= 1`` against ``report.X[name]``
        # inside the loop would have the same effect but would force the
        # report dict to be initialized to ``0`` for every component;
        # the snapshot avoids that coupling.
        pre_steps = dict(self._steps_per_component)
        pre_rollbacks = dict(self._rollbacks)

        report = RunReport(
            steps_per_component={n: 0 for n in self.plugins},
            rollbacks={n: 0 for n in self.plugins},
            step_losses={n: [] for n in self.plugins},
        )

        for global_step in range(max_steps):
            name = self.scheduler.pick()
            plugin = self.plugins[name]

            batch = self.data_stream.next(plugin, batch_size=self.batch_size)
            step_report = plugin.train_step(batch)
            self._steps_per_component[name] += 1
            report.step_losses[name].append(step_report.loss)
            self.scheduler.record_step(name)

            if self.log_every > 0 and (global_step + 1) % self.log_every == 0:
                logger.info(
                    "step=%d component=%s loss=%.4f%s",
                    global_step + 1,
                    name,
                    step_report.loss,
                    " (non-finite)" if not step_report.is_finite() else "",
                )

            # Periodic evaluation. The trigger is per-component step
            # count (not global step count) so multi-component runs eval
            # each component every ``eval_every`` of its own steps.
            comp_steps = self._steps_per_component[name]
            if self.eval_every > 0 and comp_steps % self.eval_every == 0:
                self._evaluate_and_maybe_rollback(name, plugin)

        # Per-call deltas, not cumulative totals.
        report.steps_per_component = {
            n: self._steps_per_component[n] - pre_steps.get(n, 0)
            for n in self.plugins
        }
        report.rollbacks = {
            n: self._rollbacks[n] - pre_rollbacks.get(n, 0)
            for n in self.plugins
        }
        report.total_steps = sum(report.steps_per_component.values())
        # Final evals — one per component if a registry is configured.
        if self.eval_registry is not None:
            for n, plugin in self.plugins.items():
                if self.eval_registry.has(n):
                    report.final_evals[n] = plugin.evaluate(
                        self.eval_registry.get(n)
                    )
        return report

    # ------------------------------------------------------------------
    # Evaluation + rollback (design doc §4.6)
    # ------------------------------------------------------------------

    def _evaluate_and_maybe_rollback(
        self, name: str, plugin: ComponentPlugin,
    ) -> EvalReport:
        probe = (
            self.eval_registry.get(name)
            if self.eval_registry is not None and self.eval_registry.has(name)
            else None
        )
        eval_report = plugin.evaluate(probe)
        self.scheduler.record_eval(
            name,
            eval_report.score,
            higher_is_better=eval_report.higher_is_better,
        )

        if self.vault is None:
            return eval_report

        # Compare against the current best before saving the new snapshot.
        prev_best_id = self.vault.best(name)
        regressed = False
        prev_best_score: float | None = None
        if prev_best_id is not None:
            prev_best = self.vault.load(prev_best_id)
            prev_best_score = prev_best.report.score
            if eval_report.higher_is_better:
                regressed = (
                    eval_report.score < prev_best_score - self.rollback_tol
                )
            else:
                regressed = (
                    eval_report.score > prev_best_score + self.rollback_tol
                )

        comp_steps = self._steps_per_component[name]
        if regressed:
            logger.warning(
                "regression detected on %s (score %.4f vs best %.4f); "
                "rolling back",
                name,
                eval_report.score,
                prev_best_score,
            )
            self.vault.rollback(name, plugin)
            self._rollbacks[name] += 1
        else:
            self.vault.save(name, plugin, eval_report, step=comp_steps)

        return eval_report


__all__ = ["Orchestrator", "RunReport"]
