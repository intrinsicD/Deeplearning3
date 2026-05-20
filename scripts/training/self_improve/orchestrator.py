"""Main training loop for the self-improvement protocol.

See ``docs/self_improvement.md`` §3 for the pseudocode this implements.
Phase 5 ships the single-component variant: one plugin, scheduler
trivial, eval+snapshot+rollback wired through the vault, replay/EMA/EWC
optional via ``attach_*`` calls on the plugin.

Phase 6 enables multi-component (the scheduler picks among several);
phase 7 adds the pseudo-label broker. The loop body below is shaped so
those phases extend it without restructure.

This module also wires the §4.4 capacity-expansion trigger
(:class:`scripts.training.self_improve.forgetting.PlateauDetector` →
``expand_omnilatent_capacity``) and the §6.2 per-component compute
budget into the main loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401  (kept for type hints in subclasses)
from typing import Any, Callable, Mapping

from scripts.training.self_improve.data_stream import DataStream, SyntheticDataStream
from scripts.training.self_improve.eval_registry import EvalRegistry
from scripts.training.self_improve.forgetting import PlateauDetector
from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
)
from scripts.training.self_improve.pseudo_labels import PseudoLabelBroker
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
    #: Per-component count of plateau-triggered capacity expansions
    #: that fired in this run.
    expansions: dict[str, int] = field(default_factory=dict)
    #: Per-component names that were paused this run because they hit
    #: their compute budget (§6.2).
    budget_exhausted: list[str] = field(default_factory=list)


#: Signature of the callable :class:`Orchestrator` invokes when a
#: plateau triggers on a component. Receives the plugin (so it can
#: introspect or mutate it) and the orchestrator (for access to the
#: data stream, scheduler, etc.). Returning ``True`` means expansion
#: occurred and the plateau detector should reset; ``False`` means
#: skip (and the detector stays latched).
ExpansionAction = Callable[[ComponentPlugin, "Orchestrator"], bool]


def _broker_inputs(batch: Any) -> list[Any] | None:
    """Extract per-sample inputs from a batch for the pseudo-label broker.

    The broker's label functions operate per-sample (they snapshot the
    producer's weights once and run the label_fn over the samples
    inside the swap). We need to slice the orchestrator's batch into
    per-sample tensors:

    - Plain tensor batch (shape ``(N, ...)``) ⇒ ``[batch[i] for i in
      range(N)]``.
    - Dict batch with an ``"image"`` key ⇒ slice the image tensor.
    - Anything else ⇒ ``None`` (the broker is image-centric for now;
      multimodal label flows land alongside the dataset adapters).
    """
    import torch as _torch

    if isinstance(batch, _torch.Tensor):
        return [batch[i] for i in range(batch.shape[0])]
    if isinstance(batch, dict) and "image" in batch:
        img = batch["image"]
        if isinstance(img, _torch.Tensor):
            return [img[i] for i in range(img.shape[0])]
    return None


def _default_expansion_action(
    plugin: ComponentPlugin, _orchestrator: "Orchestrator",
) -> bool:
    """Default plateau-trigger action: expand if the plugin is OmniLatent.

    The §4.4 capacity-expansion path is currently only wired for
    OmniLatent (its ``HookManager`` is the only one already plumbed
    through the forward pass). For other plugins this is a no-op until
    they grow their own expansion functions.
    """
    if plugin.name != "omnilatent":
        return False
    from scripts.training.self_improve.forgetting import expand_omnilatent_capacity

    expand_omnilatent_capacity(plugin)
    return True


class Orchestrator:
    """Drives plugin training under a scheduler.

    The orchestrator's responsibilities (per design doc §3):

    1. Pick a component (scheduler).
    2. Fetch a batch (data_stream).
    3. Run one ``train_step`` and record the loss.
    4. Every ``eval_every`` steps on each component, evaluate against
       the frozen probe set, snapshot the result, and roll back if it
       regressed beyond ``rollback_tol``.
    5. After each eval, optionally feed the score into a
       :class:`PlateauDetector`; on trigger, invoke an
       ``ExpansionAction`` (default: register a new hook on OmniLatent
       per §4.4).
    6. Optionally pause a component once its per-component compute
       budget is exhausted (§6.2).

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
        expansion_action: ExpansionAction | None = None,
        plateau_config: Mapping[str, Any] | None = None,
        compute_budget: Mapping[str, int] | int | None = None,
        pseudo_label_broker: PseudoLabelBroker | None = None,
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

        # --- Capacity expansion wiring (§4.4) ----------------------------
        # One detector per component, lazily constructed when the first
        # eval lands. Plateau config is shared across components but
        # subclasses can override _make_plateau_detector for per-component
        # tuning.
        self._expansion_action = expansion_action or _default_expansion_action
        self._plateau_config: dict[str, Any] = dict(plateau_config or {})
        self._plateau_detectors: dict[str, PlateauDetector] = {}
        self._expansions: dict[str, int] = {n: 0 for n in plugins}

        # --- Pseudo-label broker (§5) ------------------------------------
        # When attached, the orchestrator queries ``broker.sample`` for
        # each consumer before its ``train_step`` and drops the
        # resulting batches onto ``plugin._pending_pseudo_labels``. The
        # plugin consults the slot and folds a consistency loss into
        # its main backward.
        self.pseudo_label_broker = pseudo_label_broker

        # --- Compute budget (§6.2) --------------------------------------
        # ``compute_budget`` can be a single int (applied to every
        # component) or a dict mapping component name → cap. A component
        # whose persistent step count meets/exceeds its cap is paused for
        # the rest of subsequent runs (the scheduler still picks it, but
        # the orchestrator skips the train_step and reports it in
        # ``budget_exhausted``).
        self.compute_budget: dict[str, int] = self._normalize_budget(compute_budget)
        self._budget_exhausted_seen: set[str] = set()

        # Persistent per-component bookkeeping.
        self._steps_per_component: dict[str, int] = {n: 0 for n in plugins}
        self._rollbacks: dict[str, int] = {n: 0 for n in plugins}

    def _normalize_budget(
        self, budget: Mapping[str, int] | int | None,
    ) -> dict[str, int]:
        """Turn the constructor argument into a ``{name: cap}`` dict.

        ``None`` ⇒ no budget. Integer ⇒ same cap for every component.
        Dict ⇒ as-is (components not listed are unbounded).
        """
        if budget is None:
            return {}
        if isinstance(budget, int):
            return {n: int(budget) for n in self.plugins}
        out: dict[str, int] = {}
        for name, cap in budget.items():
            if name not in self.plugins:
                raise KeyError(
                    f"compute_budget references unknown component {name!r}; "
                    f"known: {list(self.plugins)}"
                )
            out[name] = int(cap)
        return out

    def _make_plateau_detector(self, name: str) -> PlateauDetector:
        cfg = self._plateau_config
        # ``higher_is_better`` is read from the first eval landing on
        # this component (see ``_evaluate_and_maybe_rollback``); the
        # detector's default is True but we override in the caller once
        # the eval signal is in.
        return PlateauDetector(
            patience=int(cfg.get("patience", 5)),
            tol=float(cfg.get("tol", 1e-3)),
            higher_is_better=bool(cfg.get("higher_is_better", True)),
        )

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
        pre_expansions = dict(self._expansions)

        report = RunReport(
            steps_per_component={n: 0 for n in self.plugins},
            rollbacks={n: 0 for n in self.plugins},
            step_losses={n: [] for n in self.plugins},
            expansions={n: 0 for n in self.plugins},
        )

        for global_step in range(max_steps):
            name = self.scheduler.pick()
            plugin = self.plugins[name]

            # Compute budget gate (§6.2). The scheduler still gets to
            # pick a budget-exhausted component (its priority just won't
            # be updated by a new eval), but we don't actually train on
            # it. This keeps the bandit's round-robin warmup well-defined
            # in the rare case every component is exhausted.
            cap = self.compute_budget.get(name)
            if cap is not None and self._steps_per_component[name] >= cap:
                if name not in self._budget_exhausted_seen:
                    logger.info(
                        "component %s reached compute budget %d; "
                        "subsequent picks skip training",
                        name, cap,
                    )
                    self._budget_exhausted_seen.add(name)
                # Still call record_step so the scheduler's staleness
                # counter advances; otherwise an exhausted component
                # would dominate the bandit's "staleness" priority and
                # starve the others.
                self.scheduler.record_step(name)
                continue

            batch = self.data_stream.next(plugin, batch_size=self.batch_size)

            # Pseudo-label broker (§5): sample for this consumer, hand
            # the resulting batches to the plugin via its pending-labels
            # slot, then advance the broker's per-consumer step counter
            # after the train_step so staleness budgeting stays correct.
            if self.pseudo_label_broker is not None:
                inputs = _broker_inputs(batch)
                if inputs is not None:
                    plugin._pending_pseudo_labels = self.pseudo_label_broker.sample(
                        name, inputs,
                    )

            step_report = plugin.train_step(batch)
            # Clear any unconsumed labels so they don't leak into the
            # next step (plugins that don't subclass the consumer mixin
            # never read them; clearing keeps the state clean).
            if self.pseudo_label_broker is not None:
                plugin._pending_pseudo_labels = None
                self.pseudo_label_broker.record_consumer_step(name)

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
        report.expansions = {
            n: self._expansions[n] - pre_expansions.get(n, 0)
            for n in self.plugins
        }
        report.total_steps = sum(report.steps_per_component.values())
        report.budget_exhausted = sorted(self._budget_exhausted_seen)
        # Final evals — one per component if a registry is configured.
        if self.eval_registry is not None:
            for n, plugin in self.plugins.items():
                if self.eval_registry.has(n):
                    report.final_evals[n] = plugin.evaluate(
                        self.eval_registry.get(n)
                    )
        return report

    # ------------------------------------------------------------------
    # Evaluation + rollback (design doc §4.6) + expansion (§4.4)
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

        # Feed the score into this component's plateau detector. We
        # construct lazily so the detector's higher_is_better matches
        # the actual eval convention rather than a global guess.
        detector = self._plateau_detectors.get(name)
        if detector is None:
            detector = self._make_plateau_detector(name)
            detector.higher_is_better = eval_report.higher_is_better
            self._plateau_detectors[name] = detector
        plateaued = detector.record(eval_report.score)
        if plateaued:
            try:
                fired = self._expansion_action(plugin, self)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "expansion action raised on %s: %s; leaving plateau "
                    "detector latched", name, exc,
                )
                fired = False
            if fired:
                self._expansions[name] += 1
                detector.reset()
                logger.info(
                    "plateau-triggered expansion fired on %s (total %d)",
                    name, self._expansions[name],
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


__all__ = ["ExpansionAction", "Orchestrator", "RunReport"]
