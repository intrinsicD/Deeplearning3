"""Schedulers for the self-improvement orchestrator.

Two implementations:

- :class:`RoundRobinScheduler` — picks components in fixed order. Used
  in single-component mode (trivially) and as the warmup phase of the
  bandit. Sufficient through phase 6.
- :class:`BanditScheduler` — the full design-doc §3.1 priority bandit
  (regression gap, staleness, uncertainty). Currently a stub that
  inherits round-robin behavior; phase 6+ will fill in the priority
  formula once the orchestrator drives multi-component training and we
  have enough eval signal to compute the priority terms.

The :class:`Scheduler` ABC keeps both implementations interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence


class Scheduler(ABC):
    """Picks which component to train this step.

    Subclasses also receive ``record(component, ...)`` calls from the
    orchestrator after each step (for the bandit's state update);
    round-robin implementations may ignore them.
    """

    @abstractmethod
    def pick(self) -> str:
        """Return the next component to train."""

    def record_step(self, component: str) -> None:
        """Notify the scheduler that a training step ran on ``component``."""

    def record_eval(
        self, component: str, score: float, *, higher_is_better: bool,
    ) -> None:
        """Notify the scheduler of an evaluation outcome."""

    def components(self) -> Sequence[str]:
        return tuple()


class RoundRobinScheduler(Scheduler):
    """Cycles through components in the order provided.

    Trivial but sufficient: single-component mode passes ``[name]`` and
    every ``pick()`` returns that name. Phase 6's all-five mode passes
    all five and gets fair time-slicing.
    """

    def __init__(self, components: Sequence[str]) -> None:
        if not components:
            raise ValueError("RoundRobinScheduler requires ≥1 component")
        self._components: list[str] = list(components)
        self._idx = 0

    def pick(self) -> str:
        name = self._components[self._idx]
        self._idx = (self._idx + 1) % len(self._components)
        return name

    def components(self) -> Sequence[str]:
        return tuple(self._components)


class BanditScheduler(Scheduler):
    """Priority bandit per ``docs/self_improvement.md`` §3.1.

    State per component:
      ``b_i``  best score so far
      ``s_i``  latest score
      ``t_i``  steps since last eval
      ``p_i``  stagnation pressure (std of recent scores)

    Priority formula:
      priority_i = α · regression_gap_i + β · staleness_i + γ · uncertainty_i
      regression_gap_i = max(0, b_i − s_i)            (favors recovering losses)
      staleness_i      = t_i / EVAL_EVERY             (favors unrun components)
      uncertainty_i    = std(recent_eval_scores)       (favors noisy ones)

    A fixed quota (10%) is always spent on the worst-performing
    component to prevent abandonment.

    Phase 5 ships only the **warmup** behavior (round-robin until each
    component has at least one eval); full priority bandit lands in
    phase 6 alongside multi-component co-training. The state shape and
    record_* methods are final so plugin code doesn't need to change
    between phases.
    """

    def __init__(
        self,
        components: Sequence[str],
        *,
        warmup_rounds: int = 1,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.25,
        worst_quota: float = 0.1,
        eval_every: int = 100,
    ) -> None:
        if not components:
            raise ValueError("BanditScheduler requires ≥1 component")
        self._components = list(components)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.worst_quota = worst_quota
        self.eval_every = eval_every

        # Round-robin warmup state.
        self._warmup_rounds = max(0, int(warmup_rounds))
        self._warmup_idx = 0

        # Per-component bookkeeping.
        self.best: dict[str, float] = {}
        self.latest: dict[str, float] = {}
        self.steps_since_eval: dict[str, int] = defaultdict(int)
        self.eval_history: dict[str, list[float]] = defaultdict(list)
        self.higher_is_better: dict[str, bool] = {}
        self.total_picks: dict[str, int] = defaultdict(int)

    def pick(self) -> str:
        if self._in_warmup():
            name = self._components[self._warmup_idx % len(self._components)]
            self._warmup_idx += 1
            self.total_picks[name] += 1
            return name
        # Post-warmup: phase 6 fills in the priority formula. For now we
        # round-robin so existing behavior is well-defined.
        name = self._components[self._warmup_idx % len(self._components)]
        self._warmup_idx += 1
        self.total_picks[name] += 1
        return name

    def _in_warmup(self) -> bool:
        if self._warmup_rounds == 0:
            return False
        required = self._warmup_rounds * len(self._components)
        return self._warmup_idx < required

    def record_step(self, component: str) -> None:
        self.steps_since_eval[component] += 1

    def record_eval(
        self, component: str, score: float, *, higher_is_better: bool,
    ) -> None:
        self.eval_history[component].append(score)
        self.latest[component] = score
        self.higher_is_better[component] = higher_is_better
        prev_best = self.best.get(component)
        if prev_best is None:
            self.best[component] = score
        else:
            self.best[component] = (
                max(prev_best, score) if higher_is_better else min(prev_best, score)
            )
        self.steps_since_eval[component] = 0

    def components(self) -> Sequence[str]:
        return tuple(self._components)


__all__ = ["BanditScheduler", "RoundRobinScheduler", "Scheduler"]
