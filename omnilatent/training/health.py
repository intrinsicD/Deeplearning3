"""Training health monitor with severe-pathology abort signals.

Watches the running loss and gradient statistics and raises
``TrainingDiverged`` when an invariant of a well-learning network is
violated badly enough that continuing is wasteful.

Invariants (see README/curriculum_train docs for the full discussion):

  Hard:
    1. Loss is finite.
    2. Gradients are finite.

  Soft (checked only after a calibration window):
    3. Loss is below ``init_loss`` (no sustained divergence).
    4. Loss EMA(short) trends below EMA(long) — or sits at a floor.
    5. Grad norm stays in ``[g_min, g_max]``.

The monitor accumulates statistics in ring buffers so the checks have a
constant memory footprint regardless of run length.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


class TrainingDiverged(RuntimeError):
    """Raised when the health monitor decides training cannot recover."""


@dataclass
class HealthThresholds:
    """Tunable abort thresholds.  Defaults are intentionally lenient."""

    # Hard invariants — checked every step.
    consecutive_bad_grad_steps: int = 3        # signal B

    # Soft invariants — checked once we have enough history.
    calibration_steps: int = 200               # warmup before any soft check
    plateau_window: int = 2000                 # signal D — sliding window length
    plateau_relative_improvement: float = 0.01 # require >1 % drop over the window
    divergence_factor: float = 1.5             # signal C — loss/init_loss ratio
    divergence_persistence: int = 200          # how long to tolerate divergence
    grad_min: float = 1e-8                     # signal E
    grad_max_factor: float = 100.0             # signal F — multiplied by clip_norm
    grad_persistence: int = 200                # consecutive bad-grad-norm steps

    # EMA smoothing for loss tracking.
    ema_short_alpha: float = 0.05              # ~20 step horizon
    ema_long_alpha: float = 0.001              # ~1000 step horizon


@dataclass
class HealthStats:
    """Snapshot of the monitor's state — useful for logging."""

    step: int = 0
    init_loss: float = math.nan
    ema_short: float = math.nan
    ema_long: float = math.nan
    last_loss: float = math.nan
    last_grad_norm: float = math.nan
    consecutive_bad_grad: int = 0
    consecutive_diverged: int = 0
    consecutive_grad_oob: int = 0
    plateau_buffer_len: int = 0
    aborted: bool = False
    abort_reason: str = ""


class TrainingHealthMonitor:
    """Per-step abort-on-pathology checker.

    Usage:
        monitor = TrainingHealthMonitor(clip_norm=1.0)
        ...
        monitor.record_init(first_loss)
        for step, batch in enumerate(loader):
            loss, grad_norm = train_step(batch)
            try:
                monitor.update(step, loss, grad_norm)
            except TrainingDiverged as exc:
                # Save final checkpoint, log, exit cleanly.
                ...
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        thresholds: HealthThresholds | None = None,
    ) -> None:
        self.clip_norm = clip_norm
        self.t = thresholds or HealthThresholds()
        self.stats = HealthStats()
        # Ring buffer of recent losses for plateau detection.
        self._loss_window: Deque[float] = deque(maxlen=self.t.plateau_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_init(self, loss: float) -> None:
        """Record the loss at step 0 (or the first finite loss).

        The divergence check uses this as its baseline.  Calling more than
        once is allowed — the monitor keeps the *first* finite value.
        """
        if not math.isfinite(loss):
            return
        if math.isnan(self.stats.init_loss):
            self.stats.init_loss = float(loss)
            self.stats.ema_short = float(loss)
            self.stats.ema_long = float(loss)

    def update(self, step: int, loss: float, grad_norm: float) -> None:
        """Run all checks for one optimizer step.  Raises ``TrainingDiverged``."""
        s = self.stats
        s.step = step
        s.last_loss = float(loss)
        s.last_grad_norm = float(grad_norm)

        self._check_finite(loss, grad_norm)
        # If loss/grad were finite, register the first one as init.
        self.record_init(loss)
        self._update_emas(loss)
        self._loss_window.append(float(loss))
        s.plateau_buffer_len = len(self._loss_window)

        # Soft checks — only after the calibration window.
        if step >= self.t.calibration_steps:
            self._check_divergence(loss)
            self._check_grad_band(grad_norm)
            self._check_plateau(step)

    def snapshot(self) -> HealthStats:
        """Return a copy of the current stats (for logging)."""
        s = self.stats
        return HealthStats(
            step=s.step,
            init_loss=s.init_loss,
            ema_short=s.ema_short,
            ema_long=s.ema_long,
            last_loss=s.last_loss,
            last_grad_norm=s.last_grad_norm,
            consecutive_bad_grad=s.consecutive_bad_grad,
            consecutive_diverged=s.consecutive_diverged,
            consecutive_grad_oob=s.consecutive_grad_oob,
            plateau_buffer_len=s.plateau_buffer_len,
            aborted=s.aborted,
            abort_reason=s.abort_reason,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_finite(self, loss: float, grad_norm: float) -> None:
        if not math.isfinite(loss):
            self._abort(f"non-finite loss ({loss}) at step {self.stats.step}")
        if not math.isfinite(grad_norm):
            self.stats.consecutive_bad_grad += 1
            if self.stats.consecutive_bad_grad >= self.t.consecutive_bad_grad_steps:
                self._abort(
                    f"non-finite grad_norm for "
                    f"{self.stats.consecutive_bad_grad} consecutive steps "
                    f"(last={grad_norm}) at step {self.stats.step}"
                )
        else:
            self.stats.consecutive_bad_grad = 0

    def _update_emas(self, loss: float) -> None:
        s, t = self.stats, self.t
        if math.isnan(s.ema_short):
            s.ema_short = float(loss)
            s.ema_long = float(loss)
            return
        s.ema_short += t.ema_short_alpha * (float(loss) - s.ema_short)
        s.ema_long += t.ema_long_alpha * (float(loss) - s.ema_long)

    def _check_divergence(self, loss: float) -> None:
        s, t = self.stats, self.t
        if math.isnan(s.init_loss):
            return
        threshold = s.init_loss * t.divergence_factor
        if loss > threshold:
            s.consecutive_diverged += 1
            if s.consecutive_diverged >= t.divergence_persistence:
                self._abort(
                    f"loss diverged: {loss:.4f} > {threshold:.4f} "
                    f"(= {t.divergence_factor}x init {s.init_loss:.4f}) "
                    f"for {s.consecutive_diverged} consecutive steps"
                )
        else:
            s.consecutive_diverged = 0

    def _check_grad_band(self, grad_norm: float) -> None:
        s, t = self.stats, self.t
        upper = self.clip_norm * t.grad_max_factor
        if grad_norm < t.grad_min or grad_norm > upper:
            s.consecutive_grad_oob += 1
            if s.consecutive_grad_oob >= t.grad_persistence:
                self._abort(
                    f"grad_norm out of band [{t.grad_min:g}, {upper:g}] "
                    f"for {s.consecutive_grad_oob} consecutive steps "
                    f"(last={grad_norm:g})"
                )
        else:
            s.consecutive_grad_oob = 0

    def _check_plateau(self, step: int) -> None:
        """Abort if the loss has not improved meaningfully over a long window."""
        t = self.t
        # Need a full window of history before this fires.
        if len(self._loss_window) < t.plateau_window:
            return
        old = self._loss_window[0]
        new = self._loss_window[-1]
        if not (math.isfinite(old) and math.isfinite(new)) or old <= 0:
            return
        relative = (old - new) / max(abs(old), 1e-12)
        if relative < t.plateau_relative_improvement:
            self._abort(
                f"loss plateau: only {relative * 100:.2f}% improvement over "
                f"the last {t.plateau_window} steps "
                f"(window {old:.4f} -> {new:.4f}); required >="
                f"{t.plateau_relative_improvement * 100:.1f}%"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _abort(self, reason: str) -> None:
        self.stats.aborted = True
        self.stats.abort_reason = reason
        raise TrainingDiverged(reason)
