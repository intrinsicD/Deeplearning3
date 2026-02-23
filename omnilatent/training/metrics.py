"""Training metrics collection and logging.

Provides a MetricsAggregator that accumulates per-step and per-epoch
metrics, computes running averages, and formats them for logging.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    """Metrics from a single training step."""
    loss_total: float = 0.0
    loss_per_modality: dict[str, float] = field(default_factory=dict)
    grad_norm: float = 0.0
    source_modality: str = ""
    target_modality: str = ""
    learning_rate: float = 0.0


class MetricsAggregator:
    """Accumulates training metrics and computes epoch summaries."""

    def __init__(self) -> None:
        self.step_count: int = 0
        self._running_loss: float = 0.0
        self._modality_losses: dict[str, list[float]] = defaultdict(list)
        self._task_counts: dict[str, int] = defaultdict(int)
        self._grad_norms: list[float] = []

    def log_step(self, metrics: StepMetrics) -> None:
        """Record metrics from a training step."""
        self.step_count += 1
        self._running_loss += metrics.loss_total

        for mod, loss_val in metrics.loss_per_modality.items():
            self._modality_losses[mod].append(loss_val)

        task_key = f"{metrics.source_modality}->{metrics.target_modality}"
        self._task_counts[task_key] += 1

        if metrics.grad_norm > 0:
            self._grad_norms.append(metrics.grad_norm)

    @property
    def avg_loss(self) -> float:
        if self.step_count == 0:
            return 0.0
        return self._running_loss / self.step_count

    def avg_loss_per_modality(self) -> dict[str, float]:
        return {
            mod: (sum(vals) / len(vals) if vals else 0.0)
            for mod, vals in self._modality_losses.items()
        }

    def avg_grad_norm(self) -> float:
        if not self._grad_norms:
            return 0.0
        return sum(self._grad_norms) / len(self._grad_norms)

    def task_distribution(self) -> dict[str, int]:
        return dict(self._task_counts)

    def summary(self) -> dict[str, float | dict]:
        """Return a summary dict for logging."""
        return {
            "steps": self.step_count,
            "avg_loss": self.avg_loss,
            "avg_loss_per_modality": self.avg_loss_per_modality(),
            "avg_grad_norm": self.avg_grad_norm(),
            "task_distribution": self.task_distribution(),
        }

    def reset(self) -> None:
        """Reset all accumulated metrics (call at epoch boundary)."""
        self.step_count = 0
        self._running_loss = 0.0
        self._modality_losses.clear()
        self._task_counts.clear()
        self._grad_norms.clear()
