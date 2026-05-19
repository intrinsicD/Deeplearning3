"""Self-improvement (continual self-supervised learning) infrastructure.

See ``docs/self_improvement.md`` for the design. This package currently
implements **phase 1** of the rollout: plugin scaffolding around the five
existing trainers, with a uniform ``ComponentPlugin`` interface but no
orchestrator, replay, EMA, EWC, or pseudo-label broker yet.
"""

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)

__all__ = ["ComponentPlugin", "EvalReport", "StepReport"]
