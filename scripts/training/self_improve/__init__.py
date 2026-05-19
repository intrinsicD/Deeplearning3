"""Self-improvement (continual self-supervised learning) infrastructure.

See ``docs/self_improvement.md`` for the design. This package implements
the rollout up through **phase 3**:

- Phase 1: plugin scaffolding around the five existing trainers.
- Phase 2: content-addressed vault + frozen-probe eval registry.
- Phase 3: DER++ replay buffer + EMA teacher distillation; all five
  plugins accept ``attach_replay`` / ``attach_ema``.

Pending: EWC + SI (phase 4), orchestrator (phase 5), multi-component
co-training (phase 6), pseudo-label broker (phase 7), hook-based
capacity expansion (phase 8), A-GEM (phase 9), control-center wiring
(phase 10).
"""

from scripts.training.self_improve.eval_registry import EvalRegistry, ProbeSet
from scripts.training.self_improve.forgetting import (
    EMATeacher,
    EWCSI,
    ReplayBank,
    ReplayItem,
)
from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)
from scripts.training.self_improve.vault import Snapshot, SnapshotID, Vault

__all__ = [
    "ComponentPlugin",
    "EMATeacher",
    "EWCSI",
    "EvalRegistry",
    "EvalReport",
    "ProbeSet",
    "ReplayBank",
    "ReplayItem",
    "Snapshot",
    "SnapshotID",
    "StepReport",
    "Vault",
]
