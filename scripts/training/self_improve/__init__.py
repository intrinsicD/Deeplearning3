"""Self-improvement (continual self-supervised learning) infrastructure.

See ``docs/self_improvement.md`` for the design. This package implements
the rollout up through **phase 5**:

- Phase 1: plugin scaffolding around the five existing trainers.
- Phase 2: content-addressed vault + frozen-probe eval registry.
- Phase 3: DER++ replay buffer + EMA teacher distillation; all five
  plugins accept ``attach_replay`` / ``attach_ema``.
- Phase 4: online EWC + Synaptic Intelligence; wired into OmniLatent
  and MMWM.
- Phase 5: orchestrator, scheduler, data stream, and CLI for
  single-component runs.

Pending: multi-component co-training (phase 6), pseudo-label broker
(phase 7), hook-based capacity expansion (phase 8), A-GEM (phase 9),
control-center wiring (phase 10).
"""

from scripts.training.self_improve.data_stream import DataStream, SyntheticDataStream
from scripts.training.self_improve.eval_registry import EvalRegistry, ProbeSet
from scripts.training.self_improve.forgetting import (
    EMATeacher,
    EWCSI,
    ReplayBank,
    ReplayItem,
)
from scripts.training.self_improve.orchestrator import Orchestrator, RunReport
from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)
from scripts.training.self_improve.pseudo_labels import (
    EdgeConfig,
    PseudoLabelBatch,
    PseudoLabelBroker,
)
from scripts.training.self_improve.scheduler import (
    BanditScheduler,
    RoundRobinScheduler,
    Scheduler,
)
from scripts.training.self_improve.vault import Snapshot, SnapshotID, Vault

__all__ = [
    "BanditScheduler",
    "ComponentPlugin",
    "DataStream",
    "EMATeacher",
    "EWCSI",
    "EdgeConfig",
    "EvalRegistry",
    "EvalReport",
    "Orchestrator",
    "ProbeSet",
    "PseudoLabelBatch",
    "PseudoLabelBroker",
    "ReplayBank",
    "ReplayItem",
    "RoundRobinScheduler",
    "RunReport",
    "Scheduler",
    "Snapshot",
    "SnapshotID",
    "StepReport",
    "SyntheticDataStream",
    "Vault",
]
