"""Self-improvement (continual self-supervised learning) infrastructure.

See ``docs/self_improvement.md`` for the design. This package implements
the full 10-phase rollout plus follow-up extensions:

- Phases 1–10: plugin scaffolding through control-center integration.
- Follow-up A: orchestrator-integrated plateau-driven capacity
  expansion + per-component compute budget (§4.4, §6.2).
- Follow-up B: fused-DER++ replay across **all five plugins** (LGQ
  joined the fused set after its GAN dual-optimizer step was
  restructured to expose the backward), plus EWC wiring extended to
  every plugin.
- Follow-up C: concrete pseudo-label edge plumbing
  (:mod:`scripts.training.self_improve.edges`), with one wired
  Gaussian → Gaussian image-recon edge demonstrating the broker
  end-to-end through the orchestrator. The remaining §5 edges land
  alongside the dataset adapters.
- Follow-up D: pseudo-label confidence-threshold calibration script
  (:mod:`scripts.training.self_improve.calibrate_pseudo_labels`,
  §10.4) — sweeps thresholds against held-out consumer probes and
  writes a JSON report with recommended per-edge values.
"""

from scripts.training.self_improve.data_stream import DataStream, SyntheticDataStream
from scripts.training.self_improve.edges import (
    PluginPseudoLabelConsumer,
    apply_pending_labels,
    gaussian_recon_label_fn,
    image_consistency_loss,
)
from scripts.training.self_improve.eval_registry import EvalRegistry, ProbeSet
from scripts.training.self_improve.forgetting import (
    EMATeacher,
    EWCSI,
    PlateauDetector,
    ReplayBank,
    ReplayItem,
    expand_omnilatent_capacity,
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
    "PlateauDetector",
    "PluginPseudoLabelConsumer",
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
    "apply_pending_labels",
    "expand_omnilatent_capacity",
    "gaussian_recon_label_fn",
    "image_consistency_loss",
]
