"""Self-improvement (continual self-supervised learning) infrastructure.

See ``docs/self_improvement.md`` for the design. This package implements
the full 10-phase rollout plus three follow-up extensions:

- Phases 1–10: plugin scaffolding through control-center integration.
- Follow-up A: orchestrator-integrated plateau-driven capacity
  expansion + per-component compute budget (§4.4, §6.2).
- Follow-up B: fused-DER++ replay across the four "extra-step"
  plugins (omnilatent, mmwm, hpwm; gaussian_encoder was already
  fused), plus EWC wiring extended to gaussian_encoder + hpwm. LGQ
  remains the holdout — its GAN dual-optimizer + AMP step isn't
  easily restructured around an external backward.
- Follow-up C: concrete pseudo-label edge plumbing
  (:mod:`scripts.training.self_improve.edges`), with one wired
  Gaussian → Gaussian image-recon edge demonstrating the broker
  end-to-end through the orchestrator. The remaining §5 edges land
  alongside the dataset adapters.
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
