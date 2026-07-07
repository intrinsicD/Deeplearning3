"""Multimodal Latent World Model — modular research package.

Usage:
    from MMWM.config import ModelConfig, build_model
    from MMWM.losses import WorldModelLoss
    from MMWM.trainer import Trainer
"""

from .containers import (
    LatentPacket,
    LatentState,
    MemoryState,
    ModelOutput,
    ObservationPacket,
    ToolContext,
    ToolDescriptor,
    ToolResult,
    TransitionOutput,
)
from .config import ModelConfig, build_model, build_tool_ecosystem
from .curriculum import (
    AdaptiveCurriculumScheduler,
    CurriculumPhase,
    default_curriculum_phases,
    relative_curriculum_phases,
)
from .data import (
    D4RLTransitionDataset,
    DeterministicTransitionDataset,
    MinariTransitionDataset,
    TransitionTupleDataset,
    collate_transition_batch,
    flatten_transition_value,
)
from .evaluation import EvalResult, EvaluationSuite
from .losses import LossWeights, WorldModelLoss
from .model import ModularLatentWorldModel
from .trainer import build_lr_scheduler

__all__ = [
    "AdaptiveCurriculumScheduler",
    "EvalResult",
    "EvaluationSuite",
    "LatentPacket",
    "LatentState",
    "LossWeights",
    "MemoryState",
    "ModelConfig",
    "ModelOutput",
    "ModularLatentWorldModel",
    "ObservationPacket",
    "ToolContext",
    "ToolDescriptor",
    "ToolResult",
    "CurriculumPhase",
    "D4RLTransitionDataset",
    "DeterministicTransitionDataset",
    "MinariTransitionDataset",
    "TransitionOutput",
    "TransitionTupleDataset",
    "WorldModelLoss",
    "build_lr_scheduler",
    "build_model",
    "collate_transition_batch",
    "flatten_transition_value",
    "build_tool_ecosystem",
    "default_curriculum_phases",
    "relative_curriculum_phases",
]
