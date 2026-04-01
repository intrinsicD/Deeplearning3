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
from .curriculum import CurriculumPhase, default_curriculum_phases
from .losses import LossWeights, WorldModelLoss
from .model import ModularLatentWorldModel

__all__ = [
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
    "TransitionOutput",
    "WorldModelLoss",
    "build_model",
    "build_tool_ecosystem",
    "default_curriculum_phases",
]
