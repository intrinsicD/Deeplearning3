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
from .losses import LossWeights, WorldModelLoss
from .model import ModularLatentWorldModel
from .trainer import Trainer

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
    "Trainer",
    "TransitionOutput",
    "WorldModelLoss",
    "build_model",
    "build_tool_ecosystem",
]
