"""Utility helpers for OmniLatent."""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch

Modality = Literal["text", "audio", "image", "video"]

ALL_MODALITIES: list[Modality] = ["text", "audio", "image", "video"]

# Integer IDs used as special leading token per modality
MODALITY_ID: dict[str, int] = {
    "text": 0,
    "audio": 1,
    "image": 2,
    "video": 3,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_size_mb(model: torch.nn.Module) -> float:
    """Approximate size of parameters in MB (FP32)."""
    return count_parameters(model) * 4 / (1024 ** 2)
