"""Data layer for OmniLatent.

Organized as:
  - datasets/   — per-domain dataset implementations
  - transforms/ — tokenization, resizing, patching
  - collate/    — per-modality collate functions
  - registry    — build dataset from config
"""

from omnilatent.data.registry import build_dataset

__all__ = ["build_dataset"]
