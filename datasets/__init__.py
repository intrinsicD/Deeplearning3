"""Unified public-dataset infrastructure for all models in this workspace.

Usage:
    from datasets import build_av_dataset, list_datasets

    ds = build_av_dataset("vggsound", data_dir="./data/vggsound", split="train")
    ds.download()
    sample = ds[0]   # AudioVisualSample
"""

from .registry import build_av_dataset, list_datasets, DATASET_REGISTRY, register_dataset
from .base import AudioVisualSample, BaseAVDataset

# Auto-import all downloaders so they self-register
from . import downloaders  # noqa: F401

__all__ = [
    "build_av_dataset",
    "list_datasets",
    "DATASET_REGISTRY",
    "register_dataset",
    "AudioVisualSample",
    "BaseAVDataset",
]
