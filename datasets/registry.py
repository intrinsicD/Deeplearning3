"""Dataset registry – maps name → BaseAVDataset subclass."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import BaseAVDataset

DATASET_REGISTRY: Dict[str, Type[BaseAVDataset]] = {}


def register_dataset(name: str):
    """Class decorator that adds the class to DATASET_REGISTRY."""
    def _inner(cls: Type[BaseAVDataset]) -> Type[BaseAVDataset]:
        DATASET_REGISTRY[name] = cls
        return cls
    return _inner


def build_av_dataset(
    name: str,
    data_dir: str,
    split: str = "train",
    auto_prepare: bool = True,
    **kwargs: Any,
) -> BaseAVDataset:
    """Instantiate a registered dataset.

    Args:
        name        : registered name, e.g. "vggsound", "kinetics400"
        data_dir    : root directory where data lives / will be downloaded
        split       : "train" | "val" | "test"
        auto_prepare: call .prepare() automatically (builds index)
        **kwargs    : forwarded to the dataset constructor

    Returns:
        A ready-to-use BaseAVDataset instance.

    Raises:
        KeyError if the name is not registered.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")

    cls = DATASET_REGISTRY[name]
    ds = cls(data_dir=data_dir, split=split, **kwargs)
    if auto_prepare:
        ds.prepare()
    return ds


def list_datasets() -> List[str]:
    """Return a sorted list of all registered dataset names."""
    return sorted(DATASET_REGISTRY.keys())

