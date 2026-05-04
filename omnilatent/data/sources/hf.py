"""Hugging Face datasets source adapter."""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import torch

from omnilatent.data.manifest import SourceSpec
from omnilatent.data.sample import MultiModalSample


def iter_hf_stream(source: SourceSpec) -> Iterator[MultiModalSample]:
    if source.dataset is None:
        raise ValueError("HF source requires dataset")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("HF streaming requires `datasets`. Install with `pip install datasets`.") from exc

    kwargs = dict(source.options)
    split = source.split or kwargs.pop("split", "train")
    ds = load_dataset(source.dataset, split=split, streaming=source.streaming, **kwargs)
    for row in ds:
        if isinstance(row, Mapping):
            yield _row_to_sample(row, source)


def _row_to_sample(row: Mapping[str, Any], source: SourceSpec) -> MultiModalSample:
    text_keys = source.options.get("text_keys", ["text", "caption", "content", "sentence"])
    image_keys = source.options.get("image_keys", ["image", "jpg", "png"])
    vector_keys = source.options.get("vector_keys", ["vector", "embedding", "observation"])

    text = next((str(row[k]) for k in text_keys if k in row and row[k] is not None), None)
    image = None
    for key in image_keys:
        if key in row and row[key] is not None:
            image = _maybe_image_tensor(row[key])
            break
    vector = None
    for key in vector_keys:
        if key in row and row[key] is not None:
            try:
                vector = torch.as_tensor(row[key], dtype=torch.float32)
            except (TypeError, ValueError):
                vector = None
            break
    metadata = {"dataset": source.dataset, "split": source.split}
    for key in ("id", "url", "license"):
        if key in row:
            metadata[key] = row[key]
    return MultiModalSample(text=text, image=image, vector=vector, metadata=metadata)


def _maybe_image_tensor(value: Any) -> torch.Tensor | None:
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return None


__all__ = ["iter_hf_stream"]

