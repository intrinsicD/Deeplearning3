"""WebDataset/tar-shard source adapter.

This adapter intentionally supports plain local tar shards without requiring the
external ``webdataset`` package, which keeps smoke tests and small local shards
lightweight. It can be replaced by a native WebDataset pipeline later.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterator

import torch

from omnilatent.data.manifest import SourceSpec
from omnilatent.data.sample import MultiModalSample

_TEXT_EXTS = {".txt", ".text", ".caption", ".md"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_TENSOR_EXTS = {".pt", ".pth"}
_JSON_EXTS = {".json"}


def iter_webdataset_shards(source: SourceSpec) -> Iterator[MultiModalSample]:
    if source.path is None:
        raise ValueError("webdataset source requires path")
    path = Path(source.path)
    shards = sorted(path.glob("*.tar")) if path.is_dir() else [path]
    for shard in shards:
        yield from _iter_tar_shard(shard)


def _iter_tar_shard(path: Path) -> Iterator[MultiModalSample]:
    grouped: Dict[str, Dict[str, bytes]] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            key = Path(member.name).with_suffix("").as_posix()
            grouped.setdefault(key, {})[Path(member.name).suffix.lower()] = fileobj.read()
    for key, parts in grouped.items():
        sample = _parts_to_sample(key, parts)
        if sample.has_any_modality():
            yield sample.with_metadata(shard=str(path), key=key)


def _parts_to_sample(key: str, parts: Dict[str, bytes]) -> MultiModalSample:
    text = None
    image = None
    vector = None
    metadata: Dict[str, Any] = {}
    metadata["key"] = key
    for ext, payload in parts.items():
        if ext in _TEXT_EXTS:
            text = payload.decode("utf-8", errors="replace")
        elif ext in _JSON_EXTS:
            try:
                decoded = json.loads(payload.decode("utf-8"))
                if isinstance(decoded, dict):
                    for key, value in decoded.items():
                        metadata[str(key)] = value
            except json.JSONDecodeError:
                metadata["json_error"] = "true"
        elif ext in _IMAGE_EXTS:
            image = _decode_image(payload)
        elif ext in _TENSOR_EXTS:
            vector = torch.load(io.BytesIO(payload), map_location="cpu")
            if not isinstance(vector, torch.Tensor):
                vector = torch.as_tensor(vector, dtype=torch.float32)
    return MultiModalSample(text=text, image=image, vector=vector, metadata=metadata)


def _decode_image(payload: bytes) -> torch.Tensor | None:
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


__all__ = ["iter_webdataset_shards"]

