"""Local file source adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import torch

from omnilatent.data.manifest import SourceSpec
from omnilatent.data.sample import MultiModalSample

_TEXT_EXTS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_PDF_EXTS = {".pdf"}
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_local_files(source: SourceSpec) -> Iterator[MultiModalSample]:
    if source.path is None:
        raise ValueError("local source requires path")
    root = Path(source.path)
    if not root.exists():
        raise FileNotFoundError(root)
    recursive = bool(source.options.get("recursive", True))
    files = [root] if root.is_file() else _iter_files(root, recursive=recursive)
    for path in files:
        sample = _sample_from_path(path)
        if sample is not None:
            yield sample.with_metadata(path=str(path), filename=path.name)


def _iter_files(root: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    return (p for p in sorted(root.glob(pattern)) if p.is_file())


def _sample_from_path(path: Path) -> MultiModalSample | None:
    ext = path.suffix.lower()
    if ext in _TEXT_EXTS:
        return MultiModalSample(text=path.read_text(encoding="utf-8", errors="replace"), metadata={"modality": "text"})
    if ext in _PDF_EXTS:
        return MultiModalSample(pdf_page=_read_pdf_text_fallback(path), metadata={"modality": "pdf"})
    if ext in _IMAGE_EXTS:
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise ImportError("Local image loading requires Pillow and numpy") from exc
        image = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return MultiModalSample(image=arr, metadata={"modality": "image"})
    if ext in _AUDIO_EXTS:
        return MultiModalSample(metadata={"modality": "audio", "path": str(path)})
    if ext in _VIDEO_EXTS:
        return MultiModalSample(metadata={"modality": "video", "path": str(path)})
    return None


def _read_pdf_text_fallback(path: Path) -> str:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return path.read_bytes()[:4096].decode("latin-1", errors="replace")
    doc = fitz.open(str(path))
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


__all__ = ["iter_local_files"]

