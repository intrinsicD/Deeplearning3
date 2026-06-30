"""Local file source adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import torch

from omnilatent.data.errors import MediaDecodeError
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
        return MultiModalSample(audio=_decode_audio_mel(path), metadata={"modality": "audio"})
    if ext in _VIDEO_EXTS:
        return MultiModalSample(video=_decode_video_frames(path), metadata={"modality": "video"})
    return None


# Default mel-spectrogram parameters (match OmniLatentConfig defaults so the
# collate bridge does not have to resample bins).
_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_N_FFT = 1024
_AUDIO_HOP = 256
_AUDIO_N_MELS = 128


def _decode_audio_mel(path: Path) -> torch.Tensor:
    """Decode an audio file into a (n_mels, T) mel spectrogram tensor.

    Raises :class:`MediaDecodeError` rather than returning a metadata-only or
    zero tensor (Audit.md A2/A10): a sample the model cannot learn from must
    not masquerade as valid training data.
    """
    try:
        import torchaudio
        from torchaudio.transforms import MelSpectrogram, Resample
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise MediaDecodeError(
            f"Decoding audio file {path} requires torchaudio "
            "(`pip install 'omnilatent[audio]'`)."
        ) from exc
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        raise MediaDecodeError(f"Failed to decode audio file {path}: {exc}") from exc
    if waveform.numel() == 0:
        raise MediaDecodeError(f"Audio file {path} decoded to an empty waveform")
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sample_rate != _AUDIO_SAMPLE_RATE:
        waveform = Resample(sample_rate, _AUDIO_SAMPLE_RATE)(waveform)
    mel = MelSpectrogram(
        sample_rate=_AUDIO_SAMPLE_RATE,
        n_fft=_AUDIO_N_FFT,
        hop_length=_AUDIO_HOP,
        n_mels=_AUDIO_N_MELS,
    )(waveform)
    mel = mel.squeeze(0)  # (n_mels, T)
    # Round T down to a multiple of 4 (audio encoder stride).
    frames = (mel.shape[1] // 4) * 4
    if frames == 0:
        raise MediaDecodeError(f"Audio file {path} too short to form one frame block")
    return mel[:, :frames].contiguous()


def _decode_video_frames(path: Path) -> torch.Tensor:
    """Decode a video file into a (C, T, H, W) float tensor in [0, 1].

    Raises :class:`MediaDecodeError` on missing decoder or decode failure.
    """
    try:
        from torchvision.io import read_video
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise MediaDecodeError(
            f"Decoding video file {path} requires torchvision "
            "(`pip install 'omnilatent[video]'`)."
        ) from exc
    try:
        # read_video returns (T, H, W, C) uint8.
        video, _audio, _info = read_video(str(path), pts_unit="sec")
    except Exception as exc:
        raise MediaDecodeError(f"Failed to decode video file {path}: {exc}") from exc
    if video.numel() == 0:
        raise MediaDecodeError(f"Video file {path} decoded to zero frames")
    video = video.permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
    return video.contiguous()


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

