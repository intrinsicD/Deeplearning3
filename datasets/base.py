"""Base types shared by all dataset downloaders."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Canonical sample type
# ---------------------------------------------------------------------------

@dataclass
class AudioVisualSample:
    """Canonical output for every dataset.

    Shapes:
        video_frames : [T, 3, H, W]  float32 in [0, 1]
        audio        : [C, L]        float32 waveform (16 kHz default)
        text_caption : str
        metadata     : free-form dict (label, clip_id, fps, sr, …)
    """
    video_frames: torch.Tensor          # [T, 3, H, W]
    audio: torch.Tensor                 # [C, L]
    text_caption: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base dataset
# ---------------------------------------------------------------------------

class BaseAVDataset(ABC, Dataset):
    """Abstract base for all public AV datasets.

    Sub-classes must implement:
        download(data_dir)
        verify(data_dir) -> bool
        _build_index(data_dir, split) -> list of sample descriptors
        _load_sample(descriptor) -> AudioVisualSample
    """

    # Canonical resolutions – adapters use these as defaults
    DEFAULT_N_FRAMES: int = 16
    DEFAULT_RESOLUTION: int = 224
    DEFAULT_AUDIO_SR: int = 16_000
    DEFAULT_AUDIO_DURATION_S: float = 4.0

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        n_frames: int = DEFAULT_N_FRAMES,
        resolution: int = DEFAULT_RESOLUTION,
        audio_sr: int = DEFAULT_AUDIO_SR,
        audio_duration_s: float = DEFAULT_AUDIO_DURATION_S,
        max_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.n_frames = n_frames
        self.resolution = resolution
        self.audio_sr = audio_sr
        self.audio_duration_s = audio_duration_s
        self.max_samples = max_samples
        self._index: List[Any] = []

    # ------------------------------------------------------------------
    # Must override
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def download(cls, data_dir: str | Path, **kwargs: Any) -> None:
        """Download and extract the dataset into *data_dir*."""

    @classmethod
    @abstractmethod
    def verify(cls, data_dir: str | Path) -> bool:
        """Return True if the dataset is fully present in *data_dir*."""

    @abstractmethod
    def _build_index(self) -> List[Any]:
        """Return a list of lightweight sample descriptors (paths, ids, …)."""

    @abstractmethod
    def _load_sample(self, descriptor: Any) -> AudioVisualSample:
        """Load one sample from its descriptor."""

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def prepare(self) -> "BaseAVDataset":
        """Build the index.  Call this after construction (or after download)."""
        self._index = self._build_index()
        if self.max_samples:
            self._index = self._index[: self.max_samples]
        return self

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> AudioVisualSample:
        return self._load_sample(self._index[idx])

    # ------------------------------------------------------------------
    # Shared video / audio helpers
    # ------------------------------------------------------------------

    def _load_video_frames(self, path: Path) -> torch.Tensor:
        """Load a video, return [T, 3, H, W] float32 in [0,1]."""
        try:
            import torchvision.io as tvio
            video, _, info = tvio.read_video(str(path), pts_unit="sec")
            # [T, H, W, 3] uint8  →  [T, 3, H, W] float32
            video = video.float() / 255.0
            video = video.permute(0, 3, 1, 2)
        except Exception:
            return torch.zeros(self.n_frames, 3, self.resolution, self.resolution)

        # Resize spatial dims
        if video.shape[2] != self.resolution or video.shape[3] != self.resolution:
            video = torch.nn.functional.interpolate(
                video,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )

        # Temporal sub-sample / pad to exactly n_frames
        T = video.shape[0]
        if T >= self.n_frames:
            idx = torch.linspace(0, T - 1, self.n_frames).long()
            video = video[idx]
        else:
            pad = self.n_frames - T
            video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)

        return video  # [T, 3, H, W]

    def _load_audio_wav(self, path: Path) -> torch.Tensor:
        """Load audio, return [C, L] float32 at self.audio_sr."""
        target_len = int(self.audio_sr * self.audio_duration_s)
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(path))
            if sr != self.audio_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.audio_sr)
        except Exception:
            return torch.zeros(1, target_len)

        # Trim / pad to exact length
        L = waveform.shape[1]
        if L >= target_len:
            waveform = waveform[:, :target_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - L))

        return waveform  # [C, L]

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _http_download(url: str, dest: Path, desc: str = "") -> None:
        """Simple progress-printing HTTP downloader."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        label = desc or dest.name
        print(f"[download] {label}  →  {dest}")
        urllib.request.urlretrieve(url, dest, reporthook=BaseAVDataset._reporthook)
        print()

    @staticmethod
    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, int(count * block_size * 100 / total_size))
            print(f"\r  {pct}%", end="", flush=True)

    @staticmethod
    def _extract_archive(archive: Path, dest: Path) -> None:
        print(f"[extract] {archive.name}  →  {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive), str(dest))

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

