"""HPWM adapter.

Converts AudioVisualSample → `{"frames": [T, 3, H, W], "frames_raw": ..., "clip_id": int}`

Matches the schema emitted by hpwm/data.py SyntheticMovingShapes / SSv2Dataset.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..base import AudioVisualSample, BaseAVDataset

# ImageNet normalisation constants (same as hpwm/data.py)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def adapt_hpwm(
    sample: AudioVisualSample,
    resolution: int = 128,
    n_frames: int = 120,
    clip_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Convert one AudioVisualSample to an HPWM batch item."""
    frames = sample.video_frames   # [T, 3, H, W]

    # Resize spatial dims if needed
    H, W = frames.shape[2], frames.shape[3]
    if H != resolution or W != resolution:
        frames = F.interpolate(
            frames,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )

    # Temporal re-sample to exact n_frames
    T = frames.shape[0]
    if T >= n_frames:
        idx = torch.linspace(0, T - 1, n_frames).long()
        frames = frames[idx]
    else:
        pad = n_frames - T
        frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)

    frames_raw = frames.clone()

    # ImageNet normalisation
    mean = _IMAGENET_MEAN
    std  = _IMAGENET_STD
    frames_norm = (frames - mean) / std

    return {
        "frames":     frames_norm.float(),    # [T, 3, H, W] normalised
        "frames_raw": frames_raw.float(),     # [T, 3, H, W] [0,1]
        "clip_id":    torch.tensor(clip_id, dtype=torch.long),
    }


class HPWMAdapter(Dataset):
    """Wraps a BaseAVDataset and converts each sample to an HPWM batch dict."""

    def __init__(
        self,
        base_dataset: BaseAVDataset,
        resolution: int = 128,
        n_frames: int = 120,
    ) -> None:
        self.base = base_dataset
        self.resolution = resolution
        self.n_frames = n_frames

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        return adapt_hpwm(sample, self.resolution, self.n_frames, clip_id=idx)

