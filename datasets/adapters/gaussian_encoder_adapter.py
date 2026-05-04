"""Gaussian Encoder adapter.

Converts AudioVisualSample → a single image tensor [C, H, W]

The Gaussian Encoder (gaussian_encoder/train.py) was originally trained on
MNIST grayscale images.  This adapter:
  1. Extracts the middle frame of each video clip.
  2. Optionally converts to grayscale.
  3. Resizes to the target resolution.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..base import AudioVisualSample, BaseAVDataset


def adapt_gaussian_encoder(
    sample: AudioVisualSample,
    resolution: int = 28,
    grayscale: bool = True,
) -> torch.Tensor:
    """Return a single [C, H, W] float32 tensor from the middle video frame."""
    T = sample.video_frames.shape[0]
    frame = sample.video_frames[T // 2]   # [3, H, W]

    if grayscale:
        # Standard luma weights
        gray = (0.2989 * frame[0] + 0.5870 * frame[1] + 0.1140 * frame[2]).unsqueeze(0)
        frame = gray   # [1, H, W]

    if frame.shape[1] != resolution or frame.shape[2] != resolution:
        frame = F.interpolate(
            frame.unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return frame.float()   # [C, H, W]


class GaussianEncoderAdapter(Dataset):
    """Wraps a BaseAVDataset; returns single-frame tensors for Gaussian Encoder."""

    def __init__(
        self,
        base_dataset: BaseAVDataset,
        resolution: int = 28,
        grayscale: bool = True,
    ) -> None:
        self.base = base_dataset
        self.resolution = resolution
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.base[idx]
        return adapt_gaussian_encoder(sample, self.resolution, self.grayscale)

