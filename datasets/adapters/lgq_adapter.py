"""LGQ adapter.

Converts AudioVisualSample → a single image tensor [3, H, W]

Matches lgq/train.py SyntheticImageDataset.__getitem__()  which returns
a raw float32 tensor in [0,1].  Extracts every frame as a separate sample
to maximise training data from video clips.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..base import AudioVisualSample, BaseAVDataset


def adapt_lgq(
    sample: AudioVisualSample,
    resolution: int = 256,
) -> List[torch.Tensor]:
    """Return one [3, H, W] tensor per video frame."""
    frames = sample.video_frames   # [T, 3, H, W]

    H, W = frames.shape[2], frames.shape[3]
    if H != resolution or W != resolution:
        frames = F.interpolate(
            frames,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )

    return [frames[t].float() for t in range(frames.shape[0])]


class LGQAdapter(Dataset):
    """Wraps a BaseAVDataset; each video frame becomes a training image.

    The dataset returns a single [3, H, W] float32 tensor in [0,1],
    compatible with lgq.train.SyntheticImageDataset.
    """

    def __init__(
        self,
        base_dataset: BaseAVDataset,
        resolution: int = 256,
    ) -> None:
        self.base = base_dataset
        self.resolution = resolution
        # Pre-build flat index: (clip_idx, frame_idx)
        self._flat_index: List[tuple[int, int]] = []
        for clip_idx in range(len(base_dataset)):
            sample = base_dataset[clip_idx]
            n_frames = sample.video_frames.shape[0]
            self._flat_index.extend((clip_idx, f) for f in range(n_frames))

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        clip_idx, frame_idx = self._flat_index[idx]
        sample = self.base[clip_idx]
        frames = sample.video_frames   # [T, 3, H, W]

        H, W = frames.shape[2], frames.shape[3]
        if H != self.resolution or W != self.resolution:
            frames = F.interpolate(
                frames,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )

        return frames[frame_idx].float()   # [3, H, W]

