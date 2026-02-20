"""Synthetic data generators and dataset utilities.

Provides:
  * SyntheticMultiModalDataset -- generates random tensors in the correct
    shapes for each modality.  Useful for testing, debugging, and verifying
    that the full pipeline trains without errors before plugging in real
    data.
  * Helper functions for building DataLoaders with proper collation.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from omnilatent.config import OmniLatentConfig
from omnilatent.utils import ALL_MODALITIES, Modality


class SyntheticMultiModalDataset(Dataset):
    """Generates synthetic data for all modalities.

    Each sample is a dict with a random subset of modalities.  This is
    useful for:
      * Verifying gradient flow and trainability
      * Debugging the full pipeline
      * Benchmarking memory and speed

    In real usage, replace this with your actual dataset.
    """

    def __init__(
        self,
        config: OmniLatentConfig,
        length: int = 10_000,
        modalities: Sequence[Modality] | None = None,
        paired: bool = True,
    ) -> None:
        self.config = config
        self.length = length
        self.modalities = list(modalities or ALL_MODALITIES)
        self.paired = paired  # if True, every sample has ALL modalities

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        c = self.config
        sample: dict[str, torch.Tensor] = {}

        if self.paired:
            active = self.modalities
        else:
            # Random subset of modalities (at least 1)
            k = random.randint(1, len(self.modalities))
            active = random.sample(self.modalities, k)

        if "text" in active:
            seq_len = random.randint(16, c.text_max_len)
            sample["text"] = torch.randint(1, c.vocab_size, (seq_len,))

        if "audio" in active:
            n_frames = random.randint(64, c.audio_max_frames)
            # Make n_frames divisible by 4 (encoder stride)
            n_frames = (n_frames // 4) * 4
            sample["audio"] = torch.randn(c.audio_n_mels, n_frames)

        if "image" in active:
            sample["image"] = torch.randn(
                c.image_channels, c.image_size, c.image_size
            )

        if "video" in active:
            sample["video"] = torch.randn(
                c.video_channels,
                c.video_max_frames,
                c.video_size,
                c.video_size,
            )

        return sample


def collate_multimodal(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a batch of multi-modal samples.

    Pads text and audio to the maximum length in the batch.
    Images and videos are stacked directly (fixed spatial size).
    """
    result: dict[str, torch.Tensor] = {}

    # Find which modalities are present in ALL samples of this batch
    common_modalities = set(batch[0].keys())
    for sample in batch[1:]:
        common_modalities &= set(sample.keys())

    if "text" in common_modalities:
        max_len = max(s["text"].shape[0] for s in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, s in enumerate(batch):
            padded[i, : s["text"].shape[0]] = s["text"]
        result["text"] = padded

    if "audio" in common_modalities:
        max_frames = max(s["audio"].shape[1] for s in batch)
        n_mels = batch[0]["audio"].shape[0]
        padded = torch.zeros(len(batch), n_mels, max_frames)
        for i, s in enumerate(batch):
            padded[i, :, : s["audio"].shape[1]] = s["audio"]
        result["audio"] = padded

    if "image" in common_modalities:
        result["image"] = torch.stack([s["image"] for s in batch])

    if "video" in common_modalities:
        result["video"] = torch.stack([s["video"] for s in batch])

    return result


def build_dataloader(
    config: OmniLatentConfig,
    dataset: Dataset | None = None,
    **kwargs,
) -> DataLoader:
    """Build a DataLoader with proper collation."""
    if dataset is None:
        dataset = SyntheticMultiModalDataset(config)

    # --- ADDED: Calculate optimal workers based on CPU ---
    import os
    optimal_workers = min(8, os.cpu_count() or 1)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=optimal_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **kwargs,
    )
