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


#: Suffix marking a per-modality row-provenance tensor in a collated batch.
#: ``batch[f"{mod}{ROW_SUFFIX}"]`` is a LongTensor of the original sample index
#: each stacked row of ``batch[mod]`` came from. Consumers that iterate
#: modalities must skip these keys (see ``Trainer._train_step``).
ROW_SUFFIX = "__rows"


def collate_multimodal(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate a batch of multi-modal samples.

    Uses **union** semantics: every modality present in *any* sample survives,
    stacked over just the samples that carry it. The previous intersection
    semantics silently dropped a modality whenever a single sample lacked it,
    which on mixed real batches could empty the batch entirely and produce a
    fake zero-loss step (Audit.md A3).

    Consequence: per-modality batch sizes may differ. Because the stacks are
    built independently, row *i* of one modality need not come from the same
    original sample as row *i* of another. To let cross-modal training pair
    only genuinely co-occurring rows, each modality also gets a
    ``f"{mod}{ROW_SUFFIX}"`` LongTensor recording the original sample index of
    every stacked row (``Trainer._train_step`` uses it to align pairs).

    Pads text and audio to the per-modality maximum length; images and videos
    are stacked directly (fixed spatial size).
    """
    result: dict[str, torch.Tensor] = {}

    # Union of modalities present across the batch.
    present: set[str] = set()
    for sample in batch:
        present |= set(sample.keys())

    def _rows(mod: str) -> torch.Tensor:
        return torch.tensor([i for i, s in enumerate(batch) if mod in s], dtype=torch.long)

    if "text" in present:
        texts = [s["text"] for s in batch if "text" in s]
        max_len = max(t.shape[0] for t in texts)
        padded = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, t in enumerate(texts):
            padded[i, : t.shape[0]] = t
        result["text"] = padded
        result["text" + ROW_SUFFIX] = _rows("text")

    if "audio" in present:
        audios = [s["audio"] for s in batch if "audio" in s]
        max_frames = max(a.shape[1] for a in audios)
        n_mels = audios[0].shape[0]
        padded = torch.zeros(len(audios), n_mels, max_frames)
        for i, a in enumerate(audios):
            padded[i, :, : a.shape[1]] = a
        result["audio"] = padded
        result["audio" + ROW_SUFFIX] = _rows("audio")

    if "image" in present:
        result["image"] = torch.stack([s["image"] for s in batch if "image" in s])
        result["image" + ROW_SUFFIX] = _rows("image")

    if "video" in present:
        result["video"] = torch.stack([s["video"] for s in batch if "video" in s])
        result["video" + ROW_SUFFIX] = _rows("video")

    return result


def build_dataloader(
    config: OmniLatentConfig,
    dataset: Dataset | None = None,
    num_workers: int | None = None,
    **kwargs,
) -> DataLoader:
    """Build a DataLoader with proper collation."""
    if dataset is None:
        dataset = SyntheticMultiModalDataset(config)

    import os

    if num_workers is None:
        env_workers = os.getenv("OMNILATENT_NUM_WORKERS")
        if env_workers is not None:
            try:
                num_workers = int(env_workers)
            except ValueError as exc:
                raise ValueError(
                    "OMNILATENT_NUM_WORKERS must be an integer, "
                    f"got {env_workers!r}"
                ) from exc
        else:
            num_workers = min(8, os.cpu_count() or 1)
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_multimodal,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **kwargs,
    )
