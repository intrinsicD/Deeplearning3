"""Bug 1 — cross-modal training must pair only genuinely co-occurring rows."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.data import ROW_SUFFIX, collate_multimodal
from omnilatent.training.trainer import Trainer


def _cfg() -> OmniLatentConfig:
    return OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)


class _EmptyDS(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, i):
        return {}


def _trainer() -> Trainer:
    cfg = _cfg()
    return Trainer(OmniLatentModel(cfg), cfg, DataLoader(_EmptyDS()))


def test_collate_records_row_provenance() -> None:
    cfg = _cfg()
    batch = [
        {"text": torch.randint(1, cfg.vocab_size, (5,))},                  # sample 0
        {"image": torch.randn(3, cfg.image_size, cfg.image_size)},         # sample 1
        {"text": torch.randint(1, cfg.vocab_size, (4,)),
         "image": torch.randn(3, cfg.image_size, cfg.image_size)},          # sample 2
    ]
    out = collate_multimodal(batch)
    assert out["text" + ROW_SUFFIX].tolist() == [0, 2]
    assert out["image" + ROW_SUFFIX].tolist() == [1, 2]


def test_align_rows_pairs_only_cooccurring_samples() -> None:
    tr = _trainer()
    cfg = tr.config
    # text rows from samples [0,2]; image rows from samples [1,2]; only sample 2
    # carries both → exactly one genuine cross-modal pair.
    data = {
        "text": torch.randint(1, cfg.vocab_size, (2, 6)),
        "image": torch.randn(2, 3, cfg.image_size, cfg.image_size),
    }
    rows = {"text": torch.tensor([0, 2]), "image": torch.tensor([1, 2])}
    paired = tr._align_rows(data, rows, "text", "image")
    assert paired is not None
    src, tgt = paired
    assert src.shape[0] == 1 and tgt.shape[0] == 1
    # The paired rows are the ones from sample 2 (text row 1, image row 1).
    assert torch.equal(src[0], data["text"][1])
    assert torch.equal(tgt[0], data["image"][1])


def test_unpaired_batch_demotes_to_self_recon_not_unrelated_pairing() -> None:
    tr = _trainer()
    cfg = tr.config
    # text-only and image-only samples: equal size (1) but NO shared sample.
    data = {
        "text": torch.randint(1, cfg.vocab_size, (1, 6)),
        "image": torch.randn(1, 3, cfg.image_size, cfg.image_size),
    }
    rows = {"text": torch.tensor([0]), "image": torch.tensor([1])}
    # Size-only guard would have allowed cross-modal on unrelated rows; provenance
    # correctly reports no genuine pair.
    assert tr._align_rows(data, rows, "text", "image") is None


def test_train_step_on_unpaired_batch_is_safe() -> None:
    tr = _trainer()
    cfg = tr.config
    # Build a real unpaired batch via the collator and run a step: it must not
    # crash and must not be a fake zero-loss step.
    samples = [
        {"text": torch.randint(1, cfg.vocab_size, (6,))},
        {"image": torch.randn(3, cfg.image_size, cfg.image_size)},
    ]
    batch = collate_multimodal(samples)
    import math

    for _ in range(5):
        result = tr._train_step(batch)
        assert "skipped" not in result
        assert math.isfinite(result["total"])


def test_aligned_latents_excludes_unrelated_rows() -> None:
    tr = _trainer()
    cfg = tr.config
    tr.config.contrastive_weight = 0.1
    data = {
        "text": torch.randint(1, cfg.vocab_size, (1, 6)),
        "image": torch.randn(1, 3, cfg.image_size, cfg.image_size),
    }
    rows = {"text": torch.tensor([0]), "image": torch.tensor([1])}
    # No shared sample → fewer than two co-occurring modalities → no contrastive.
    assert tr._aligned_latents(data, rows, "text") is None
