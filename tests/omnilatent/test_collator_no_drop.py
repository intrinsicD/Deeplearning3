"""W0.3 — collator preserves modalities; trainer never fakes a zero-loss step."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.data import collate_multimodal
from omnilatent.training.trainer import Trainer


def _cfg() -> OmniLatentConfig:
    return OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)


class _EmptyDS(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, i):
        return {}


def test_collate_union_preserves_all_modalities() -> None:
    cfg = _cfg()
    # Three single-modality samples — intersection would be empty.
    batch = [
        {"text": torch.randint(1, cfg.vocab_size, (5,))},
        {"image": torch.randn(3, cfg.image_size, cfg.image_size)},
        {"text": torch.randint(1, cfg.vocab_size, (7,))},
    ]
    out = collate_multimodal(batch)
    assert set(out.keys()) == {"text", "image"}  # nothing dropped
    assert out["text"].shape[0] == 2  # two text samples
    assert out["image"].shape[0] == 1  # one image sample


def test_empty_batch_is_skipped_not_zero_loss() -> None:
    cfg = _cfg()
    model = OmniLatentModel(cfg)
    tr = Trainer(model, cfg, DataLoader(_EmptyDS()))
    result = tr._train_step({})
    assert result.get("skipped") == 1.0
    assert "total" not in result  # never a fake zero-loss


def test_mismatched_crossmodal_demotes_to_self_recon() -> None:
    cfg = _cfg()
    model = OmniLatentModel(cfg)
    tr = Trainer(model, cfg, DataLoader(_EmptyDS()))
    # text B=2, image B=1 — any cross-modal pick must demote to self-recon
    # and still yield a real (finite, non-skipped) loss every time.
    for _ in range(8):
        batch = {
            "text": torch.randint(1, cfg.vocab_size, (2, 6)),
            "image": torch.randn(1, 3, cfg.image_size, cfg.image_size),
        }
        result = tr._train_step(batch)
        assert "skipped" not in result
        assert "total" in result
        assert math.isfinite(result["total"])
