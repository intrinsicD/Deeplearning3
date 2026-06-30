"""W0.2 — manifest data must reach the trainer as dict[str, Tensor]."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from omnilatent.config import OmniLatentConfig
from omnilatent.data import (
    DataManifest,
    MediaDecodeError,
    MultiModalSample,
    SourceSpec,
    StreamingMultiModalDataset,
    build_sample_collator,
)
from omnilatent.data.collate import byte_tokenize, sample_to_inputs
from omnilatent.data.sources.local import _sample_from_path


def _cfg() -> OmniLatentConfig:
    return OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)


def test_sample_to_inputs_text_and_image() -> None:
    cfg = _cfg()
    sample = MultiModalSample(text="hello", image=torch.rand(3, 10, 12))
    out = sample_to_inputs(sample, cfg)
    assert out["text"].dtype == torch.long
    assert out["image"].shape == (3, cfg.image_size, cfg.image_size)


def test_pdf_text_is_tokenized_as_text() -> None:
    cfg = _cfg()
    out = sample_to_inputs(MultiModalSample(pdf_page="pdf body"), cfg)
    assert "text" in out and out["text"].dtype == torch.long


def test_streaming_dataset_collates_to_tensor_dict(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello local text", encoding="utf-8")
    (tmp_path / "b.txt").write_text("another document here", encoding="utf-8")
    manifest = DataManifest([SourceSpec(name="local_text", type="local", path=str(tmp_path))])

    loader = DataLoader(
        StreamingMultiModalDataset(manifest),
        batch_size=2,
        collate_fn=build_sample_collator(_cfg()),
    )
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert "text" in batch
    assert isinstance(batch["text"], torch.Tensor)
    assert batch["text"].shape[0] == 2


def test_mixed_modalities_use_union_not_intersection() -> None:
    cfg = _cfg()
    batch = [
        MultiModalSample(text="words"),
        MultiModalSample(image=torch.rand(3, 8, 8)),
    ]
    out = build_sample_collator(cfg)(batch)
    # Both modalities survive (a text-only and an image-only sample).
    assert "text" in out and out["text"].shape[0] == 1
    assert "image" in out and out["image"].shape[0] == 1


def test_streaming_bool_coercion_false_string() -> None:
    # "false" must parse to False, not bool("false") == True (Audit.md A2).
    spec = SourceSpec.from_dict({"name": "s", "type": "local", "streaming": "false"})
    assert spec.streaming is False
    assert SourceSpec.from_dict({"name": "s", "type": "local", "streaming": "true"}).streaming is True
    assert SourceSpec.from_dict({"name": "s", "type": "local"}).streaming is True


def test_local_audio_video_fail_loud_not_metadata_only(tmp_path: Path) -> None:
    # A bogus media file must raise MediaDecodeError, never yield a
    # metadata-only / empty sample the model cannot learn from (Audit.md A2).
    bad_audio = tmp_path / "x.wav"
    bad_audio.write_bytes(b"not audio")
    with pytest.raises(MediaDecodeError):
        _sample_from_path(bad_audio)

    bad_video = tmp_path / "x.mp4"
    bad_video.write_bytes(b"not video")
    with pytest.raises(MediaDecodeError):
        _sample_from_path(bad_video)


def test_byte_tokenize_reserves_zero_for_padding() -> None:
    ids = byte_tokenize("abc", max_len=8, vocab_size=32000)
    assert ids.min().item() >= 1  # 0 reserved for padding
