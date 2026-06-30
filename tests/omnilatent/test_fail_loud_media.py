"""W0.4 — bad media must raise MediaDecodeError, never return zeros (A10)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.data.errors import MediaDecodeError


def test_coco_corrupt_image_raises() -> None:
    from omnilatent.training.coco_dataset import CocoCaptionsDataset

    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    # Bypass __init__ (needs torchvision/PIL + annotation files) and exercise
    # only the decode path of __getitem__.
    ds = object.__new__(CocoCaptionsDataset)
    ds.config = cfg
    ds.tokenizer_fn = None
    ds.transform = lambda img: img  # never reached — open() fails first
    ds.samples = [(Path("/nonexistent/does_not_exist.jpg"), "a real caption")]

    with pytest.raises(MediaDecodeError):
        ds[0]


def test_mmwm_audio_to_mel_fails_loud() -> None:
    from datasets.adapters.mmwm_adapter import _audio_to_mel

    # torchaudio is an optional dep; whether it is missing (ImportError) or the
    # waveform is undecodable, the function must raise MediaDecodeError rather
    # than return a zero spectrogram.
    with pytest.raises(MediaDecodeError):
        _audio_to_mel(torch.zeros(1, 0))  # empty waveform / missing dep


def test_video_watching_getitem_raises_on_empty(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    from omnilatent.training.video_dataset import VideoWatchingDataset

    cfg = OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)
    ds = object.__new__(VideoWatchingDataset)
    ds.config = cfg
    ds.index = [(Path("/nonexistent.mp4"), 0.0, None)]
    ds.tokenizer_fn = None
    # _load_clip on a missing file returns (None, None, {}) → empty sample →
    # must raise rather than fabricate a zero audio tensor.
    ds._load_clip = lambda path, start: (None, None, {})

    with pytest.raises(MediaDecodeError):
        ds[0]
