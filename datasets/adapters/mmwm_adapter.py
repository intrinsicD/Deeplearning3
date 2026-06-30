"""MMWM adapter.

Converts AudioVisualSample → the batch dict expected by
MMWM/trainer.py Trainer and MMWM/data.py DeterministicTransitionDataset.

Key schema (matching data.py):
    vector_t / vector_tp1 / vector_target  : [vector_dim]
    action                                 : [action_dim]
    image_t / image_tp1 / image_target     : [3, H, W]
    audio_t / audio_tp1 / audio_target     : [C, L]
    text_t  / text_tp1  / text_target      : [text_len]  (token ids)
    prefix_tokens                          : [text_len]
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omnilatent.data.errors import MediaDecodeError

from ..base import AudioVisualSample, BaseAVDataset


# Simple character-level tokenizer (no external dependency)
def _tokenize(text: str, seq_len: int, vocab_size: int = 256) -> torch.Tensor:
    ids = [min(ord(c), vocab_size - 1) for c in text[:seq_len]]
    ids += [0] * (seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def _audio_to_mel(waveform: torch.Tensor, n_mels: int = 80, n_fft: int = 400,
                  hop_length: int = 160, sr: int = 16_000) -> torch.Tensor:
    """Return log-mel spectrogram [1, n_mels, T].

    Fails loud rather than returning a zero spectrogram, which would feed the
    world model a corrupted audio target while keeping the loss finite
    (Audit.md A10).
    """
    try:
        import torchaudio.transforms as T
    except ImportError as exc:
        raise MediaDecodeError(
            "Computing the mel spectrogram requires torchaudio "
            "(`pip install torchaudio`)."
        ) from exc
    try:
        mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )(waveform.mean(0, keepdim=True))
        return torch.log(mel.clamp(min=1e-5))
    except Exception as exc:
        raise MediaDecodeError(f"Failed to compute mel spectrogram: {exc}") from exc


def adapt_mmwm(
    sample: AudioVisualSample,
    vector_dim: int = 16,
    action_dim: int = 8,
    text_len: int = 8,
    vocab_size: int = 256,
    image_size: int = 64,
    audio_mel_bins: int = 80,
    audio_mel_frames: int = 128,
    include_text: bool = True,
    include_image: bool = True,
    include_audio: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert one AudioVisualSample to an MMWM batch item."""
    # Use first frame as t, second frame as tp1 (or repeat if only 1 frame)
    T = sample.video_frames.shape[0]
    frame_t   = sample.video_frames[0]          # [3, H, W]
    frame_tp1 = sample.video_frames[min(1, T-1)]

    # Resize image
    def _resize(img: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            img.unsqueeze(0), size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

    frame_t   = _resize(frame_t)
    frame_tp1 = _resize(frame_tp1)

    # Vector = mean-pooled per-channel statistics from video frames
    flat = sample.video_frames.mean(dim=[2, 3])  # [T, 3]
    vec_t   = flat[0].repeat((vector_dim + 2) // 3)[:vector_dim]
    vec_tp1 = flat[min(1, T-1)].repeat((vector_dim + 2) // 3)[:vector_dim]
    action  = torch.zeros(action_dim)

    item: Dict[str, torch.Tensor] = {
        "vector_t":      vec_t.float(),
        "vector_tp1":    vec_tp1.float(),
        "vector_target": vec_tp1.float(),
        "action":        action.float(),
    }

    if include_text:
        text_t   = _tokenize(sample.text_caption, text_len, vocab_size)
        text_tp1 = _tokenize(sample.text_caption, text_len, vocab_size)
        item.update({
            "text_t":        text_t,
            "text_tp1":      text_tp1,
            "text_target":   text_tp1,
            "prefix_tokens": torch.cat([torch.zeros(1, dtype=torch.long), text_tp1[:-1]]),
        })

    if include_image:
        item.update({
            "image_t":      frame_t.float(),
            "image_tp1":    frame_tp1.float(),
            "image_target": frame_tp1.float(),
        })

    if include_audio:
        mel_t   = _audio_to_mel(sample.audio)          # [1, n_mels, T_mel]
        mel_t   = F.interpolate(
            mel_t.unsqueeze(0), size=(audio_mel_bins, audio_mel_frames),
            mode="bilinear", align_corners=False,
        ).squeeze(0)                                   # [1, n_mels, audio_mel_frames]
        item.update({
            "audio_t":      mel_t.float(),
            "audio_tp1":    mel_t.float(),
            "audio_target": mel_t.float(),
        })

    return item


class MMWMAdapter(Dataset):
    """Wraps a BaseAVDataset and converts each sample to an MMWM batch dict."""

    def __init__(self, base_dataset: BaseAVDataset, **kwargs: Any) -> None:
        self.base = base_dataset
        self.kwargs = kwargs

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        return adapt_mmwm(sample, **self.kwargs)

