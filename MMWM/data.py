"""Minimal deterministic datasets for MMWM smoke training.

These utilities intentionally avoid external environment dependencies. They produce
trainer-compatible batch dictionaries for quick integration tests and loss-decrease
checks before connecting real RL/video/text datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class DeterministicTransitionDataset(Dataset):
    """Synthetic but learnable transition tuples.

    Each item follows a fixed linear vector dynamics rule and can optionally add
    aligned text, image, and audio observations/targets. Randomness is keyed by
    the sample index, so data is deterministic across workers and epochs.
    """

    def __init__(
        self,
        length: int = 1024,
        vector_dim: int = 16,
        action_dim: int = 8,
        text_len: int = 8,
        vocab_size: int = 256,
        image_size: int = 32,
        audio_channels: int = 1,
        audio_length: int = 128,
        include_text: bool = False,
        include_image: bool = False,
        include_audio: bool = False,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.length = int(length)
        self.vector_dim = int(vector_dim)
        self.action_dim = int(action_dim)
        self.text_len = int(text_len)
        self.vocab_size = int(vocab_size)
        self.image_size = int(image_size)
        self.audio_channels = int(audio_channels)
        self.audio_length = int(audio_length)
        self.include_text = include_text
        self.include_image = include_image
        self.include_audio = include_audio
        self.seed = int(seed)

        g = torch.Generator().manual_seed(seed)
        self.action_to_vector = torch.randn(action_dim, vector_dim, generator=g) / max(action_dim, 1) ** 0.5
        self.vector_drift = torch.randn(vector_dim, generator=g) * 0.01

    def __len__(self) -> int:
        return self.length

    def _generator(self, idx: int) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed + int(idx) * 9973)

    def _render_image(self, vector: torch.Tensor) -> torch.Tensor:
        h = w = self.image_size
        yy = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
        weights = vector[:3].view(-1, 1, 1)
        if weights.shape[0] < 3:
            weights = torch.nn.functional.pad(weights, (0, 0, 0, 0, 0, 3 - weights.shape[0]))
        phase = vector[3:6].mean() if vector.numel() >= 6 else vector.mean()
        image = torch.sigmoid(weights[:3] * xx + torch.flip(weights[:3], dims=[0]) * yy + phase)
        return image.float()

    def _render_audio(self, vector: torch.Tensor) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, self.audio_length)
        channels = []
        base = vector.abs().mean().clamp_min(0.1)
        for ch in range(self.audio_channels):
            freq = 1.0 + base * float(ch + 1)
            phase = vector[ch % vector.numel()]
            channels.append(torch.sin(2.0 * torch.pi * freq * t + phase))
        return torch.stack(channels, dim=0).float()

    def _tokens_from_vector(self, vector: torch.Tensor) -> torch.Tensor:
        scaled = torch.round((vector[: self.text_len].abs() * 100.0)).long()
        if scaled.numel() < self.text_len:
            scaled = torch.nn.functional.pad(scaled, (0, self.text_len - scaled.numel()))
        return (scaled[: self.text_len] % max(self.vocab_size, 1)).long()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = self._generator(idx)
        vector_t = torch.randn(self.vector_dim, generator=g)
        action = torch.randn(self.action_dim, generator=g)
        vector_tp1 = 0.85 * vector_t + 0.15 * (action @ self.action_to_vector) + self.vector_drift

        item: Dict[str, torch.Tensor] = {
            "vector_t": vector_t.float(),
            "vector_tp1": vector_tp1.float(),
            "vector_target": vector_tp1.float(),
            "action": action.float(),
        }

        if self.include_text:
            text_t = self._tokens_from_vector(vector_t)
            text_tp1 = self._tokens_from_vector(vector_tp1)
            item.update({
                "text_t": text_t,
                "text_tp1": text_tp1,
                "text_target": text_tp1,
                "prefix_tokens": torch.cat([torch.zeros(1, dtype=torch.long), text_tp1[:-1]], dim=0),
            })

        if self.include_image:
            image_t = self._render_image(vector_t)
            image_tp1 = self._render_image(vector_tp1)
            item.update({
                "image_t": image_t,
                "image_tp1": image_tp1,
                "image_target": image_tp1,
            })

        if self.include_audio:
            audio_t = self._render_audio(vector_t)
            audio_tp1 = self._render_audio(vector_tp1)
            item.update({
                "audio_t": audio_t,
                "audio_tp1": audio_tp1,
                "audio_target": audio_tp1,
            })

        return item


def collate_transition_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Stack a list of deterministic transition items into a trainer batch."""
    if not items:
        raise ValueError("collate_transition_batch received an empty item list")
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items], dim=0) for key in keys}

