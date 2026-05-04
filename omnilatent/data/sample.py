"""Canonical samples emitted by the unified streaming data layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class MultiModalSample:
    """Normalized sample shared by text, media, control, and KB data sources."""

    text: Optional[str] = None
    image: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    pdf_page: Optional[str] = None
    vector: Optional[torch.Tensor] = None
    action: Optional[torch.Tensor] = None
    reward: Optional[float] = None
    next_observation: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_any_modality(self) -> bool:
        return any(
            value is not None
            for value in (self.text, self.image, self.audio, self.video, self.pdf_page, self.vector)
        )

    def with_metadata(self, **metadata: Any) -> "MultiModalSample":
        merged = dict(self.metadata)
        merged.update(metadata)
        return MultiModalSample(
            text=self.text,
            image=self.image,
            audio=self.audio,
            video=self.video,
            pdf_page=self.pdf_page,
            vector=self.vector,
            action=self.action,
            reward=self.reward,
            next_observation=self.next_observation,
            metadata=merged,
        )


__all__ = ["MultiModalSample"]

