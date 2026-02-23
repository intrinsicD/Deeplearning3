"""Base protocol for modality adapters.

A ModalityAdapter provides a uniform contract for:
  1. encode()      — raw input → latent tokens
  2. get_queries() — generate target query tokens
  3. decode()      — latent tokens → modality-specific output

This allows the backbone and training loop to remain modality-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class Encoded:
    """Result of encoding a modality input."""
    tokens: torch.Tensor          # (B, N, D) latent token sequence
    meta: dict[str, Any] = field(default_factory=dict)
    # Original shape info, sample rate, patch size, etc.


@dataclass
class QueryResult:
    """Result of generating target queries."""
    queries: torch.Tensor          # (B, Q, D) query token sequence
    meta: dict[str, Any] = field(default_factory=dict)
    # Mapping from query index → output structure (patch grid, frame idx, etc.)


@runtime_checkable
class ModalityAdapter(Protocol):
    """Protocol that each modality adapter must implement.

    This defines the contract between modality-specific code and the
    modality-agnostic backbone/trainer.
    """

    name: str

    def encode(self, data: torch.Tensor) -> Encoded:
        """Encode raw modality data into latent tokens.

        Args:
            data: raw input tensor (shape depends on modality).

        Returns:
            Encoded dataclass with tokens and metadata.
        """
        ...

    def get_queries(self, batch_size: int, device: torch.device) -> QueryResult:
        """Generate learned target queries for this modality.

        For text, this is typically not used (teacher forcing instead).
        For image/audio/video, returns learned query parameters.

        Args:
            batch_size: batch size for expanding queries.
            device: target device.

        Returns:
            QueryResult with query tokens and metadata.
        """
        ...

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode latent tokens back to modality-specific output.

        Args:
            tokens: (B, N, D) latent tokens from backbone.

        Returns:
            Modality-specific output tensor.
        """
        ...
