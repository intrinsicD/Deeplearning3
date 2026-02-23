"""Standardized query providers for target modalities.

Each non-text modality uses learned target queries (similar to DETR
object queries or Perceiver latent arrays). This module provides a
uniform interface for query generation with explicit metadata about
the output structure each query maps to.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from omnilatent.config import OmniLatentConfig
from omnilatent.model.adapters.base import QueryResult


class QueryProvider(nn.Module):
    """Generates learned target queries for a specific modality.

    Each query maps to a specific position in the output structure
    (e.g., a spatial patch position for images, a temporal position
    for audio, a spatio-temporal position for video).
    """

    def __init__(
        self,
        num_queries: int,
        hidden_dim: int,
        name: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.num_queries = num_queries
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )

    def forward(self, batch_size: int) -> QueryResult:
        """Return expanded query tokens with metadata.

        Args:
            batch_size: expand queries to this batch size.

        Returns:
            QueryResult with queries and modality metadata.
        """
        return QueryResult(
            queries=self.queries.expand(batch_size, -1, -1),
            meta={"modality": self.name, "num_queries": self.num_queries},
        )


class ImageQueryProvider(QueryProvider):
    """Query provider for image targets."""

    def __init__(self, config: OmniLatentConfig) -> None:
        grid = config.image_size // config.image_patch_size
        super().__init__(
            num_queries=grid * grid,
            hidden_dim=config.hidden_dim,
            name="image",
        )
        self.grid_size = grid

    def forward(self, batch_size: int) -> QueryResult:
        result = super().forward(batch_size)
        result.meta["grid_size"] = self.grid_size
        result.meta["patch_size"] = None  # resolved from config
        return result


class AudioQueryProvider(QueryProvider):
    """Query provider for audio targets."""

    def __init__(self, config: OmniLatentConfig) -> None:
        num_queries = config.audio_max_frames // config.audio_patch_frames
        super().__init__(
            num_queries=num_queries,
            hidden_dim=config.hidden_dim,
            name="audio",
        )

    def forward(self, batch_size: int) -> QueryResult:
        result = super().forward(batch_size)
        result.meta["temporal_tokens"] = self.num_queries
        return result


class VideoQueryProvider(QueryProvider):
    """Query provider for video targets."""

    def __init__(self, config: OmniLatentConfig) -> None:
        num_queries = (
            (config.video_max_frames // config.video_temporal_patch)
            * config.video_spatial_patches
        )
        super().__init__(
            num_queries=num_queries,
            hidden_dim=config.hidden_dim,
            name="video",
        )
        self.temporal_tokens = config.video_max_frames // config.video_temporal_patch
        self.spatial_grid = config.video_size // config.video_patch_size

    def forward(self, batch_size: int) -> QueryResult:
        result = super().forward(batch_size)
        result.meta["temporal_tokens"] = self.temporal_tokens
        result.meta["spatial_grid"] = self.spatial_grid
        return result
