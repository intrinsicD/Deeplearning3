"""Data layer for OmniLatent.

Organized as:
  - datasets/   — per-domain dataset implementations
  - transforms/ — tokenization, resizing, patching
  - collate/    — per-modality collate functions
  - registry    — build dataset from config
"""

from omnilatent.data.manifest import DataManifest, SourceSpec
from omnilatent.data.registry import build_dataset
from omnilatent.data.sample import MultiModalSample
from omnilatent.data.streaming import StreamingMultiModalDataset, stream_samples

__all__ = [
    "DataManifest",
    "MultiModalSample",
    "SourceSpec",
    "StreamingMultiModalDataset",
    "build_dataset",
    "stream_samples",
]
