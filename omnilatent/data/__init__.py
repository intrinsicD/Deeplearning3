"""Data layer for OmniLatent.

Organized as:
  - datasets/   — per-domain dataset implementations
  - transforms/ — tokenization, resizing, patching
  - collate/    — per-modality collate functions
  - registry    — build dataset from config
"""

from omnilatent.data.collate import build_sample_collator, collate_multimodal_samples
from omnilatent.data.errors import MediaDecodeError
from omnilatent.data.manifest import DataManifest, SourceSpec
from omnilatent.data.registry import build_dataset
from omnilatent.data.sample import MultiModalSample
from omnilatent.data.streaming import StreamingMultiModalDataset, stream_samples

__all__ = [
    "DataManifest",
    "MediaDecodeError",
    "MultiModalSample",
    "SourceSpec",
    "StreamingMultiModalDataset",
    "build_dataset",
    "build_sample_collator",
    "collate_multimodal_samples",
    "stream_samples",
]
