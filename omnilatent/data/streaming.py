"""Manifest-driven unified multimodal streaming."""

from __future__ import annotations

from itertools import islice
from typing import Callable, Dict, Iterable, Iterator

from torch.utils.data import IterableDataset

from omnilatent.data.manifest import DataManifest, SourceSpec
from omnilatent.data.sample import MultiModalSample

SourceFactory = Callable[[SourceSpec], Iterable[MultiModalSample]]


def _default_registry() -> Dict[str, SourceFactory]:
    from omnilatent.data.sources.hf import iter_hf_stream
    from omnilatent.data.sources.local import iter_local_files
    from omnilatent.data.sources.minari import iter_minari_transitions
    from omnilatent.data.sources.webdataset import iter_webdataset_shards

    return {
        "hf": iter_hf_stream,
        "hf_stream": iter_hf_stream,
        "huggingface": iter_hf_stream,
        "local": iter_local_files,
        "local_files": iter_local_files,
        "local_text": iter_local_files,
        "minari": iter_minari_transitions,
        "webdataset": iter_webdataset_shards,
        "wds": iter_webdataset_shards,
    }


class StreamingMultiModalDataset(IterableDataset[MultiModalSample]):
    """PyTorch IterableDataset over one or more manifest sources."""

    def __init__(
        self,
        manifest: DataManifest,
        registry: Dict[str, SourceFactory] | None = None,
    ) -> None:
        super().__init__()
        self.manifest = manifest.validate()
        self.registry = _default_registry() if registry is None else dict(registry)

    def __iter__(self) -> Iterator[MultiModalSample]:
        for source in self.manifest.sources:
            source_type = source.type.lower().strip()
            if source_type not in self.registry:
                raise ValueError(f"Unknown source type {source.type!r}; available: {sorted(self.registry)}")
            iterator = iter(self.registry[source_type](source))
            if source.max_samples is not None:
                iterator = islice(iterator, source.max_samples)
            for sample in iterator:
                yield sample.with_metadata(source=source.name, source_type=source.type)


def stream_samples(manifest: DataManifest, registry: Dict[str, SourceFactory] | None = None) -> Iterator[MultiModalSample]:
    """Convenience iterator for scripts that do not need a DataLoader."""
    return iter(StreamingMultiModalDataset(manifest, registry=registry))


__all__ = ["SourceFactory", "StreamingMultiModalDataset", "stream_samples"]

