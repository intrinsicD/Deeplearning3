"""Manifest types for unified multimodal streaming."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast


@dataclass
class SourceSpec:
    """Declarative description of one streaming data source."""

    name: str
    type: str
    path: Optional[str] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    streaming: bool = True
    max_samples: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "SourceSpec":
        if not self.name:
            raise ValueError("SourceSpec.name must not be empty")
        if not self.type:
            raise ValueError("SourceSpec.type must not be empty")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("SourceSpec.max_samples must be > 0 when provided")
        return self

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceSpec":
        known = {"name", "type", "path", "dataset", "split", "streaming", "max_samples", "options"}
        raw_options = payload.get("options", {})
        if raw_options is None:
            raw_options = {}
        if not isinstance(raw_options, Mapping):
            raise TypeError("SourceSpec.options must be a mapping")
        options: Dict[str, Any] = {str(k): v for k, v in raw_options.items()}
        for key, value in payload.items():
            if key not in known:
                options[key] = value
        kwargs: Dict[str, Any] = {
            "name": str(payload["name"]),
            "type": str(payload["type"]),
            "path": None if payload.get("path") is None else str(payload.get("path")),
            "dataset": None if payload.get("dataset") is None else str(payload.get("dataset")),
            "split": None if payload.get("split") is None else str(payload.get("split")),
            "streaming": bool(payload.get("streaming", True)),
            "max_samples": None if payload.get("max_samples") is None else int(payload["max_samples"]),
            "options": options,
        }
        return cls(**kwargs).validate()

    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], asdict(self))


@dataclass
class DataManifest:
    """Collection of source specs for a training/eval stream."""

    sources: List[SourceSpec] = field(default_factory=list)

    def validate(self) -> "DataManifest":
        if not self.sources:
            raise ValueError("DataManifest.sources must not be empty")
        names = set()
        for source in self.sources:
            source.validate()
            if source.name in names:
                raise ValueError(f"Duplicate source name: {source.name}")
            names.add(source.name)
        return self

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataManifest":
        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            raise TypeError("DataManifest sources must be a list")
        return cls([SourceSpec.from_dict(item) for item in raw_sources]).validate()

    def to_dict(self) -> Dict[str, Any]:
        return {"sources": [source.to_dict() for source in self.sources]}

    @classmethod
    def from_json(cls, path: str | Path) -> "DataManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["DataManifest", "SourceSpec"]

