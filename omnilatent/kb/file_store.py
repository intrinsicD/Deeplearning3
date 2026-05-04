"""JSONL chunk store for the lightweight local KB."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, cast


@dataclass
class KBRecord:
    id: str
    source_path: str
    content_hash: str
    modality: str
    chunk_text: str
    embedding: List[float]
    entities: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "KBRecord":
        return cls(
            id=str(payload["id"]),
            source_path=str(payload.get("source_path", "")),
            content_hash=str(payload.get("content_hash", "")),
            modality=str(payload.get("modality", "text")),
            chunk_text=str(payload.get("chunk_text", "")),
            embedding=[float(x) for x in payload.get("embedding", [])],
            entities=[str(x) for x in payload.get("entities", [])],
            relations=list(payload.get("relations", [])),
            created_at=str(payload.get("created_at", datetime.now(timezone.utc).isoformat())),
            metadata=cast(Dict[str, Any], dict(payload.get("metadata", {}))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KBFileStore:
    def __init__(self, root_dir: str | Path, chunks_path: str = "chunks.jsonl") -> None:
        self.root_dir = Path(root_dir)
        self.path = self.root_dir / chunks_path

    def write_records(self, records: Iterable[KBRecord], *, append: bool = True) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        count = 0
        with self.path.open(mode, encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                count += 1
        return count

    def read_records(self) -> Iterator[KBRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield KBRecord.from_dict(json.loads(line))

    def get(self, record_id: str) -> Optional[KBRecord]:
        for record in self.read_records():
            if record.id == record_id:
                return record
        return None


def stable_record_id(source_path: str | Path, chunk_index: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(str(source_path).encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


__all__ = ["KBFileStore", "KBRecord", "content_hash", "stable_record_id"]


