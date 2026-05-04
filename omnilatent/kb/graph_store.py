"""Simple JSONL triple store for KB provenance and extracted facts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, cast


@dataclass
class TripleRecord:
    subject: str
    predicate: str
    object: str
    source_id: str
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TripleRecord":
        return cls(
            subject=str(payload["subject"]),
            predicate=str(payload["predicate"]),
            object=str(payload["object"]),
            source_id=str(payload.get("source_id", "")),
            confidence=float(payload.get("confidence", 1.0)),
            created_at=str(payload.get("created_at", datetime.now(timezone.utc).isoformat())),
            metadata=cast(Dict[str, Any], dict(payload.get("metadata", {}))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GraphStore:
    def __init__(self, root_dir: str | Path, triples_path: str = "triples.jsonl") -> None:
        self.root_dir = Path(root_dir)
        self.path = self.root_dir / triples_path

    def write_triples(self, triples: Iterable[TripleRecord], *, append: bool = True) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        count = 0
        with self.path.open(mode, encoding="utf-8") as f:
            for triple in triples:
                f.write(json.dumps(triple.to_dict(), sort_keys=True) + "\n")
                count += 1
        return count

    def read_triples(self) -> Iterator[TripleRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield TripleRecord.from_dict(json.loads(line))

    def query(self, *, subject: str | None = None, predicate: str | None = None, object: str | None = None) -> List[TripleRecord]:
        results = []
        for triple in self.read_triples():
            if subject is not None and triple.subject != subject:
                continue
            if predicate is not None and triple.predicate != predicate:
                continue
            if object is not None and triple.object != object:
                continue
            results.append(triple)
        return results


__all__ = ["GraphStore", "TripleRecord"]


