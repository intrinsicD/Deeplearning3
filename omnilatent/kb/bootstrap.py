"""Local file bootstrap for the lightweight KB."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Sequence

from omnilatent.core.config import KBConfig
from omnilatent.kb.file_store import KBFileStore, KBRecord, content_hash, stable_record_id
from omnilatent.kb.graph_store import GraphStore, TripleRecord

_SUPPORTED_EXTS = {".txt", ".md", ".py", ".pdf"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")


def discover_files(root: str | Path, extensions: Sequence[str] | None = None) -> List[Path]:
    root = Path(root)
    exts = {e.lower() for e in (extensions or _SUPPORTED_EXTS)}
    if root.is_file():
        return [root] if root.suffix.lower() in exts else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def read_supported_file(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _read_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, chunk_tokens: int = 512, chunk_overlap: int = 64) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_tokens:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_tokens")
    chunks = []
    step = chunk_tokens - chunk_overlap
    for start in range(0, len(tokens), step):
        piece = tokens[start:start + chunk_tokens]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_tokens >= len(tokens):
            break
    return chunks


def embed_text(text: str, dim: int = 512) -> List[float]:
    if dim <= 0:
        raise ValueError("embedding dim must be > 0")
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        idx = hash_token(token) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def hash_token(token: str) -> int:
    # Stable FNV-1a style hash; avoids Python's randomized hash().
    h = 2166136261
    for b in token.encode("utf-8", errors="replace"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def extract_entities(text: str, limit: int = 16) -> List[str]:
    seen = []
    for ent in _ENTITY_RE.findall(text):
        if ent not in seen:
            seen.append(ent)
        if len(seen) >= limit:
            break
    return seen


def records_from_file(path: str | Path, config: KBConfig) -> tuple[List[KBRecord], List[TripleRecord]]:
    path = Path(path)
    text = read_supported_file(path)
    chunks = chunk_text(text, config.chunk_tokens, config.chunk_overlap)
    records: List[KBRecord] = []
    triples: List[TripleRecord] = []
    for i, chunk in enumerate(chunks):
        entities = extract_entities(chunk)
        record_id = stable_record_id(path, i, chunk)
        relations = [{"subject": record_id, "predicate": "mentions", "object": e} for e in entities]
        record = KBRecord(
            id=record_id,
            source_path=str(path),
            content_hash=content_hash(chunk),
            modality="pdf" if path.suffix.lower() == ".pdf" else "text",
            chunk_text=chunk,
            embedding=embed_text(chunk, config.embedding_dim),
            entities=entities,
            relations=relations,
            metadata={"chunk_index": i, "extension": path.suffix.lower()},
        )
        records.append(record)
        triples.append(TripleRecord(subject=str(path), predicate="has_chunk", object=record_id, source_id=record_id))
        for ent in entities:
            triples.append(TripleRecord(subject=record_id, predicate="mentions", object=ent, source_id=record_id))
    return records, triples


def bootstrap_kb(input_path: str | Path, config: KBConfig, *, append: bool = False) -> dict[str, int]:
    files = discover_files(input_path, config.indexed_extensions)
    all_records: List[KBRecord] = []
    all_triples: List[TripleRecord] = []
    for path in files:
        records, triples = records_from_file(path, config)
        all_records.extend(records)
        all_triples.extend(triples)
    chunk_count = KBFileStore(config.root_dir, config.chunks_path).write_records(all_records, append=append)
    triple_count = GraphStore(config.root_dir, config.triples_path).write_triples(all_triples, append=append)
    return {"files": len(files), "chunks": chunk_count, "triples": triple_count}


def _read_pdf_text(path: Path) -> str:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return path.read_bytes()[:65536].decode("latin-1", errors="replace")
    doc = fitz.open(str(path))
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


__all__ = [
    "bootstrap_kb",
    "chunk_text",
    "discover_files",
    "embed_text",
    "extract_entities",
    "records_from_file",
    "read_supported_file",
]

