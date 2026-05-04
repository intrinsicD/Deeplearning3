"""KB retrieval and canonical KB tools."""

from __future__ import annotations

from pathlib import Path
from dataclasses import fields
from typing import List

import torch

from omnilatent.agent.tools import BaseTool
from omnilatent.core.config import KBConfig
from omnilatent.kb.bootstrap import embed_text
from omnilatent.kb.file_store import KBFileStore, KBRecord, content_hash, stable_record_id
from omnilatent.kb.graph_store import GraphStore, TripleRecord
from omnilatent.protocol import LatentPacket, ToolContext, ToolDescriptor, ToolResult


def retrieve_top_k(query: str, config: KBConfig, top_k: int | None = None) -> List[tuple[KBRecord, float]]:
    query_vec = torch.tensor(embed_text(query, config.embedding_dim), dtype=torch.float32)
    k = config.top_k if top_k is None else top_k
    scored: List[tuple[KBRecord, float]] = []
    for record in KBFileStore(config.root_dir, config.chunks_path).read_records():
        if not record.embedding:
            continue
        emb = torch.tensor(record.embedding, dtype=torch.float32)
        score = float(torch.dot(query_vec, emb).item())
        scored.append((record, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


class KBReadTool(BaseTool):
    """Retrieve top-k KB chunks and attach them to packet metadata."""

    def __init__(self, descriptor: ToolDescriptor, config: KBConfig) -> None:
        super().__init__(descriptor)
        self.config = config

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        query = str(context.raw_inputs.get("query") or packet.metadata.get("query") or "")
        top_k = context.raw_inputs.get("top_k")
        top_k_int = None if top_k is None else int(str(top_k))
        hits = retrieve_top_k(query, self.config, top_k_int)
        payload = [
            {"id": record.id, "text": record.chunk_text, "score": score, "source_path": record.source_path}
            for record, score in hits
        ]
        next_packet = LatentPacket(
            state=packet.state,
            confidence=packet.confidence,
            source_tool=self.descriptor.tool_id,
            timestamp=packet.timestamp,
            trace=packet.trace + [self.descriptor.tool_id],
            metadata={**packet.metadata, "kb_hits": payload},
        )
        confidence = torch.tensor(payload[0]["score"] if payload else 0.0, dtype=packet.state.z_sem.dtype, device=packet.state.z_sem.device)
        return ToolResult(packet=next_packet, raw_output=payload, confidence=confidence, metrics={"hits": float(len(payload))})


class KBWriteTool(BaseTool):
    """Append one explicit text chunk to the KB JSONL stores."""

    def __init__(self, descriptor: ToolDescriptor, config: KBConfig) -> None:
        super().__init__(descriptor)
        self.config = config

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        text = context.raw_inputs.get("text") or packet.metadata.get("kb_write_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("KBWriteTool requires non-empty raw_inputs['text'] or packet.metadata['kb_write_text']")
        source_path = str(context.raw_inputs.get("source_path", "tool://kb_write"))
        record_id = stable_record_id(source_path, int(context.raw_inputs.get("chunk_index", 0)), text)
        entities = [str(x) for x in context.raw_inputs.get("entities", [])]
        record = KBRecord(
            id=record_id,
            source_path=source_path,
            content_hash=content_hash(text),
            modality=str(context.raw_inputs.get("modality", "text")),
            chunk_text=text,
            embedding=embed_text(text, self.config.embedding_dim),
            entities=entities,
            metadata={"writer": self.descriptor.tool_id},
        )
        chunk_count = KBFileStore(self.config.root_dir, self.config.chunks_path).write_records([record], append=True)
        triples = [TripleRecord(subject=source_path, predicate="has_chunk", object=record_id, source_id=record_id)]
        triples.extend(TripleRecord(subject=record_id, predicate="mentions", object=e, source_id=record_id) for e in entities)
        triple_count = GraphStore(self.config.root_dir, self.config.triples_path).write_triples(triples, append=True)
        next_packet = LatentPacket(
            state=packet.state,
            confidence=packet.confidence,
            source_tool=self.descriptor.tool_id,
            timestamp=packet.timestamp,
            trace=packet.trace + [self.descriptor.tool_id],
            metadata={**packet.metadata, "kb_written_id": record_id},
        )
        return ToolResult(
            packet=next_packet,
            raw_output=record.to_dict(),
            metrics={"chunks_written": float(chunk_count), "triples_written": float(triple_count)},
            side_effects={"kb_written": record_id, "chunks_path": str(Path(self.config.root_dir) / self.config.chunks_path)},
        )


def build_kb_read_tool(descriptor: ToolDescriptor) -> BaseTool:
    return KBReadTool(descriptor, _config_from_descriptor(descriptor))


def build_kb_write_tool(descriptor: ToolDescriptor) -> BaseTool:
    return KBWriteTool(descriptor, _config_from_descriptor(descriptor))


def _config_from_descriptor(descriptor: ToolDescriptor) -> KBConfig:
    cfg = descriptor.config.get("kb_config")
    if isinstance(cfg, KBConfig):
        return cfg
    if isinstance(cfg, dict):
        return KBConfig(**cfg)
    field_names = {field.name for field in fields(KBConfig)}
    return KBConfig(**{k: v for k, v in descriptor.config.items() if k in field_names})


__all__ = [
    "KBReadTool",
    "KBWriteTool",
    "build_kb_read_tool",
    "build_kb_write_tool",
    "retrieve_top_k",
]

