from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from omnilatent.core.config import KBConfig
from omnilatent.kb import (
    GraphStore,
    KBFileStore,
    KBReadTool,
    KBWriteTool,
    bootstrap_kb,
    chunk_text,
    discover_files,
    embed_text,
    retrieve_top_k,
)
from omnilatent.protocol import LatentPacket, LatentState, ToolContext, ToolDescriptor, ToolResult


def _kb_config(tmp_path: Path) -> KBConfig:
    return KBConfig(
        root_dir=str(tmp_path / "kb"),
        embedding_dim=64,
        chunk_tokens=12,
        chunk_overlap=0,
        top_k=2,
    )


def test_kb_import_does_not_import_mmwm_in_fresh_process() -> None:
    code = """
import sys
import omnilatent.kb
print(any(name == 'MMWM' or name.startswith('MMWM.') for name in sys.modules))
"""
    proc = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert proc.stdout.strip() == "False"


def test_discover_chunk_and_embed_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Alpha Beta Gamma", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"skip")

    files = discover_files(tmp_path)
    chunks = chunk_text("one two three four five", chunk_tokens=2, chunk_overlap=1)
    emb_a = embed_text("Alpha Beta", dim=16)
    emb_b = embed_text("Alpha Beta", dim=16)

    assert [p.name for p in files] == ["a.txt"]
    assert chunks == ["one two", "two three", "three four", "four five"]
    assert emb_a == emb_b
    assert abs(sum(x * x for x in emb_a) - 1.0) < 1e-6


def test_bootstrap_indexes_txt_md_py_pdf_and_writes_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "notes.txt").write_text("Neural hooks retrieve knowledge about AlphaSystem.", encoding="utf-8")
    (source / "readme.md").write_text("# Retrieval\nGraph memory stores BetaEntity facts.", encoding="utf-8")
    (source / "code.py").write_text("class GammaTool:\n    pass\n", encoding="utf-8")
    (source / "paper.pdf").write_bytes(b"%PDF-1.4\nDeltaPDF content about retrieval")
    cfg = _kb_config(tmp_path)

    stats = bootstrap_kb(source, cfg)

    assert stats["files"] == 4
    assert stats["chunks"] >= 4
    assert stats["triples"] >= stats["chunks"]
    chunks_path = Path(cfg.root_dir) / cfg.chunks_path
    triples_path = Path(cfg.root_dir) / cfg.triples_path
    assert chunks_path.exists()
    assert triples_path.exists()
    records = list(KBFileStore(cfg.root_dir, cfg.chunks_path).read_records())
    triples = list(GraphStore(cfg.root_dir, cfg.triples_path).read_triples())
    assert any(record.modality == "pdf" for record in records)
    assert any("AlphaSystem" in record.entities for record in records)
    assert any(triple.predicate == "has_chunk" for triple in triples)


def test_retrieve_top_k_ranks_relevant_chunks(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "neural.txt").write_text("neural hook latent retrieval retrieval", encoding="utf-8")
    (source / "cooking.txt").write_text("soup onion kitchen recipe", encoding="utf-8")
    cfg = _kb_config(tmp_path)
    bootstrap_kb(source, cfg)

    hits = retrieve_top_k("latent retrieval hook", cfg, top_k=1)

    assert len(hits) == 1
    assert "neural.txt" in hits[0][0].source_path
    assert hits[0][1] > 0.0


def test_kb_read_tool_attaches_hits_to_packet_metadata(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "kb.md").write_text("KnowledgeBase retrieval finds Omnilatent chunks.", encoding="utf-8")
    cfg = _kb_config(tmp_path)
    bootstrap_kb(source, cfg)
    tool = KBReadTool(ToolDescriptor(tool_id="kb.read.v1", kind="kb.read", version="1"), cfg)
    packet = LatentPacket(state=LatentState(z_sem=torch.zeros(1, 4)))

    result = tool.run(packet, ToolContext(raw_inputs={"query": "Omnilatent retrieval", "top_k": 1}))

    assert isinstance(result, ToolResult)
    assert result.packet is not None
    assert result.packet.source_tool == "kb.read.v1"
    assert result.packet.metadata["kb_hits"]
    assert result.confidence is not None
    assert result.metrics["hits"] == 1.0


def test_kb_write_tool_writes_jsonl_and_retrieval_finds_record(tmp_path: Path) -> None:
    cfg = _kb_config(tmp_path)
    tool = KBWriteTool(ToolDescriptor(tool_id="kb.write.v1", kind="kb.write", version="1"), cfg)
    packet = LatentPacket(state=LatentState(z_sem=torch.zeros(1, 4)))

    result = tool.run(packet, ToolContext(raw_inputs={
        "text": "AgentGraph writes durable Omnilatent memory.",
        "source_path": "tool://unit",
        "entities": ["AgentGraph", "Omnilatent"],
    }))

    assert result.packet is not None
    assert "kb_written_id" in result.packet.metadata
    assert result.side_effects["kb_written"] == result.packet.metadata["kb_written_id"]
    assert Path(cfg.root_dir, cfg.chunks_path).exists()
    assert Path(cfg.root_dir, cfg.triples_path).exists()
    hits = retrieve_top_k("durable memory AgentGraph", cfg, top_k=1)
    assert len(hits) == 1
    assert hits[0][0].source_path == "tool://unit"


def test_kb_read_and_write_tools_can_be_built_from_descriptors(tmp_path: Path) -> None:
    from omnilatent.kb.retrieval import build_kb_read_tool, build_kb_write_tool

    cfg = _kb_config(tmp_path)
    read_tool = build_kb_read_tool(ToolDescriptor(
        tool_id="read",
        kind="kb.read",
        version="1",
        config={"kb_config": cfg},
    ))
    write_tool = build_kb_write_tool(ToolDescriptor(
        tool_id="write",
        kind="kb.write",
        version="1",
        config={"root_dir": cfg.root_dir, "embedding_dim": cfg.embedding_dim},
    ))

    assert isinstance(read_tool, KBReadTool)
    assert isinstance(write_tool, KBWriteTool)

