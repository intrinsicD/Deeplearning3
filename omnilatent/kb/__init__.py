"""Lightweight local knowledge-base bootstrap and retrieval."""

from omnilatent.kb.bootstrap import bootstrap_kb, chunk_text, discover_files, embed_text
from omnilatent.kb.file_store import KBFileStore, KBRecord
from omnilatent.kb.graph_store import GraphStore, TripleRecord
from omnilatent.kb.retrieval import KBReadTool, KBWriteTool, retrieve_top_k

__all__ = [
    "GraphStore",
    "KBFileStore",
    "KBReadTool",
    "KBRecord",
    "KBWriteTool",
    "TripleRecord",
    "bootstrap_kb",
    "chunk_text",
    "discover_files",
    "embed_text",
    "retrieve_top_k",
]

