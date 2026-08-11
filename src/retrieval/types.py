from __future__ import annotations

from typing import Any, Literal, TypedDict


class SourceChunk(TypedDict, total=False):
    doc_id: str
    chunk_id: str
    title: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any]


class RetrievalTrace(TypedDict, total=False):
    query: str
    rewritten_query: str
    retriever: Literal["vector", "hybrid", "bm25", "web"]
    reranker: str
    top_k: int
    filters: dict[str, Any]
    latency_ms: int
    num_candidates: int
    num_reranked: int
    notes: str
