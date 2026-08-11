from __future__ import annotations

from typing import Any, Callable

from .types import RetrievalTrace, SourceChunk
from .vector_store import search_vector_store


def vector_search(query: str, *, top_k: int = 5, filters: dict[str, Any] | None = None) -> tuple[list[SourceChunk], RetrievalTrace]:
    chunks = search_vector_store(query, top_k=top_k, filters=filters)
    trace: RetrievalTrace = {
        "query": query,
        "rewritten_query": query,
        "retriever": "vector",
        "top_k": top_k,
        "filters": filters or {},
        "latency_ms": 0,
        "num_candidates": len(chunks),
        "notes": "vector search placeholder",
    }
    return chunks, trace


def hybrid_search(query: str, *, top_k: int = 5, filters: dict[str, Any] | None = None) -> tuple[list[SourceChunk], RetrievalTrace]:
    chunks, trace = vector_search(query, top_k=top_k, filters=filters)
    trace["retriever"] = "hybrid"
    trace["notes"] = "hybrid search placeholder"
    return chunks, trace


def bm25_search(query: str, *, top_k: int = 5, filters: dict[str, Any] | None = None) -> tuple[list[SourceChunk], RetrievalTrace]:
    chunks, trace = vector_search(query, top_k=top_k, filters=filters)
    trace["retriever"] = "bm25"
    trace["notes"] = "bm25 search placeholder"
    return chunks, trace


def build_retrieval_fn(search_fn: Callable[..., tuple[list[SourceChunk], RetrievalTrace]]) -> Callable[[str, Any], Any]:
    def _call(state: Any) -> Any:
        query = state.get("rewritten_query") or state.get("query") or ""
        chunks, trace = search_fn(query, top_k=state.get("top_k", 5), filters=state.get("filters"))
        state["retrieved_chunks"] = chunks
        state["retrieval_trace"] = trace
        return state

    return _call
