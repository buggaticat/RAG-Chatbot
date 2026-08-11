from __future__ import annotations

from .types import RetrievalTrace, SourceChunk


def rerank_chunks(query: str, chunks: list[SourceChunk], *, top_k: int = 5) -> list[SourceChunk]:
    _ = query
    return chunks[:top_k]


def format_source_chunks(chunks: list[SourceChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or "Untitled source"
        source = chunk.get("source") or chunk.get("doc_id") or "unknown"
        content = (chunk.get("content") or "").strip()
        score = chunk.get("score")
        header = f"[{index}] {title} | {source}"
        if score is not None:
            header = f"{header} | score={score:.4f}"
        lines.append(header)
        if content:
            lines.append(content)
            lines.append("")
    return "\n".join(lines).strip()


def update_retrieval_trace(trace: RetrievalTrace, *, reranker: str, num_reranked: int) -> RetrievalTrace:
    trace = dict(trace)
    trace["reranker"] = reranker
    trace["num_reranked"] = num_reranked
    return trace
