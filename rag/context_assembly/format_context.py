"""Format retrieved chunks into a prompt-friendly context block."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .config import get_tokenizer

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore

def _lookup_value(node: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an attribute-based object."""

    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _extract_source_nodes(response_or_chunks: Any) -> list[NodeWithScore]:
    """Normalize a response object or chunk collection into a list of chunks."""

    if response_or_chunks is None:
        return []

    source_nodes = getattr(response_or_chunks, "source_nodes", None)
    if callable(source_nodes):
        source_nodes = source_nodes()
    if source_nodes is not None:
        return list(source_nodes)

    if isinstance(response_or_chunks, Sequence) and not isinstance(response_or_chunks, (str, bytes)):
        return list(response_or_chunks)

    return []


def _node_payload(node: NodeWithScore) -> tuple[str, str, float | None, str, str]:
    """Extract the display payload used to render one chunk in context."""

    metadata = _lookup_value(node, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    doc_id = (
        metadata.get("doc_id")
        or metadata.get("source")
        or metadata.get("paper_id")
        or _lookup_value(node, "doc_id")
        or _lookup_value(node, "source")
        or _lookup_value(node, "paper_id")
        or "unknown"
    )
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or _lookup_value(node, "chunk_id") or _lookup_value(node, "chunk_index") or "unknown"
    title = metadata.get("title") or _lookup_value(node, "title") or ""
    score = node.get("score") if isinstance(node, dict) else getattr(node, "score", None)
    if score is None:
        get_score = getattr(node, "get_score", None)
        if callable(get_score):
            try:
                score = get_score()
            except Exception:
                score = None

    text = _lookup_value(node, "text", "") or _lookup_value(node, "content", "")
    if not text:
        inner_node = node.get("node") if isinstance(node, dict) else getattr(node, "node", None)
        if inner_node is not None:
            text = getattr(inner_node, "text", "") or getattr(inner_node, "content", "")

    return str(doc_id), str(chunk_id), score, title or "", str(text).strip()


def _estimate_tokens(text: str, tokenizer: Any | None) -> int:
    """Estimate token usage for a rendered context block."""

    if tokenizer is None:
        return len(text.split())
    encoded = tokenizer.encode(text)
    return len(encoded)


def format_context(
    response_or_chunks: Any,
    max_tokens: int | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Render retrieved chunks into a compact, labeled context string."""

    if tokenizer is None:
        tokenizer = get_tokenizer()

    chunks = _extract_source_nodes(response_or_chunks)
    blocks: list[str] = []
    used_tokens = 0

    for i, chunk in enumerate(chunks, start=1):
        doc_id, chunk_id, score, title, text = _node_payload(chunk)

        header_lines = [f"[Chunk {i}] doc={doc_id} chunk={chunk_id}"]
        if score is not None:
            try:
                header_lines[0] += f" score={float(score):.2f}"
            except Exception:
                header_lines[0] += f" score={score}"
        if title:
            header_lines.append(f"Title: {title}")

        block = "\n".join(header_lines + ([text] if text else []))
        block_tokens = _estimate_tokens(block, tokenizer)

        if max_tokens is not None and block_tokens + used_tokens > max_tokens:
            break

        blocks.append(block)
        used_tokens += block_tokens

    return "\n\n".join(blocks)
