from __future__ import annotations

from typing import Any

from .types import SourceChunk


def search_vector_store(query: str, *, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[SourceChunk]:
    _ = (query, top_k, filters)
    return []
