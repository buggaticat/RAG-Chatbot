"""Retrieval helpers for the RAG chatbot."""

from __future__ import annotations

from typing import Any

__all__ = ["run_hybrid_search"]


def __getattr__(name: str) -> Any:
    """Lazily expose the hybrid search entry point without eager imports."""

    if name == "run_hybrid_search":
        from .hybrid_search import run_hybrid_search

        return run_hybrid_search
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
