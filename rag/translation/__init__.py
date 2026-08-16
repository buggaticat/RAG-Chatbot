"""Lazy exports for translation helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["translate_user_query"]


def __getattr__(name: str) -> Any:
    """Lazily expose the translation entry point without eager imports."""

    if name == "translate_user_query":
        from .translate import translate_user_query

        return translate_user_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
