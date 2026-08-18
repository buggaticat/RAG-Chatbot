"""Translate user queries into English before retrieval."""

from __future__ import annotations

from typing import Any

from .config import JIGSAW_APIKEY, TARGET_LANGUAGE

jigsaw: Any | None = None

def _extract_translated_text(response) -> str:
    """Normalize the translation response into a plain string."""

    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        for key in ("text", "translation", "translated_text", "output"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    value = getattr(response, "text", None)
    if isinstance(value, str) and value.strip():
        return value.strip()

    return str(response).strip()


def _get_jigsaw_client() -> Any | None:
    """Construct the translation client lazily so import-time failures are avoided."""

    global jigsaw

    if jigsaw is not None:
        return jigsaw

    if not JIGSAW_APIKEY:
        return None

    try:
        from jigsawstack import JigsawStack
    except Exception:
        return None

    try:
        jigsaw = JigsawStack(api_key=JIGSAW_APIKEY)
    except Exception:
        jigsaw = None
    return jigsaw


def translate_user_query(user_query: str) -> str:
    """Translate a user query to English using the configured translation service."""

    if not user_query:
        return ""

    client = _get_jigsaw_client()
    if client is None:
        return user_query

    try:
        response = client.translate.text({
            "text": user_query,
            "target_language": TARGET_LANGUAGE,
        })
        translated = _extract_translated_text(response)
        return translated or user_query
    except Exception:
        return user_query
