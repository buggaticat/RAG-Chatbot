"""Translate user queries into English before retrieval."""

from jigsawstack import JigsawStack
from .config import JIGSAW_APIKEY, TARGET_LANGUAGE

jigsaw = JigsawStack(api_key=JIGSAW_APIKEY)

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

def translate_user_query(user_query: str) -> str:
    """Translate a user query to English using the configured translation service."""

    if not user_query:
        return ""

    try:
        response = jigsaw.translate.text({
            "text": user_query,
            "target_language": TARGET_LANGUAGE,
        })
        translated = _extract_translated_text(response)
        return translated or user_query
    except Exception:
        return user_query
    
