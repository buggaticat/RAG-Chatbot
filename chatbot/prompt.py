"""Language-aware query preprocessing prompts for the RAG chatbot."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .utils import build_chat_template, clean_output_text, invoke_text_model, normalize_text, prompt_messages

_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[\.\)])\s*")


@dataclass(frozen=True)
class NormalizedQuery:
    """Normalized query state used by the chatbot graph."""

    original: str
    normalized: str


def normalize_query_language(
    user_query: str,
    *,
    translator: Callable[[str], str] | None = None,
) -> NormalizedQuery:
    """Translate a query into the working language using the configured translator."""

    cleaned = normalize_text(user_query)
    if not cleaned:
        return NormalizedQuery(original=user_query, normalized="")

    if translator is None:
        try:
            from rag.translation.translate import translate_user_query as translator  # type: ignore[no-redef]
        except Exception:
            translator = lambda value: value  # type: ignore[assignment]

    translated = normalize_text(translator(cleaned))
    return NormalizedQuery(original=user_query, normalized=translated or cleaned)


_REWRITE_SYSTEM_PROMPT = """You rewrite user questions for retrieval.
Return a single clear query in English.
Keep entities, numbers, dates, technical terms, and intent.
Remove filler words, ambiguity, and conversational phrasing.
If the user query is already precise, preserve it while making it retrieval-friendly.
"""

_REWRITE_HUMAN_PROMPT = """Rewrite this user query into a cleaner retrieval-optimized query.

User query:
{query}

Optional retrieval feedback:
{feedback}
"""

_DECOMPOSE_SYSTEM_PROMPT = """You split a retrieval query into smaller independent sub-questions only when decomposition is necessary.
If the question can be answered directly without splitting, do not decompose it.
Only decompose when the query clearly has multiple parts/questions, or when answering the final question requires first answering another question.
Each item should be answerable on its own and should not depend on another item's answer.
Prefer concise objectives that help retrieval.
Return JSON with a single key named "subquestions" whose value is a list of strings.
Return at most 3 subquestions.
If no decomposition is needed, return an empty list.
"""

_DECOMPOSE_HUMAN_PROMPT = """Break this rewritten query into smaller, independent sub-questions or objectives.

Rewritten query:
{query}

Optional retrieval feedback:
{feedback}
"""

_REWRITE_TEMPLATE = build_chat_template(_REWRITE_SYSTEM_PROMPT, _REWRITE_HUMAN_PROMPT)
_DECOMPOSE_TEMPLATE = build_chat_template(_DECOMPOSE_SYSTEM_PROMPT, _DECOMPOSE_HUMAN_PROMPT)


def _parse_subquestions(payload: str) -> list[str]:
    """Parse a model response into a normalized list of sub-questions."""

    stripped = payload.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
    except Exception:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(stripped[start : end + 1])
            except Exception:
                parsed = None
        else:
            parsed = None

    if isinstance(parsed, dict):
        for key in ("subquestions", "questions", "objectives", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                candidates = value
                break
        else:
            candidates = []
    elif isinstance(parsed, list):
        candidates = parsed
    else:
        candidates = []

    if candidates:
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, str):
                candidate = str(candidate)
            cleaned = clean_output_text(candidate)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
        return normalized

    lines = []
    for raw_line in stripped.splitlines():
        cleaned = _BULLET_RE.sub("", raw_line).strip()
        if cleaned:
            lines.append(clean_output_text(cleaned))

    seen: set[str] = set()
    normalized: list[str] = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(line)
    return normalized


def rewrite_user_query(
    user_query: str,
    *,
    model: Any | None = None,
    feedback: str | None = None,
) -> str:
    """Rewrite a user query into a retrieval-optimized form."""

    query = normalize_text(user_query)
    if not query:
        return ""

    feedback_text = normalize_text(feedback or "")
    if _REWRITE_TEMPLATE is None or model is None:
        return clean_output_text(query)

    messages = prompt_messages(
        _REWRITE_TEMPLATE,
        query=query,
        feedback=feedback_text or "None",
    )
    response = invoke_text_model(model, messages)
    return clean_output_text(response or query)


def decompose_rewritten_query(
    rewritten_query: str,
    *,
    model: Any | None = None,
    feedback: str | None = None,
    max_subquestions: int = 3,
) -> list[str]:
    """Break a rewritten query into smaller independent sub-questions."""

    query = normalize_text(rewritten_query)
    if not query:
        return []

    feedback_text = normalize_text(feedback or "")
    if _DECOMPOSE_TEMPLATE is None or model is None:
        return [query]

    messages = prompt_messages(
        _DECOMPOSE_TEMPLATE,
        query=query,
        feedback=feedback_text or "None",
    )
    response = invoke_text_model(model, messages)
    subquestions = _parse_subquestions(response or "")
    if not subquestions:
        return [query]
    return subquestions[:max_subquestions]
