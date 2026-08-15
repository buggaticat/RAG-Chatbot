from __future__ import annotations

import json
import re
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import MAX_NOT_FOUND_WORDS, NOT_FOUND_SENTINEL, VALID_CONFIDENCE_LEVELS

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{4})\b")
_TABLE_MARKER_RE = re.compile(r"\btable\b", re.IGNORECASE)

_NUMERIC_RE = re.compile(
    r"""
    (?<![\w/.-])
    [-+]?
    (?:
        (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
        |
        \.\d+
    )
    (?:[eE][-+]?\d+)?
    (?:%)?
    (?![\w/.-])
    """,
    re.VERBOSE,
)


@dataclass
class DeterministicValidationResult:
    """Container for deterministic validation status and human-readable errors."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)


def _lookup_value(node: Any, key: str, default: Any = None) -> Any:
    """Read a value from either a dict-like object or a plain attribute object."""

    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _extract_source_nodes(response_or_chunks: Any) -> list["NodeWithScore"]:
    """Normalize a response object or chunk list into a plain list of source nodes."""

    if response_or_chunks is None:
        return []

    source_nodes = getattr(response_or_chunks, "source_nodes", None)
    if callable(source_nodes):
        source_nodes = source_nodes()
    if source_nodes is not None:
        return list(source_nodes)

    if isinstance(response_or_chunks, list):
        return list(response_or_chunks)

    if isinstance(response_or_chunks, tuple):
        return list(response_or_chunks)

    return []


def _node_payload(node: "NodeWithScore") -> tuple[str, str, str, str, str]:
    """Extract doc id, chunk id, section id, paper id, and text from a chunk-like object."""

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
    section_id = metadata.get("section_id") or _lookup_value(node, "section_id") or ""
    paper_id = metadata.get("paper_id") or _lookup_value(node, "paper_id") or doc_id

    text = _lookup_value(node, "text", "") or _lookup_value(node, "content", "")
    if not text:
        inner_node = _lookup_value(node, "node")
        if inner_node is not None:
            text = _lookup_value(inner_node, "text", "") or _lookup_value(inner_node, "content", "")

    return str(doc_id), str(chunk_id), str(section_id), str(paper_id), str(text).strip()


def _normalize_text(value: str) -> str:
    """Collapse repeated whitespace so exact string comparisons are stable."""

    return " ".join(value.strip().split())


def _find_exact_numeric_literals(text: str) -> list[str]:
    """Find numeric literals in text while preserving the exact surface form."""

    matches = []
    for match in _NUMERIC_RE.finditer(text):
        token = match.group(0)
        if token in {"+", "-"}:
            continue
        matches.append(token)
    return matches


def _find_date_literals(text: str) -> list[str]:
    """Find ISO-like years and dates in text."""

    return [match.group(1) for match in _DATE_RE.finditer(text)]


def _is_valid_date_literal(value: str) -> bool:
    """Check whether a literal is a valid ISO date or year."""

    if re.fullmatch(r"\d{4}", value):
        return True
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _answer_is_short_abstention(answer: str) -> bool:
    """Check whether an abstention answer stays short and non-committal."""

    return len(answer.split()) <= MAX_NOT_FOUND_WORDS and answer.strip() == NOT_FOUND_SENTINEL


def _answer_looks_like_abstention(answer: str) -> bool:
    """Detect answers that start with the abstention sentinel."""

    normalized = answer.strip()
    return normalized == NOT_FOUND_SENTINEL or normalized.startswith(f"{NOT_FOUND_SENTINEL} ")


def _has_duplicate_citations(citations: list[dict[str, Any]]) -> bool:
    """Detect repeated citation targets."""

    seen: set[tuple[str, str]] = set()
    for citation in citations:
        key = (citation["doc_id"], citation["chunk_id"])
        if key in seen:
            return True
        seen.add(key)
    return False


def _citation_count_is_sane(citations: list[dict[str, Any]], answer: str) -> bool:
    """Apply a simple citation-count sanity check for short answers."""

    word_count = len(answer.split())
    if word_count <= 12:
        return len(citations) <= 2
    return len(citations) <= 5


def _looks_like_table_answer(answer: str) -> bool:
    """Detect whether the answer is explicitly referring to a table."""

    return bool(_TABLE_MARKER_RE.search(answer)) or "row" in answer.lower() or "column" in answer.lower()


def _chunk_looks_like_table(text: str) -> bool:
    """Detect simple table-like structure in chunk text."""

    return "|" in text or "\t" in text or bool(_TABLE_MARKER_RE.search(text))


def _validate_metadata_and_structure(
    answer: str,
    citations: list[dict[str, Any]],
    chunk_map: dict[tuple[str, str], tuple[str, str, str, str]],
) -> list[str]:
    """Run shallow metadata, date, and table-structure checks."""

    errors: list[str] = []
    cited_texts: list[str] = []

    for citation in citations:
        key = (citation["doc_id"], citation["chunk_id"])
        payload = chunk_map.get(key)
        if not payload:
            matching_chunk_ids = [item for item_key, item in chunk_map.items() if item_key[1] == citation["chunk_id"]]
            if matching_chunk_ids:
                _, section_id, paper_id, _ = matching_chunk_ids[0]
                if citation["doc_id"] != paper_id:
                    errors.append(f"Citation {citation['doc_id']}:{citation['chunk_id']} does not match the cited paper metadata.")
            continue
        _, _, _, text = payload
        cited_texts.append(text)
        doc_id, section_id, paper_id, _ = payload

        if not paper_id or paper_id == "unknown":
            errors.append(f"Citation {doc_id}:{citation['chunk_id']} is missing paper metadata.")
        if citation["doc_id"] != paper_id and citation["doc_id"] != doc_id:
            errors.append(f"Citation {citation['doc_id']}:{citation['chunk_id']} does not match the cited paper metadata.")
        if section_id and not str(section_id).strip():
            errors.append(f"Citation {doc_id}:{citation['chunk_id']} has an invalid section_id.")

    answer_dates = _find_date_literals(answer)
    for literal in answer_dates:
        if not _is_valid_date_literal(literal):
            errors.append(f"Date literal '{literal}' is not valid.")

    if len(answer_dates) >= 2 and re.search(r"\bto\b|\-|\bthrough\b", answer, re.IGNORECASE):
        parsed_dates: list[date] = []
        for literal in answer_dates[:2]:
            if re.fullmatch(r"\d{4}", literal):
                parsed_dates.append(date(int(literal), 1, 1))
            else:
                parsed_dates.append(datetime.strptime(literal, "%Y-%m-%d").date())
        if len(parsed_dates) == 2 and parsed_dates[0] > parsed_dates[1]:
            errors.append("Date ordering is invalid.")

    if _looks_like_table_answer(answer):
        if not any(_chunk_looks_like_table(text) for text in cited_texts):
            errors.append("Table-oriented answers must cite a table-like chunk.")

    return errors


def _cited_context_text(citations: list[dict[str, Any]], chunks: list["NodeWithScore"]) -> str:
    """Concatenate the text for all cited chunks that can be resolved locally."""

    chunk_map: dict[tuple[str, str], str] = {}
    for chunk in chunks:
        doc_id, chunk_id, _, _, text = _node_payload(chunk)
        chunk_map[(doc_id, chunk_id)] = text

    parts: list[str] = []
    for citation in citations:
        doc_id = str(citation.get("doc_id", "")).strip()
        chunk_id = str(citation.get("chunk_id", "")).strip()
        text = chunk_map.get((doc_id, chunk_id))
        if text:
            parts.append(text)
    return "\n".join(parts)


def _parse_output(output: Any) -> tuple[str, list[dict[str, Any]], str | None, list[str]]:
    """Parse the model output and validate the expected JSON shape."""

    errors: list[str] = []

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return "", [], None, ["Output is not valid JSON."]

    if not isinstance(output, dict):
        return "", [], None, ["Output must be a JSON object."]

    answer = output.get("answer")
    citations = output.get("citations")
    confidence = output.get("confidence")

    if not isinstance(answer, str) or not answer.strip():
        errors.append("Missing or empty 'answer'.")
        answer = "" if answer is None else str(answer)

    if citations is None:
        citations = []
    if not isinstance(citations, list):
        errors.append("'citations' must be a list.")
        citations = []

    normalized_citations: list[dict[str, Any]] = []
    for i, citation in enumerate(citations):
        if not isinstance(citation, dict):
            errors.append(f"Citation at index {i} must be an object.")
            continue
        doc_id = citation.get("doc_id")
        chunk_id = citation.get("chunk_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            errors.append(f"Citation at index {i} is missing a valid 'doc_id'.")
            continue
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            errors.append(f"Citation at index {i} is missing a valid 'chunk_id'.")
            continue
        normalized_citations.append({"doc_id": doc_id.strip(), "chunk_id": chunk_id.strip()})

    return answer, normalized_citations, confidence if isinstance(confidence, str) else None, errors


def validate_deterministic_output(output: Any, response_or_chunks: Any) -> DeterministicValidationResult:
    """Run deterministic checks over a grounded answer and its retrieved chunks."""

    answer, citations, confidence, errors = _parse_output(output)
    chunks = _extract_source_nodes(response_or_chunks)

    if errors:
        return DeterministicValidationResult(False, errors)

    chunk_map: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    for chunk in chunks:
        doc_id, chunk_id, section_id, paper_id, text = _node_payload(chunk)
        chunk_map[(doc_id, chunk_id)] = (doc_id, section_id, paper_id, text)

    chunk_keys = set(chunk_map)
    cited_keys = {(citation["doc_id"], citation["chunk_id"]) for citation in citations}

    if _has_duplicate_citations(citations):
        errors.append("Citations must not contain duplicates.")

    if confidence not in VALID_CONFIDENCE_LEVELS:
        errors.append("'confidence' must be one of: low, medium, high.")

    unknown_citations = sorted(cited_keys - chunk_keys)
    if unknown_citations:
        errors.append(
            "Citations must reference retrieved chunks only: "
            + ", ".join(f"{doc_id}:{chunk_id}" for doc_id, chunk_id in unknown_citations)
        )

    metadata_errors = _validate_metadata_and_structure(answer, citations, chunk_map)
    errors.extend(metadata_errors)

    if _answer_looks_like_abstention(answer):
        if citations:
            errors.append("If the answer is 'Not found in context', citations must be empty.")
        if not _answer_is_short_abstention(answer):
            errors.append("Abstention answers must stay short and exact.")
        return DeterministicValidationResult(not errors, errors)

    if not _citation_count_is_sane(citations, answer):
        errors.append("Citation count looks excessive for the answer length.")

    cited_context = _normalize_text(_cited_context_text(citations, chunks))
    if not cited_context:
        errors.append("Answer must include at least one valid citation with retrievable context.")
        return DeterministicValidationResult(False, errors)

    answer_numbers = _find_exact_numeric_literals(answer)
    for number in answer_numbers:
        if number not in cited_context:
            errors.append(f"Numeric value '{number}' must appear exactly in cited context.")

    answer_dates = _find_date_literals(answer)
    for literal in answer_dates:
        if literal not in cited_context:
            errors.append(f"Date literal '{literal}' must appear exactly in cited context.")

    return DeterministicValidationResult(not errors, errors)
