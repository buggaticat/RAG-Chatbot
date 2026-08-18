"""Build LlamaIndex documents from the Open RAG Bench corpus."""

import hashlib
import json
import re
import time
from typing import Any, Dict, Iterator, List

import boto3
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from .config import ABSTRACT_SPLITTER, SECTION_SPLITTER, S3_BUCKET_NAME, S3_PREFIX

splitter = SECTION_SPLITTER
S3_READ_RETRIES = 3
S3_READ_RETRY_DELAY_SECONDS = 1.0


def _safe_metadata(value: Any) -> Any:
    """Convert nested values into metadata-friendly primitives."""
    if isinstance(value, dict):
        return {k: _safe_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_metadata(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_text(text: str) -> str:
    """Normalize line endings and excess blank lines in document text."""

    text = re.sub(r"\r\n", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(
    text: str,
    base_metadata: Dict[str, Any],
    source_field: str,
    splitter: SentenceSplitter,
) -> List[Document]:
    """Chunk normalized text into LlamaIndex documents with shared metadata."""

    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks = splitter.split_text(normalized)
    documents: List[Document] = []

    for chunk_index, chunk in enumerate(chunks):
        documents.append(
            Document(
                text=chunk,
                metadata={
                    **base_metadata,
                    "source_field": source_field,
                    "chunk_index": chunk_index,
                    "chunking_strategy": "section_aware_sentence_splitter",
                },
            )
        )

    return documents


def _normalize_section_text(section: Dict[str, Any]) -> str:
    """Combine section title and body text into a single chunking string."""

    parts: List[str] = []

    section_title = section.get("section_title") or section.get("title")
    if section_title:
        parts.append(str(section_title).strip())

    section_text = section.get("text", "")
    if section_text:
        parts.append(str(section_text).strip())

    return "\n\n".join(part for part in parts if part)


def _build_documents_from_paper(json_data: Dict[str, Any]) -> List[Document]:
    """Build retrieval documents from a paper JSON payload."""

    base_metadata = {
        "paper_id": json_data.get("id"),
        "title": json_data.get("title"),
        "authors": _safe_metadata(json_data.get("authors", [])),
        "categories": _safe_metadata(json_data.get("categories", [])),
        "updated": json_data.get("updated"),
        "published": json_data.get("published"),
    }

    documents: List[Document] = []

    abstract = json_data.get("abstract", "")
    documents.extend(_chunk_text(abstract, base_metadata, "abstract", ABSTRACT_SPLITTER))

    for section in json_data.get("sections", []) or []:
        section_id = section.get("section_id")
        section_metadata = {
            **base_metadata,
            "section_id": section_id,
            "section_title": section.get("section_title") or section.get("title"),
            "tables": _safe_metadata(section.get("tables", {})),
            "images": _safe_metadata(section.get("images", {})),
        }

        documents.extend(
            _chunk_text(
                _normalize_section_text(section),
                section_metadata,
                "sections.text",
                SECTION_SPLITTER,
            )
        )

        for table_id, table_markdown in (section.get("tables", {}) or {}).items():
            if not table_markdown:
                continue
            documents.extend(
                _chunk_text(
                    table_markdown,
                    {**section_metadata, "table_id": table_id},
                    "sections.tables",
                    SECTION_SPLITTER,
                )
            )

    return documents


def hash_json_bytes(raw_bytes: bytes) -> str:
    """Hash raw JSON bytes so source changes can be tracked deterministically."""

    return hashlib.sha256(raw_bytes).hexdigest()


def _read_s3_object_bytes(s3, key: str) -> bytes:
    """Read an S3 object with a small retry buffer for transient stream breaks."""

    last_exc: Exception | None = None
    for attempt in range(1, S3_READ_RETRIES + 1):
        try:
            file_obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            return file_obj["Body"].read()
        except Exception as exc:  # pragma: no cover - network/transient retry path
            last_exc = exc
            if attempt < S3_READ_RETRIES:
                time.sleep(S3_READ_RETRY_DELAY_SECONDS * attempt)
                continue
            raise

    if last_exc is not None:  # pragma: no cover - defensive fallback
        raise last_exc
    raise RuntimeError(f"Failed to read S3 object: {key}")


def iter_all_documents() -> Iterator[Document]:
    """Stream corpus documents from S3 one object at a time."""

    continuation_token = None
    s3 = boto3.client("s3")

    while True:
        request_kwargs = {
            "Bucket": S3_BUCKET_NAME,
            "Prefix": S3_PREFIX,
        }
        if continuation_token:
            request_kwargs["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**request_kwargs)

        for obj in response.get("Contents", []):
            key = obj["Key"]
            raw_bytes = _read_s3_object_bytes(s3, key)
            json_data = json.loads(raw_bytes.decode("utf-8"))
            documents = _build_documents_from_paper(json_data)
            source_hash = hash_json_bytes(raw_bytes)
            for document in documents:
                metadata = getattr(document, "metadata", None)
                if metadata is None:
                    continue
                metadata["source_key"] = key
                metadata["source_hash"] = source_hash
                yield document

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")


def build_all_documents() -> List[Document]:
    """Load every corpus JSON object from S3 and convert it into documents."""

    return list(iter_all_documents())


if __name__ == "__main__":
    documents = build_all_documents()
    print(f"Built {len(documents)} documents")
