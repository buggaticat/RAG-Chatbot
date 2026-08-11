import json
import os
import re
import hashlib
from typing import Any, Dict, List

import boto3
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "open_ragbench/pdf/arxiv/corpus/"

ABSTRACT_SPLITTER = SentenceSplitter(chunk_size=240, chunk_overlap=24)
SECTION_SPLITTER = SentenceSplitter(chunk_size=280, chunk_overlap=28)
splitter = SECTION_SPLITTER


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
    text = re.sub(r"\r\n", "\n", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(
    text: str,
    base_metadata: Dict[str, Any],
    source_field: str,
    splitter: SentenceSplitter,
) -> List[Document]:
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
    parts: List[str] = []

    section_title = section.get("section_title") or section.get("title")
    if section_title:
        parts.append(str(section_title).strip())

    section_text = section.get("text", "")
    if section_text:
        parts.append(str(section_text).strip())

    return "\n\n".join(part for part in parts if part)


def _build_documents_from_paper(json_data: Dict[str, Any]) -> List[Document]:
    """
    Build retrieval documents from the paper JSON using section-aware chunking.
    """
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
    documents.extend(
        _chunk_text(abstract, base_metadata, "abstract", ABSTRACT_SPLITTER)
    )

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
    return hashlib.sha256(raw_bytes).hexdigest()


def build_all_documents() -> List[Document]:
    all_documents: List[Document] = []

    continuation_token = None
    s3 = boto3.client("s3")

    while True:
        request_kwargs = {
            "Bucket": BUCKET_NAME,
            "Prefix": S3_PREFIX,
        }
        if continuation_token:
            request_kwargs["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**request_kwargs)

        for obj in response.get("Contents", []):
            key = obj["Key"]
            file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            raw_bytes = file_obj["Body"].read()
            json_data = json.loads(raw_bytes.decode("utf-8"))
            documents = _build_documents_from_paper(json_data)
            source_hash = hash_json_bytes(raw_bytes)
            for document in documents:
                metadata = getattr(document, "metadata", None)
                if metadata is None:
                    continue
                metadata["source_key"] = key
                metadata["source_hash"] = source_hash
            all_documents.extend(documents)

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")

    return all_documents


if __name__ == "__main__":
    documents = build_all_documents()
    print(f"Built {len(documents)} documents")
