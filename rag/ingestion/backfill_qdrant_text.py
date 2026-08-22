"""Backfill missing `text` payloads for existing Qdrant points."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from qdrant_client import QdrantClient

from .config import (
    COLLECTION_NAME,
    QDRANT_APIKEY,
    QDRANT_CLUSTER_ENDPOINT,
    QDRANT_REQUEST_TIMEOUT_SECONDS,
)
from .build_documents import iter_all_documents


@dataclass
class BackfillResult:
    """Summary of a text backfill run."""

    scanned_points: int
    updated_points: int
    skipped_with_text: int
    skipped_without_source_text: int

    def to_dict(self) -> dict[str, int]:
        """Render the run summary into JSON-safe data."""

        return asdict(self)


def _get_client() -> QdrantClient:
    """Build the Qdrant client used for in-place payload updates."""

    return QdrantClient(
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_APIKEY,
        timeout=QDRANT_REQUEST_TIMEOUT_SECONDS,
    )


def _normalize_key_value(value: Any) -> str:
    """Convert a metadata value into a stable lookup fragment."""

    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _chunk_lookup_key(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Build a metadata key that matches source chunks to Qdrant points."""

    return (
        _normalize_key_value(metadata.get("source_key")),
        _normalize_key_value(metadata.get("paper_id")),
        _normalize_key_value(metadata.get("section_id")),
        _normalize_key_value(metadata.get("chunk_index")),
        _normalize_key_value(metadata.get("source_field")),
        _normalize_key_value(metadata.get("table_id")),
        _normalize_key_value(metadata.get("image_id")),
    )


def _build_text_index(documents: Iterable[Any] | None = None) -> dict[str, str]:
    """Map stable point IDs to source text strings."""

    text_by_point_id: dict[str, str] = {}
    source_documents = documents if documents is not None else iter_all_documents()
    for document in source_documents:
        text = getattr(document, "text", None)
        metadata = getattr(document, "metadata", None) or {}
        if not isinstance(metadata, dict):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        text_by_point_id["|".join(_chunk_lookup_key(metadata))] = text
    return text_by_point_id


def _scroll_collection(client: QdrantClient, *, limit: int) -> Iterable[Any]:
    """Stream all Qdrant records from the target collection."""

    offset: Any = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break
        for record in records:
            yield record
        if offset is None:
            break


def _payload_text(payload: Any) -> str | None:
    """Extract an existing payload text field if one is already present."""

    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return None


def _payload_lookup_key(payload: Any) -> str | None:
    """Build a lookup key from a Qdrant point payload."""

    if not isinstance(payload, dict):
        return None
    metadata = {
        "source_key": payload.get("source_key"),
        "paper_id": payload.get("paper_id"),
        "section_id": payload.get("section_id"),
        "chunk_index": payload.get("chunk_index"),
        "source_field": payload.get("source_field"),
        "table_id": payload.get("table_id"),
        "image_id": payload.get("image_id"),
    }
    if not any(value not in (None, "", [], {}) for value in metadata.values()):
        return None
    return "|".join(_chunk_lookup_key(metadata))


def backfill_qdrant_text(
    *,
    client: QdrantClient | None = None,
    documents: Iterable[Any] | None = None,
    scroll_limit: int = 256,
    dry_run: bool = False,
) -> BackfillResult:
    """Patch missing `text` payloads for points in the Qdrant collection."""

    qdrant_client = client or _get_client()
    text_by_point_id = _build_text_index(documents)

    scanned_points = 0
    updated_points = 0
    skipped_with_text = 0
    skipped_without_source_text = 0

    for record in _scroll_collection(qdrant_client, limit=scroll_limit):
        scanned_points += 1
        payload = getattr(record, "payload", None) or {}
        point_id = str(getattr(record, "id", ""))

        if _payload_text(payload):
            skipped_with_text += 1
            continue

        lookup_key = _payload_lookup_key(payload)
        source_text = text_by_point_id.get(lookup_key or "")
        if not source_text:
            skipped_without_source_text += 1
            continue

        if not dry_run:
            qdrant_client.set_payload(
                collection_name=COLLECTION_NAME,
                points=[point_id],
                payload={"text": source_text},
                wait=True,
            )
        updated_points += 1

    return BackfillResult(
        scanned_points=scanned_points,
        updated_points=updated_points,
        skipped_with_text=skipped_with_text,
        skipped_without_source_text=skipped_without_source_text,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the backfill utility."""

    parser = argparse.ArgumentParser(
        prog="backfill_qdrant_text",
        description="Backfill missing Qdrant text payloads from the source corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be updated without writing to Qdrant.",
    )
    parser.add_argument(
        "--scroll-limit",
        type=int,
        default=256,
        help="Number of records to fetch per Qdrant scroll page.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = build_parser().parse_args(argv)
    result = backfill_qdrant_text(dry_run=args.dry_run, scroll_limit=args.scroll_limit)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("Qdrant text backfill")
        print(f"Scanned points: {result.scanned_points}")
        print(f"Updated points: {result.updated_points}")
        print(f"Skipped with text: {result.skipped_with_text}")
        print(f"Skipped without source text: {result.skipped_without_source_text}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    raise SystemExit(main())
