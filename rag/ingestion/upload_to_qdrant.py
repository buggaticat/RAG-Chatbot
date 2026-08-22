"""Upload embeddings into Qdrant and keep the collection in sync."""

from __future__ import annotations

import json
import hashlib
import time
import uuid
from pathlib import Path

from typing import List
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from eval.embedding_and_index_health import EmbeddingSnapshot, IndexEventLedger
from eval.embedding_and_index_health.metrics import DEFAULT_EMBEDDING_DRIFT_PATH

from .embed import (
    _clear_checkpoint,
    _load_checkpoint,
    _save_progress_checkpoint,
    get_checkpoint_embeddings,
    iter_document_embedding_batches,
)
from .config import (
    BATCH_SIZE,
    COLLECTION_NAME,
    INGESTION_STATE_PATH,
    QDRANT_APIKEY,
    QDRANT_CLUSTER_ENDPOINT,
    QDRANT_REQUEST_TIMEOUT_SECONDS,
    QDRANT_UPSERT_BATCH_SIZE,
    QDRANT_UPSERT_MAX_RETRIES,
    QDRANT_UPSERT_RETRY_BACKOFF_SECONDS,
    S3_BUCKET_NAME,
    S3_PREFIX,
)

client: QdrantClient | None = None
index_event_ledger = IndexEventLedger()


def _record_index_event(stage: str, success: bool, exc: Exception | None = None, metadata: dict | None = None) -> None:
    """Persist a structured index event for health evaluation."""

    index_event_ledger.record_event(
        "qdrant_ingestion",
        success=success,
        stage=stage,
        message=str(exc) if exc else None,
        error_type=exc.__class__.__name__ if exc else None,
        metadata=metadata or {},
    )


def _get_client() -> QdrantClient:
    """Create or reuse the Qdrant client lazily."""

    global client

    if client is None:
        client = QdrantClient(
            url=QDRANT_CLUSTER_ENDPOINT,
            api_key=QDRANT_APIKEY,
            timeout=QDRANT_REQUEST_TIMEOUT_SECONDS,
        )
    return client


def _log(message: str) -> None:
    """Print a simple ingestion progress message."""

    print(f"[qdrant-ingestion] {message}")


def _save_embedding_snapshot(vectors: list[list[float]]) -> None:
    """Persist the latest embedding snapshot so eval can report drift later."""

    snapshot = EmbeddingSnapshot.from_vectors(vectors, label="current")
    payload = {"current": snapshot.to_dict()}
    existing = {}
    if DEFAULT_EMBEDDING_DRIFT_PATH.exists():
        try:
            existing_payload = json.loads(DEFAULT_EMBEDDING_DRIFT_PATH.read_text(encoding="utf-8"))
            if isinstance(existing_payload, dict):
                existing = {key: value for key, value in existing_payload.items() if key != "current"}
        except json.JSONDecodeError:
            existing = {}
    existing.update(payload)
    DEFAULT_EMBEDDING_DRIFT_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def _chunked(items, size: int):
    """Yield fixed-size slices from an in-memory sequence."""

    for start in range(0, len(items), size):
        yield items[start : start + size]


def _is_retryable_qdrant_error(exc: Exception) -> bool:
    """Detect timeout-style failures that are worth retrying."""

    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if "timeout" in name:
        return True
    return "timed out" in message or "timeout" in message


def _upsert_points(points: list[models.PointStruct], *, batch_label: str) -> None:
    """Upsert a set of points into Qdrant with retries and smaller transport batches."""

    qdrant_client = _get_client()
    total_batches = list(_chunked(points, max(1, QDRANT_UPSERT_BATCH_SIZE)))
    for batch_index, batch_points in enumerate(total_batches, start=1):
        attempt = 0
        while True:
            try:
                _log(
                    f"Uploading {batch_label} sub-batch {batch_index}/{len(total_batches)} "
                    f"with {len(batch_points)} vector(s)"
                )
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch_points,
                )
                _log(f"{batch_label.capitalize()} sub-batch {batch_index} uploaded successfully")
                break
            except Exception as exc:
                attempt += 1
                if attempt >= max(1, QDRANT_UPSERT_MAX_RETRIES) or not _is_retryable_qdrant_error(exc):
                    raise
                delay_s = QDRANT_UPSERT_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                _log(
                    f"{batch_label.capitalize()} sub-batch {batch_index} hit a timeout "
                    f"({attempt}/{QDRANT_UPSERT_MAX_RETRIES}); retrying in {delay_s:.1f}s"
                )
                time.sleep(delay_s)


def _prepare_points(
    embeddings,
    *,
    current_keys: set[str],
    indexed_hashes: dict[str, str],
    current_hashes: dict[str, str],
    deleted_source_keys: set[str],
):
    """Convert embeddings into Qdrant points while applying sync filters."""

    points: List[models.PointStruct] = []
    skipped_unchanged = 0
    skipped_missing_source = 0
    deleted_replaced = 0
    processed_embeddings = 0

    for content, embedding, metadata in embeddings:
        processed_embeddings += 1
        source_key = metadata.get("source_key")
        source_hash = metadata.get("source_hash")
        if source_key and source_key not in current_keys:
            skipped_missing_source += 1
            continue
        if source_key:
            current_hashes[source_key] = source_hash
            if indexed_hashes.get(source_key) == source_hash:
                skipped_unchanged += 1
                continue
            if (
                indexed_hashes.get(source_key)
                and indexed_hashes.get(source_key) != source_hash
                and source_key not in deleted_source_keys
            ):
                deleted_replaced += 1
                _delete_by_source_key(source_key)
                deleted_source_keys.add(source_key)

        stable_id_source = "|".join(
            [
                str(metadata.get("paper_id", "")),
                str(metadata.get("section_id", "")),
                str(metadata.get("chunk_index", "")),
                str(metadata.get("source_field", "")),
                str(metadata.get("embedding_model", "")),
                str(metadata.get("embedding_version", "")),
                str(metadata.get("table_id", "")),
                str(metadata.get("image_id", "")),
            ]
        )
        stable_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_id_source))
        points.append(
            models.PointStruct(
                id=stable_id,
                vector=embedding,
                payload={
                    **metadata,
                    "text": content,
                },
            )
        )

    return points, {
        "processed_embeddings": processed_embeddings,
        "skipped_unchanged": skipped_unchanged,
        "skipped_missing_source": skipped_missing_source,
        "deleted_replaced": deleted_replaced,
    }


def _point_vector_size(point) -> int:
    """Return the vector width for either a real Qdrant point or a test double."""

    vector = point["vector"] if isinstance(point, dict) else point.vector
    return len(vector)

def _ensure_collection(vector_size: int) -> None:
    """Create the Qdrant collection if it does not already exist."""

    qdrant_client = _get_client()

    existing_collections = {
        collection.name for collection in qdrant_client.get_collections().collections
    }

    if COLLECTION_NAME in existing_collections:
        return

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def _load_state() -> dict:
    """Load the ingestion bookkeeping state from disk."""

    if not INGESTION_STATE_PATH.exists():
        return {"source_hashes": {}}
    try:
        return json.loads(INGESTION_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"source_hashes": {}}


def _save_state(state: dict) -> None:
    """Persist the ingestion bookkeeping state to disk."""

    INGESTION_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _delete_by_source_key(source_key: str) -> None:
    """Delete all points previously indexed for a specific source key."""

    _get_client().delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_key",
                        match=models.MatchValue(value=source_key),
                    )
                ]
            )
        ),
    )


def _list_current_s3_keys() -> set[str]:
    """List the current S3 keys under the configured corpus prefix."""

    import boto3

    s3 = boto3.client("s3")
    keys: set[str] = set()
    continuation_token = None

    while True:
        request_kwargs = {"Bucket": S3_BUCKET_NAME, "Prefix": S3_PREFIX}
        if continuation_token:
            request_kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**request_kwargs)
        for obj in response.get("Contents", []):
            keys.add(obj["Key"])
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")

    return keys

def main() -> None:
    """Sync the current embedding set into Qdrant."""

    try:
        _log(f"Starting sync into collection '{COLLECTION_NAME}'")
        _log("Loading ingestion state")
        state = _load_state()
        _log("Listing current S3 keys")
        current_keys = _list_current_s3_keys()
        _log(f"Found {len(current_keys)} S3 objects under prefix '{S3_PREFIX}'")
        indexed_hashes = state.get("source_hashes", {})
        current_hashes = {}
        upserted_points = 0

        _log("Connecting to Qdrant")
        qdrant_client = _get_client()

        checkpoint = _load_checkpoint()
        checkpoint_embeddings = get_checkpoint_embeddings(checkpoint)
        embedding_vectors = [vector for _, vector, _ in checkpoint_embeddings]
        next_task_index = int(checkpoint.get("next_task_index", 0) or 0)
        collection_ready = False
        batch_number = 0
        deleted_source_keys: set[str] = set()

        def commit_state() -> None:
            indexed_hashes.update(current_hashes)
            state["source_hashes"] = indexed_hashes
            _save_state(state)

        if checkpoint_embeddings:
            _log(
                f"Draining {len(checkpoint_embeddings)} saved embedding(s) "
                "from the local checkpoint"
            )
            for batch in _chunked(checkpoint_embeddings, BATCH_SIZE):
                points, stats = _prepare_points(
                    batch,
                    current_keys=current_keys,
                    indexed_hashes=indexed_hashes,
                    current_hashes=current_hashes,
                    deleted_source_keys=deleted_source_keys,
                )
                if points:
                    if not collection_ready:
                        _ensure_collection(vector_size=_point_vector_size(points[0]))
                        collection_ready = True
                    batch_number += 1
                    _log(
                        f"Uploading checkpoint batch {batch_number} with {len(points)} vector(s) "
                        f"after processing {stats['processed_embeddings']} embedding(s)"
                    )
                _upsert_points(points, batch_label=f"checkpoint batch {batch_number}")
                upserted_points += len(points)
            commit_state()
            _save_progress_checkpoint(next_task_index)
        else:
            _log("No saved embeddings were found in the local checkpoint")

        _log("Streaming new embedding batches from the corpus")
        streamed_embeddings = 0
        skipped_unchanged = 0
        skipped_missing_source = 0
        deleted_replaced = 0

        for batch_next_index, batch_records in iter_document_embedding_batches(checkpoint):
            streamed_embeddings += len(batch_records)
            embedding_vectors.extend(vector for _, vector, _ in batch_records)
            points, stats = _prepare_points(
                batch_records,
                current_keys=current_keys,
                indexed_hashes=indexed_hashes,
                current_hashes=current_hashes,
                deleted_source_keys=deleted_source_keys,
            )
            skipped_unchanged += stats["skipped_unchanged"]
            skipped_missing_source += stats["skipped_missing_source"]
            deleted_replaced += stats["deleted_replaced"]

            if points:
                if not collection_ready:
                    _ensure_collection(vector_size=_point_vector_size(points[0]))
                    collection_ready = True
                batch_number += 1
                _log(
                    f"Uploading streamed batch {batch_number} with {len(points)} vector(s) "
                    f"after embedding cursor {batch_next_index}"
                )
                _upsert_points(points, batch_label=f"streamed batch {batch_number}")
                upserted_points += len(points)
            commit_state()
            _save_progress_checkpoint(batch_next_index)

        if not collection_ready:
            _log("No embeddings were produced, skipping collection creation and upsert")
        else:
            _log(
                "Upsert complete "
                f"(inserted {upserted_points} points, "
                f"skipped {skipped_unchanged} unchanged, "
                f"skipped {skipped_missing_source} missing-source, "
                f"replaced {deleted_replaced} stale source sets)"
            )

        deleted_keys = set(indexed_hashes) - current_keys
        if deleted_keys:
            _log(f"Deleting {len(deleted_keys)} stale source key(s) from Qdrant")
        for source_key in deleted_keys:
            _delete_by_source_key(source_key)
            indexed_hashes.pop(source_key, None)

        indexed_hashes.update(current_hashes)
        _log("Saving ingestion state")
        state["source_hashes"] = indexed_hashes
        _save_state(state)
        if embedding_vectors:
            _save_embedding_snapshot(embedding_vectors)
        _clear_checkpoint()
        _log(
            "Summary: "
            f"{len(current_keys)} source object(s) scanned, "
            f"{streamed_embeddings + len(checkpoint_embeddings)} embedding(s) produced, "
            f"{upserted_points} vector(s) uploaded"
        )
        _log("Sync finished successfully")
    except Exception as exc:
        _log(f"Sync failed: {exc}")
        _record_index_event("sync", False, exc, {"collection_name": COLLECTION_NAME})
        raise
    else:
        _record_index_event("sync", True, metadata={"collection_name": COLLECTION_NAME})


if __name__ == "__main__":
    main()
    
