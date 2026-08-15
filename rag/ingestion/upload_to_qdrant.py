"""Upload embeddings into Qdrant and keep the collection in sync."""

import json
import hashlib
from pathlib import Path

from typing import List
from qdrant_client import QdrantClient, models

from .embed import documents_to_embeddings
from .config import (
    BATCH_SIZE,
    COLLECTION_NAME,
    INGESTION_STATE_PATH,
    QDRANT_APIKEY,
    QDRANT_CLUSTER_ENDPOINT,
    S3_BUCKET_NAME,
    S3_PREFIX,
)

client = QdrantClient(
    url=QDRANT_CLUSTER_ENDPOINT,
    api_key=QDRANT_APIKEY
)

def _ensure_collection(vector_size: int) -> None:
    """Create the Qdrant collection if it does not already exist."""

    existing_collections = {
        collection.name for collection in client.get_collections().collections
    }

    if COLLECTION_NAME in existing_collections:
        return

    client.create_collection(
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
    return json.loads(INGESTION_STATE_PATH.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    """Persist the ingestion bookkeeping state to disk."""

    INGESTION_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _delete_by_source_key(source_key: str) -> None:
    """Delete all points previously indexed for a specific source key."""

    client.delete(
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

    state = _load_state()
    all_embedding = documents_to_embeddings()
    Point = models.PointStruct

    if not all_embedding:
        return

    _ensure_collection(vector_size=len(all_embedding[0][1]))

    current_keys = _list_current_s3_keys()
    indexed_hashes = state.get("source_hashes", {})
    current_hashes = {}

    points: List[models.PointStruct] = []
    for _, embedding, metadata in all_embedding:
        source_key = metadata.get("source_key")
        source_hash = metadata.get("source_hash")
        if source_key and source_key not in current_keys:
            continue
        if source_key:
            current_hashes[source_key] = source_hash
            if indexed_hashes.get(source_key) == source_hash:
                continue
            if indexed_hashes.get(source_key) and indexed_hashes.get(source_key) != source_hash:
                _delete_by_source_key(source_key)

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
        stable_id = hashlib.sha256(stable_id_source.encode("utf-8")).hexdigest()
        point = Point(
            id=stable_id,
            vector=embedding,
            payload=metadata,
        )
        points.append(point)

        if len(points) == BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            points = []

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    deleted_keys = set(indexed_hashes) - current_keys
    for source_key in deleted_keys:
        _delete_by_source_key(source_key)
        indexed_hashes.pop(source_key, None)

    indexed_hashes.update(current_hashes)
    state["source_hashes"] = indexed_hashes
    _save_state(state)


if __name__ == "__main__":
    main()
    
