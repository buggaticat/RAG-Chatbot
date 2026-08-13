from __future__ import annotations

from typing import Any
from collections.abc import Mapping

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex



SEARCHABLE_METADATA_KEYS: tuple[str, ...] = (
    "paper_id",
    "title",
    "authors",
    "categories",
    "updated",
    "published",
    "section_id",
    "section_title",
    "tables",
    "images",
    "source_field",
    "chunk_index",
    "chunking_strategy",
    "source_key",
    "source_hash",
    "embedding_model",
    "embedding_version",
    "preprocessing_hash",
    "embedded_at",
    "table_id",
    "image_id",
)


def _normalize_filter_spec(raw_spec: tuple[Any, bool] | Mapping[str, Any]) -> tuple[Any, bool]:
    if isinstance(raw_spec, Mapping):
        return raw_spec.get("value"), bool(raw_spec.get("enabled", True))
    return raw_spec


def _metadata_at_path(metadata: Mapping[str, Any], path: str) -> Any:
    value: Any = metadata
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return None
    return value


def _contains_value(candidate: Any, expected: Any) -> bool:
    if isinstance(candidate, Mapping):
        if isinstance(expected, Mapping):
            return all(_contains_value(candidate.get(key), value) for key, value in expected.items())
        return expected in candidate.values()

    if isinstance(candidate, list):
        if isinstance(expected, list):
            return all(any(_contains_value(item, value) for item in candidate) for value in expected)
        return any(_contains_value(item, expected) for item in candidate)

    return candidate == expected


def _document_matches_filters(document: Any, metadata_filters: Mapping[str, tuple[Any, bool] | Mapping[str, Any]] | None) -> bool:
    if not metadata_filters:
        return True

    metadata = getattr(document, "metadata", {}) or {}

    for key, raw_spec in metadata_filters.items():
        if "." not in key and key not in SEARCHABLE_METADATA_KEYS:
            continue

        value, enabled = _normalize_filter_spec(raw_spec)
        candidate = _metadata_at_path(metadata, key)
        matched = _contains_value(candidate, value)

        if enabled and not matched:
            return False
        if not enabled and matched:
            return False

    return True


def _apply_metadata_filters(
    documents: list[Any],
    metadata_filters: Mapping[str, tuple[Any, bool] | Mapping[str, Any]] | None,
) -> list[Any]:
    if not metadata_filters:
        return documents
    return [document for document in documents if _document_matches_filters(document, metadata_filters)]


def run_hybrid_search(
    query: str,
    *,
    data_dir: str = "./data",
    top_k: int = 5,
    rerank_top_n: int | None = None,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
    metadata_filters: Mapping[str, tuple[Any, bool] | Mapping[str, Any]] | None = None,
) -> Any:
    node_postprocessors: list[Any] = []
    if rerank_top_n is not None and rerank_top_n > 0:
        node_postprocessors.append(
            SentenceTransformerRerank(
                model=rerank_model,
                top_n=rerank_top_n,
            )
        )

    client = QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_hybrid_collection",
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = SimpleDirectoryReader(data_dir).load_data()
    documents = _apply_metadata_filters(documents, metadata_filters)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        vector_store_query_mode="hybrid",
        node_postprocessors=node_postprocessors,
    )

    return query_engine.query(query)


if __name__ == "__main__":
    response = run_hybrid_search(
        "What are the key details about hybrid search?",
        metadata_filters={
            "paper_id": {"value": "paper-1", "enabled": True},
            "source_field": {"value": "abstract", "enabled": True},
            "section_id": {"value": 2, "enabled": False},
        },
    )
    print(response)
