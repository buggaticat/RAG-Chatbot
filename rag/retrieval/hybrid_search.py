"""Run hybrid retrieval over the persisted Qdrant collection."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex
from .config import (
    COLLECTION_NAME,
    DEFAULT_RERANK_MODEL,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    QDRANT_APIKEY,
    QDRANT_CLUSTER_ENDPOINT,
)


def _build_llm() -> Any:
    """Create the LLM used by the query engine when the dependency is available."""

    try:
        from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
    except Exception:
        return None

    try:
        return LlamaIndexOpenAI(model=LLM_MODEL_NAME)
    except TypeError:
        return LlamaIndexOpenAI()


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
    """Normalize a filter spec into a value and enabled flag."""

    if isinstance(raw_spec, Mapping):
        return raw_spec.get("value"), bool(raw_spec.get("enabled", True))
    return raw_spec


def _metadata_at_path(metadata: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted metadata path into a nested value."""

    value: Any = metadata
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return None
    return value


def _contains_value(candidate: Any, expected: Any) -> bool:
    """Check whether a candidate value contains or equals the expected value."""

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
    """Determine whether a document satisfies the configured metadata filters."""

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
    """Filter a document list down to the documents that match metadata rules."""

    if not metadata_filters:
        return documents
    return [document for document in documents if _document_matches_filters(document, metadata_filters)]


def _build_embed_model() -> Any:
    """Create the dense query embedder used by retrieval."""

    from llama_index.embeddings.openai import OpenAIEmbedding

    try:
        return OpenAIEmbedding(model=EMBEDDING_MODEL_NAME)
    except TypeError:
        return OpenAIEmbedding()


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Build and cache the Qdrant client used for retrieval."""

    return QdrantClient(
        url=QDRANT_CLUSTER_ENDPOINT,
        api_key=QDRANT_APIKEY,
    )


@lru_cache(maxsize=1)
def _get_embed_model() -> Any:
    """Build and cache the query embedder used by retrieval."""

    return _build_embed_model()


@lru_cache(maxsize=1)
def _get_vector_store() -> QdrantVectorStore:
    """Build and cache the vector store wrapper used for retrieval."""

    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=COLLECTION_NAME,
    )


@lru_cache(maxsize=1)
def _get_index() -> VectorStoreIndex:
    """Build and cache the vector index used for retrieval."""

    return VectorStoreIndex.from_vector_store(
        _get_vector_store(),
        embed_model=_get_embed_model(),
    )


def run_hybrid_search(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    rerank_top_n: int | None = None,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    metadata_filters: Mapping[str, tuple[Any, bool] | Mapping[str, Any]] | None = None,
) -> Any:
    """Query the existing hybrid Qdrant index and return the response."""

    if rerank_top_n is not None and rerank_top_n > top_k:
        raise ValueError("rerank_top_n must be less than or equal to top_k")

    node_postprocessors: list[Any] = []
    if rerank_top_n is not None and rerank_top_n > 0:
        node_postprocessors.append(
            SentenceTransformerRerank(
                model=rerank_model,
                top_n=rerank_top_n,
            )
        )

    query_engine = _get_index().as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=node_postprocessors,
        llm=_build_llm(),
    )

    response = query_engine.query(query)
    if metadata_filters:
        source_nodes = getattr(response, "source_nodes", None)
        if source_nodes is not None:
            filtered_nodes = _apply_metadata_filters(list(source_nodes), metadata_filters)
            try:
                response.source_nodes = filtered_nodes
            except Exception:
                pass

    return response


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
