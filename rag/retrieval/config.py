"""Retrieval-specific configuration for the persisted vector store."""

import os


QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
COLLECTION_NAME = "embedding_collection"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-5.4-mini")
FASTEMBED_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_TOP_K = 5
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
