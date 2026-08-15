"""Retrieval-specific configuration for the persisted vector store."""

import os


QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
COLLECTION_NAME = "embedding_collection"
FASTEMBED_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_TOP_K = 5
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
