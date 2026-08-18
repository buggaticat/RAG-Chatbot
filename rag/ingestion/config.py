"""Ingestion-specific configuration for corpus building and embedding."""

import os
from pathlib import Path

try:
    from llama_index.core.node_parser import SentenceSplitter
except Exception:  # pragma: no cover - fallback for lean test environments
    class SentenceSplitter:  # type: ignore[override]
        """Minimal fallback splitter used when llama_index is unavailable."""

        def __init__(self, chunk_size: int = 0, chunk_overlap: int = 0) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> list[str]:
            """Return the input text as a single chunk."""

            return [text]


S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "datasets/vectara/open_ragbench/pdf/arxiv/corpus/"

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION", "2026-08-11")
BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
NORMALIZE_EMBEDDINGS = True
PREPROCESSING_HASH_SEED = "normalize_text:v1|section_aware_sentence_splitter:v1|blip_caption:v1"

ABSTRACT_SPLITTER = SentenceSplitter(chunk_size=240, chunk_overlap=24)
SECTION_SPLITTER = SentenceSplitter(chunk_size=280, chunk_overlap=28)

QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
COLLECTION_NAME = "embedding_collection"
BATCH_SIZE = 32
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "8"))
QDRANT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("QDRANT_REQUEST_TIMEOUT_SECONDS", "60"))
QDRANT_UPSERT_MAX_RETRIES = int(os.getenv("QDRANT_UPSERT_MAX_RETRIES", "3"))
QDRANT_UPSERT_RETRY_BACKOFF_SECONDS = float(os.getenv("QDRANT_UPSERT_RETRY_BACKOFF_SECONDS", "1.5"))
INGESTION_STATE_PATH = Path(os.getenv("INGESTION_STATE_PATH", ".ingestion_state.json"))
EMBEDDING_CHECKPOINT_PATH = Path(
    os.getenv("EMBEDDING_CHECKPOINT_PATH", ".embedding_checkpoint.json")
)
