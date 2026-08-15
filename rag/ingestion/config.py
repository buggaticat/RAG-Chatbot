"""Ingestion-specific configuration for corpus building and embedding."""

import os
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter


S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "open_ragbench/pdf/arxiv/corpus/"

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
BATCH_SIZE = 64
INGESTION_STATE_PATH = Path(os.getenv("INGESTION_STATE_PATH", ".ingestion_state.json"))
