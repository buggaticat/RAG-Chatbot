"""Configuration for the golden dataset download helper."""

from __future__ import annotations

import os
from pathlib import Path

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "datasets/vectara/open_ragbench/pdf/arxiv/"
GOLDEN_DATASET_FILES = ("qrels.json", "queries.json", "answers.json")
GOLDEN_DATASET_DIR = Path(
    os.getenv("GOLDEN_DATASET_DIR", str(Path(__file__).resolve().parent.parent / "golden_dataset"))
)
