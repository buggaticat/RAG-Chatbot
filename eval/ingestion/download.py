"""Download the golden evaluation dataset JSON files from S3."""

from __future__ import annotations

import time
from pathlib import Path

import boto3

from .config import (
    GOLDEN_DATASET_DIR,
    GOLDEN_DATASET_FILES,
    S3_BUCKET_NAME,
    S3_PREFIX,
)

S3_READ_RETRIES = 3
S3_READ_RETRY_DELAY_SECONDS = 1.0


def _log(message: str) -> None:
    """Print a simple download progress message."""

    print(f"[golden-dataset-ingestion] {message}")


def _resolve_s3_key(filename: str) -> str:
    """Build the S3 key for one golden dataset file."""

    return f"{S3_PREFIX}{filename}"


def _read_s3_object_bytes(s3, key: str) -> bytes:
    """Read an S3 object with a small retry buffer for transient failures."""

    last_exc: Exception | None = None
    for attempt in range(1, S3_READ_RETRIES + 1):
        try:
            file_obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            return file_obj["Body"].read()
        except Exception as exc:  # pragma: no cover - transient/network retry path
            last_exc = exc
            if attempt < S3_READ_RETRIES:
                time.sleep(S3_READ_RETRY_DELAY_SECONDS * attempt)
                continue
            raise

    if last_exc is not None:  # pragma: no cover - defensive fallback
        raise last_exc
    raise RuntimeError(f"Failed to read S3 object: {key}")


def download_golden_dataset(output_dir: Path | None = None) -> list[Path]:
    """Download the golden dataset JSON files into a local directory."""

    if not S3_BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME must be set before downloading the golden dataset")

    target_dir = Path(output_dir) if output_dir is not None else GOLDEN_DATASET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    downloaded_files: list[Path] = []

    for filename in GOLDEN_DATASET_FILES:
        source_key = _resolve_s3_key(filename)
        destination = target_dir / filename
        _log(f"Downloading {source_key} -> {destination}")
        destination.write_bytes(_read_s3_object_bytes(s3, source_key))
        downloaded_files.append(destination)

    return downloaded_files


def main() -> None:
    """Download the golden dataset files to the local eval directory."""

    downloaded_files = download_golden_dataset()
    _log(f"Downloaded {len(downloaded_files)} file(s)")


if __name__ == "__main__":
    main()
