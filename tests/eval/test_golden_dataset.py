"""Tests for downloading the golden eval dataset from S3."""

from __future__ import annotations

import importlib
import sys
import types
from io import BytesIO


def _import_module(monkeypatch):
    fake_s3_client = types.SimpleNamespace()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service_name: fake_s3_client
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    for module_name in [
        "eval.ingestion.download",
        "eval.ingestion",
        "eval",
    ]:
        sys.modules.pop(module_name, None)

    module = importlib.import_module("eval.ingestion.download")
    return module, fake_s3_client


def test_download_golden_dataset_fetches_only_expected_files(monkeypatch, tmp_path):
    module, fake_s3_client = _import_module(monkeypatch)

    requested_keys = []
    payloads = {
        "datasets/vectara/open_ragbench/pdf/arxiv/qrels.json": b'{"qrels": []}',
        "datasets/vectara/open_ragbench/pdf/arxiv/queries.json": b'{"queries": []}',
        "datasets/vectara/open_ragbench/pdf/arxiv/answers.json": b'{"answers": []}',
    }

    def get_object(Bucket, Key):
        requested_keys.append((Bucket, Key))
        return {"Body": BytesIO(payloads[Key])}

    fake_s3_client.get_object = get_object
    monkeypatch.setattr(module, "S3_BUCKET_NAME", "bucket-name")

    downloaded = module.download_golden_dataset(output_dir=tmp_path)

    assert [path.name for path in downloaded] == ["qrels.json", "queries.json", "answers.json"]
    assert [path.read_text(encoding="utf-8") for path in downloaded] == [
        '{"qrels": []}',
        '{"queries": []}',
        '{"answers": []}',
    ]
    assert requested_keys == [
        ("bucket-name", "datasets/vectara/open_ragbench/pdf/arxiv/qrels.json"),
        ("bucket-name", "datasets/vectara/open_ragbench/pdf/arxiv/queries.json"),
        ("bucket-name", "datasets/vectara/open_ragbench/pdf/arxiv/answers.json"),
    ]


def test_download_golden_dataset_uses_local_eval_directory_by_default(monkeypatch, tmp_path):
    module, fake_s3_client = _import_module(monkeypatch)

    payload = b"{}"
    fake_s3_client.get_object = lambda Bucket, Key: {"Body": BytesIO(payload)}
    monkeypatch.setattr(module, "S3_BUCKET_NAME", "bucket-name")
    monkeypatch.setattr(module, "GOLDEN_DATASET_DIR", tmp_path)

    downloaded = module.download_golden_dataset()

    assert downloaded == [
        tmp_path / "qrels.json",
        tmp_path / "queries.json",
        tmp_path / "answers.json",
    ]
    assert all(path.exists() for path in downloaded)
