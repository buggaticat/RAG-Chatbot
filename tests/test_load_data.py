"""Tests for the dataset download helper script."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _import_load_data(monkeypatch, *, bucket_name: str | None = "bucket-name", hf_token: str | None = "token"):
    fake_boto3 = types.ModuleType("boto3")
    fake_s3 = types.SimpleNamespace(upload_file=lambda *args, **kwargs: None)
    fake_boto3.client = lambda service_name: fake_s3
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda **kwargs: "C:/tmp/dataset"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    monkeypatch.setenv("S3_BUCKET_NAME", bucket_name or "")
    monkeypatch.setenv("HF_TOKEN", hf_token or "")
    sys.modules.pop("load_data", None)
    return importlib.import_module("load_data")


def test_main_requires_a_real_bucket_name(monkeypatch):
    load_data = _import_load_data(monkeypatch, bucket_name=None)

    with pytest.raises(ValueError, match="Set S3_BUCKET_NAME before running this script."):
        load_data.main()
