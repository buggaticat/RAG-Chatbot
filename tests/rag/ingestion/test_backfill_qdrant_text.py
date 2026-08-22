from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace


def _import_module(monkeypatch):
    calls = {"set_payload": []}

    class FakeClient:
        def __init__(self):
            self.records = []

        def scroll(self, **kwargs):
            if not self.records:
                return [], None
            records = self.records
            self.records = []
            return records, None

        def set_payload(self, **kwargs):
            calls["set_payload"].append(kwargs)
            return SimpleNamespace(status="ok")

    fake_qdrant = types.ModuleType("qdrant_client")
    fake_qdrant.QdrantClient = lambda **kwargs: FakeClient()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service_name: SimpleNamespace()

    fake_llama_index = types.ModuleType("llama_index")
    fake_llama_core = types.ModuleType("llama_index.core")
    fake_llama_core.Document = lambda *args, **kwargs: SimpleNamespace(*args, **kwargs)
    fake_node_parser = types.ModuleType("llama_index.core.node_parser")
    fake_node_parser.SentenceSplitter = lambda *args, **kwargs: SimpleNamespace(split_text=lambda text: [text])

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "llama_index", fake_llama_index)
    monkeypatch.setitem(sys.modules, "llama_index.core", fake_llama_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", fake_node_parser)
    sys.modules.pop("rag.ingestion.backfill_qdrant_text", None)
    module = importlib.import_module("rag.ingestion.backfill_qdrant_text")
    return module, calls, FakeClient


def test_backfill_updates_missing_text_payloads(monkeypatch):
    module, calls, FakeClient = _import_module(monkeypatch)

    metadata = {
        "paper_id": "p1",
        "section_id": 1,
        "chunk_index": 0,
        "source_field": "abstract",
        "embedding_model": "m",
        "embedding_version": "v",
        "source_key": "a.json",
        "source_hash": "hash-a",
    }
    fake_client = FakeClient()
    fake_client.records = [SimpleNamespace(id="point-1", payload=dict(metadata))]

    documents = [SimpleNamespace(text="hello world", metadata=metadata)]
    result = module.backfill_qdrant_text(client=fake_client, documents=documents)

    assert result.scanned_points == 1
    assert result.updated_points == 1
    assert result.skipped_with_text == 0
    assert result.skipped_without_source_text == 0
    assert calls["set_payload"][0]["points"] == ["point-1"]
    assert calls["set_payload"][0]["payload"] == {"text": "hello world"}


def test_backfill_dry_run_skips_writes(monkeypatch):
    module, calls, FakeClient = _import_module(monkeypatch)

    metadata = {
        "paper_id": "p1",
        "section_id": 1,
        "chunk_index": 0,
        "source_field": "abstract",
        "embedding_model": "m",
        "embedding_version": "v",
        "source_key": "a.json",
        "source_hash": "hash-a",
    }
    fake_client = FakeClient()
    fake_client.records = [SimpleNamespace(id="point-1", payload=dict(metadata))]

    documents = [SimpleNamespace(text="hello world", metadata=metadata)]
    result = module.backfill_qdrant_text(client=fake_client, documents=documents, dry_run=True)

    assert result.updated_points == 1
    assert calls["set_payload"] == []
