"""Tests for Qdrant upload synchronization behavior."""

import importlib
import json
import sys
import types


def _import_module(monkeypatch, documents, s3_keys, state=None):
    fake_points = []
    fake_deleted = []
    fake_upserts = []

    class FakeClient:
        def get_collections(self):
            return types.SimpleNamespace(collections=[])

        def create_collection(self, **kwargs):
            return None

        def delete(self, collection_name, points_selector):
            fake_deleted.append((collection_name, points_selector))

        def upsert(self, collection_name, points):
            fake_upserts.append((collection_name, list(points)))

    fake_qdrant = types.ModuleType("qdrant_client")
    fake_qdrant.QdrantClient = lambda **kwargs: FakeClient()
    fake_models = types.SimpleNamespace(
        VectorParams=lambda **kwargs: kwargs,
        Distance=types.SimpleNamespace(COSINE="cosine"),
        PointStruct=lambda **kwargs: kwargs,
        FieldCondition=lambda **kwargs: kwargs,
        MatchValue=lambda **kwargs: kwargs,
        Filter=lambda **kwargs: kwargs,
        FilterSelector=lambda **kwargs: kwargs,
    )
    fake_qdrant.models = fake_models

    fake_embed = types.ModuleType("rag.ingestion.embed")
    fake_embed.documents_to_embeddings = lambda: documents

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service_name: types.SimpleNamespace(
        list_objects_v2=lambda **kwargs: {"Contents": [{"Key": key} for key in s3_keys], "IsTruncated": False}
    )

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant)
    monkeypatch.setitem(sys.modules, "rag.ingestion.embed", fake_embed)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    sys.modules.pop("rag.ingestion.upload_to_qdrant", None)
    module = importlib.import_module("rag.ingestion.upload_to_qdrant")
    module.client = FakeClient()

    if state is not None:
        module.INGESTION_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")

    return module, fake_upserts, fake_deleted


def test_main_deletes_stale_sources_and_skips_unchanged(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    documents = [
        ("text-a", [1.0, 0.0], {"paper_id": "p1", "section_id": 1, "chunk_index": 0, "source_field": "abstract", "embedding_model": "m", "embedding_version": "v", "source_key": "a.json", "source_hash": "hash-a"}),
        ("text-b", [0.0, 1.0], {"paper_id": "p2", "section_id": 2, "chunk_index": 0, "source_field": "abstract", "embedding_model": "m", "embedding_version": "v", "source_key": "b.json", "source_hash": "hash-b"}),
    ]
    module, fake_upserts, fake_deleted = _import_module(
        monkeypatch,
        documents,
        s3_keys=["a.json"],
        state={"source_hashes": {"a.json": "hash-a", "b.json": "old-hash-b"}},
    )

    module.main()

    assert fake_upserts == []
    assert len(fake_deleted) == 1
    assert fake_deleted[0][0] == module.COLLECTION_NAME
    saved = json.loads(module.INGESTION_STATE_PATH.read_text(encoding="utf-8"))
    assert saved["source_hashes"] == {"a.json": "hash-a"}
