"""Tests for Qdrant upload synchronization behavior."""

import importlib
import json
import sys
import types


def _import_module(
    monkeypatch,
    documents,
    s3_keys,
    state=None,
    checkpoint_embeddings=None,
    next_task_index=0,
    upsert_side_effect=None,
):
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
            if callable(upsert_side_effect):
                upsert_side_effect(collection_name, list(points))
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
    fake_embed.iter_document_embedding_batches = lambda checkpoint=None: iter(
        (index, [record]) for index, record in enumerate(documents, start=1)
    )
    fake_embed.get_checkpoint_embeddings = lambda checkpoint=None: list(checkpoint_embeddings or [])
    fake_embed._load_checkpoint = lambda: {
        "next_task_index": next_task_index,
        "embeddings": list(checkpoint_embeddings or []),
    }
    fake_embed._save_progress_checkpoint = lambda next_task_index: None
    fake_embed._clear_checkpoint = lambda: None

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


def test_main_deletes_stale_sources_when_no_new_embeddings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module, fake_upserts, fake_deleted = _import_module(
        monkeypatch,
        documents=[],
        s3_keys=[],
        state={"source_hashes": {"stale.json": "old-hash"}},
    )

    module.main()

    assert fake_upserts == []
    assert len(fake_deleted) == 1
    assert fake_deleted[0][0] == module.COLLECTION_NAME
    saved = json.loads(module.INGESTION_STATE_PATH.read_text(encoding="utf-8"))
    assert saved["source_hashes"] == {}


def test_main_upserts_new_embeddings_and_persists_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    documents = [
        (
            "text-a",
            [1.0, 0.0],
            {
                "paper_id": "p1",
                "section_id": 1,
                "chunk_index": 0,
                "source_field": "abstract",
                "embedding_model": "m",
                "embedding_version": "v",
                "source_key": "a.json",
                "source_hash": "hash-a",
            },
        ),
    ]
    module, fake_upserts, fake_deleted = _import_module(
        monkeypatch,
        documents,
        s3_keys=["a.json"],
        state={"source_hashes": {}},
    )

    module.main()

    assert len(fake_upserts) == 1
    assert fake_deleted == []
    assert fake_upserts[0][0] == module.COLLECTION_NAME
    assert fake_upserts[0][1][0]["payload"]["source_key"] == "a.json"
    assert fake_upserts[0][1][0]["payload"]["text"] == "text-a"
    saved = json.loads(module.INGESTION_STATE_PATH.read_text(encoding="utf-8"))
    assert saved["source_hashes"] == {"a.json": "hash-a"}


def test_main_checkpoints_state_during_batch_uploads(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    documents = [
        (
            "text-a",
            [1.0, 0.0],
            {
                "paper_id": "p1",
                "section_id": 1,
                "chunk_index": 0,
                "source_field": "abstract",
                "embedding_model": "m",
                "embedding_version": "v",
                "source_key": "a.json",
                "source_hash": "hash-a",
            },
        ),
        (
            "text-b",
            [0.0, 1.0],
            {
                "paper_id": "p2",
                "section_id": 2,
                "chunk_index": 0,
                "source_field": "abstract",
                "embedding_model": "m",
                "embedding_version": "v",
                "source_key": "b.json",
                "source_hash": "hash-b",
            },
        ),
    ]
    module, fake_upserts, fake_deleted = _import_module(
        monkeypatch,
        documents,
        s3_keys=["a.json", "b.json"],
        state={"source_hashes": {}},
    )
    monkeypatch.setattr(module, "BATCH_SIZE", 1)

    save_calls = []
    original_save_state = module._save_state

    def tracking_save_state(state):
        save_calls.append(dict(state.get("source_hashes", {})))
        original_save_state(state)

    monkeypatch.setattr(module, "_save_state", tracking_save_state)

    module.main()

    assert len(fake_upserts) == 2
    assert fake_deleted == []
    assert len(save_calls) >= 3
    assert save_calls[0] == {"a.json": "hash-a"}
    assert save_calls[-1] == {"a.json": "hash-a", "b.json": "hash-b"}


def test_main_splits_large_upserts_into_smaller_sub_batches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    documents = []
    for index in range(3):
        documents.append(
            (
                f"text-{index}",
                [float(index), float(index + 1)],
                {
                    "paper_id": f"p{index}",
                    "section_id": index,
                    "chunk_index": 0,
                    "source_field": "abstract",
                    "embedding_model": "m",
                    "embedding_version": "v",
                    "source_key": f"{index}.json",
                    "source_hash": f"hash-{index}",
                },
            )
        )

    module, fake_upserts, fake_deleted = _import_module(
        monkeypatch,
        documents,
        s3_keys=["0.json", "1.json", "2.json"],
        state={"source_hashes": {}},
    )
    monkeypatch.setattr(module, "BATCH_SIZE", 3)
    monkeypatch.setattr(module, "QDRANT_UPSERT_BATCH_SIZE", 1)

    module.main()

    assert len(fake_upserts) == 3
    assert [len(batch[1]) for batch in fake_upserts] == [1, 1, 1]
    assert fake_deleted == []


def test_main_retries_timeout_style_upserts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    documents = [
        (
            "text-a",
            [1.0, 0.0],
            {
                "paper_id": "p1",
                "section_id": 1,
                "chunk_index": 0,
                "source_field": "abstract",
                "embedding_model": "m",
                "embedding_version": "v",
                "source_key": "a.json",
                "source_hash": "hash-a",
            },
        ),
    ]
    attempts = {"count": 0}

    def upsert_side_effect(collection_name, points):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("The write operation timed out")

    module, fake_upserts, fake_deleted = _import_module(
        monkeypatch,
        documents,
        s3_keys=["a.json"],
        state={"source_hashes": {}},
        upsert_side_effect=upsert_side_effect,
    )
    monkeypatch.setattr(module, "QDRANT_UPSERT_MAX_RETRIES", 2)
    monkeypatch.setattr(module, "QDRANT_UPSERT_BATCH_SIZE", 1)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    module.main()

    assert attempts["count"] == 2
    assert len(fake_upserts) == 1
    assert len(fake_upserts[0][1]) == 1
    assert fake_deleted == []


def test_main_drains_checkpoint_embeddings_before_streaming(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    checkpoint_embeddings = [
        (
            "saved-text",
            [9.0, 9.0],
            {
                "paper_id": "p0",
                "section_id": 0,
                "chunk_index": 0,
                "source_field": "abstract",
                "embedding_model": "m",
                "embedding_version": "v",
                "source_key": "saved.json",
                "source_hash": "saved-hash",
            },
        )
    ]
    documents = [
        (
            "new-text",
            [1.0, 0.0],
            {
                "paper_id": "p1",
                "section_id": 1,
                "chunk_index": 0,
                "source_field": "abstract",
                "embedding_model": "m",
                "embedding_version": "v",
                "source_key": "a.json",
                "source_hash": "hash-a",
            },
        ),
    ]
    module, fake_upserts, fake_deleted = _import_module(
        monkeypatch,
        documents,
        s3_keys=["saved.json", "a.json"],
        state={"source_hashes": {}},
        checkpoint_embeddings=checkpoint_embeddings,
        next_task_index=1,
    )

    module.main()

    assert len(fake_upserts) == 2
    assert fake_upserts[0][1][0]["payload"]["source_key"] == "saved.json"
    assert fake_upserts[1][1][0]["payload"]["source_key"] == "a.json"
    assert fake_deleted == []


def test_load_state_recovers_from_corrupted_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module, _, _ = _import_module(
        monkeypatch,
        documents=[],
        s3_keys=[],
        state=None,
    )
    module.INGESTION_STATE_PATH.write_text("{not valid json", encoding="utf-8")

    assert module._load_state() == {"source_hashes": {}}
