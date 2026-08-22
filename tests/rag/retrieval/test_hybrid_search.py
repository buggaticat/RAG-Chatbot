"""Tests for hybrid retrieval filtering and reranking behavior."""

import importlib
import sys
import types


def _install_retrieval_stubs(monkeypatch, documents=None):
    documents = documents or []
    calls = {"queried": None}

    class FakeDocument:
        def __init__(self, metadata):
            self.metadata = metadata

    class FakeStorageContext:
        @classmethod
        def from_defaults(cls, vector_store=None):
            return {"vector_store": vector_store}

    class FakeQueryEngine:
        def query(self, query):
            calls["queried"] = query
            return types.SimpleNamespace(answer=f"response for {query}")

    class FakeIndex:
        @classmethod
        def from_vector_store(cls, vector_store, embed_model=None, **kwargs):
            calls["vector_store"] = vector_store
            calls["embed_model"] = embed_model
            calls["from_vector_store_kwargs"] = kwargs
            return cls()

        def as_query_engine(self, **kwargs):
            calls["query_engine_kwargs"] = kwargs
            return FakeQueryEngine()

    class FakeSentenceTransformerRerank:
        def __init__(self, **kwargs):
            calls["reranker_kwargs"] = kwargs

    class FakeQdrantVectorStore:
        def __init__(self, **kwargs):
            calls["vector_store_kwargs"] = kwargs

    fake_llama_core = types.ModuleType("llama_index.core")
    fake_llama_core.StorageContext = FakeStorageContext
    fake_llama_core.VectorStoreIndex = FakeIndex

    fake_postprocessor_module = types.ModuleType("llama_index.core.postprocessor")
    fake_postprocessor_module.SentenceTransformerRerank = FakeSentenceTransformerRerank

    fake_qdrant_module = types.ModuleType("llama_index.vector_stores.qdrant")
    fake_qdrant_module.QdrantVectorStore = FakeQdrantVectorStore

    fake_embeddings_module = types.ModuleType("llama_index.embeddings")
    fake_openai_module = types.ModuleType("llama_index.embeddings.openai")

    class FakeOpenAIEmbedding:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.model = kwargs.get("model", "text-embedding-3-large")

    fake_openai_module.OpenAIEmbedding = FakeOpenAIEmbedding

    fake_qdrant_client = types.ModuleType("qdrant_client")
    fake_qdrant_client.QdrantClient = lambda **kwargs: types.SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "llama_index", types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core", fake_llama_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.postprocessor", fake_postprocessor_module)
    monkeypatch.setitem(sys.modules, "llama_index.embeddings", fake_embeddings_module)
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.openai", fake_openai_module)
    monkeypatch.setitem(sys.modules, "llama_index.vector_stores", types.ModuleType("llama_index.vector_stores"))
    monkeypatch.setitem(sys.modules, "llama_index.vector_stores.qdrant", fake_qdrant_module)
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_client)

    sys.modules.pop("rag.retrieval.hybrid_search", None)
    module = importlib.import_module("rag.retrieval.hybrid_search")
    return module, calls


def test_metadata_path_supports_nested_lookup(monkeypatch):
    hybrid_search, _ = _install_retrieval_stubs(monkeypatch)

    metadata = {"tables": {"t1": "|a|b|"}, "images": {"i1": "base64"}}

    assert hybrid_search._metadata_at_path(metadata, "tables.t1") == "|a|b|"
    assert hybrid_search._metadata_at_path(metadata, "images.i1") == "base64"
    assert hybrid_search._metadata_at_path(metadata, "images.missing") is None


def test_contains_value_supports_scalars_lists_and_nested_dicts(monkeypatch):
    hybrid_search, _ = _install_retrieval_stubs(monkeypatch)

    assert hybrid_search._contains_value("paper-1", "paper-1") is True
    assert hybrid_search._contains_value(["Alice", "Bob"], "Alice") is True
    assert hybrid_search._contains_value(["Alice", "Bob"], "Carol") is False
    assert hybrid_search._contains_value({"t1": "|a|b|"}, {"t1": "|a|b|"}) is True
    assert hybrid_search._contains_value({"t1": "|a|b|"}, {"t1": "|x|y|"}) is False


def test_document_matches_filters_handles_include_and_exclude(monkeypatch):
    hybrid_search, _ = _install_retrieval_stubs(monkeypatch)

    document = types.SimpleNamespace(
        metadata={
            "paper_id": "paper-1",
            "authors": ["Alice", "Bob"],
            "categories": ["cs.AI"],
            "tables": {"t1": "|a|b|"},
            "section_id": 0,
        }
    )

    assert hybrid_search._document_matches_filters(
        document,
        {
            "paper_id": {"value": "paper-1", "enabled": True},
            "authors": {"value": "Alice", "enabled": True},
            "tables.t1": {"value": "|a|b|", "enabled": True},
            "section_id": {"value": 1, "enabled": False},
        },
    ) is True

    assert hybrid_search._document_matches_filters(
        document,
        {
            "paper_id": {"value": "paper-2", "enabled": True},
        },
    ) is False

    assert hybrid_search._document_matches_filters(
        document,
        {
            "section_id": {"value": 0, "enabled": False},
        },
    ) is False


def test_apply_metadata_filters_keeps_only_matching_documents(monkeypatch):
    hybrid_search, _ = _install_retrieval_stubs(monkeypatch)

    docs = [
        types.SimpleNamespace(metadata={"paper_id": "p1", "authors": ["Alice"], "section_id": 0}),
        types.SimpleNamespace(metadata={"paper_id": "p2", "authors": ["Bob"], "section_id": 1}),
    ]

    filtered = hybrid_search._apply_metadata_filters(
        docs,
        {
            "authors": {"value": "Alice", "enabled": True},
            "section_id": {"value": 1, "enabled": False},
        },
    )

    assert filtered == [docs[0]]


def test_run_hybrid_search_filters_before_indexing(monkeypatch):
    docs = [
        types.SimpleNamespace(metadata={"paper_id": "p1", "authors": ["Alice"], "section_id": 0}),
        types.SimpleNamespace(metadata={"paper_id": "p2", "authors": ["Bob"], "section_id": 1}),
    ]
    hybrid_search, calls = _install_retrieval_stubs(monkeypatch, documents=docs)

    response = hybrid_search.run_hybrid_search(
        "find alice",
        top_k=3,
        metadata_filters={
            "authors": {"value": "Alice", "enabled": True},
            "section_id": {"value": 1, "enabled": False},
        },
    )

    assert calls["vector_store_kwargs"]["collection_name"] == hybrid_search.COLLECTION_NAME
    assert "enable_hybrid" not in calls["vector_store_kwargs"]
    assert "fastembed_sparse_model" not in calls["vector_store_kwargs"]
    assert calls["embed_model"].model == hybrid_search.EMBEDDING_MODEL_NAME
    assert calls["query_engine_kwargs"]["similarity_top_k"] == 3
    assert "vector_store_query_mode" not in calls["query_engine_kwargs"]
    assert calls["queried"] == "find alice"
    assert response.answer == "response for find alice"


def test_run_hybrid_search_adds_cross_encoder_reranker_when_requested(monkeypatch):
    docs = [types.SimpleNamespace(metadata={"paper_id": "p1", "authors": ["Alice"], "section_id": 0})]
    hybrid_search, calls = _install_retrieval_stubs(monkeypatch, documents=docs)

    hybrid_search.run_hybrid_search(
        "find alice",
        top_k=7,
        rerank_top_n=3,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    )

    assert calls["query_engine_kwargs"]["similarity_top_k"] == 7
    assert len(calls["query_engine_kwargs"]["node_postprocessors"]) == 1
    assert calls["reranker_kwargs"] == {
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "top_n": 3,
    }


def test_run_hybrid_search_rejects_rerank_top_n_greater_than_top_k(monkeypatch):
    hybrid_search, calls = _install_retrieval_stubs(monkeypatch, documents=[])

    try:
        hybrid_search.run_hybrid_search(
            "find alice",
            top_k=3,
            rerank_top_n=4,
        )
    except ValueError as exc:
        assert str(exc) == "rerank_top_n must be less than or equal to top_k"
    else:
        raise AssertionError("Expected ValueError when rerank_top_n > top_k")

    assert "query_engine_kwargs" not in calls
