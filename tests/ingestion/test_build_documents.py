import json
import importlib
import sys
import types
from io import BytesIO


def _install_llama_index_stubs(monkeypatch):
    class FakeDocument:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or {}

        def get_content(self, metadata_mode=None):
            return self.text

    class FakeSentenceSplitter:
        def __init__(self, chunk_size=512, chunk_overlap=50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text):
            if len(text) <= self.chunk_size:
                return [text]
            return [text[: self.chunk_size], text[self.chunk_size :]]

    core_module = types.ModuleType("llama_index.core")
    core_module.Document = FakeDocument

    node_parser_module = types.ModuleType("llama_index.core.node_parser")
    node_parser_module.SentenceSplitter = FakeSentenceSplitter

    monkeypatch.setitem(sys.modules, "llama_index", types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core", core_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", node_parser_module)


def _import_build_documents(monkeypatch):
    _install_llama_index_stubs(monkeypatch)

    fake_boto3 = types.ModuleType("boto3")
    fake_client = types.SimpleNamespace()
    fake_boto3.client = lambda service_name: fake_client
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    module = importlib.import_module("src.ingestion.build_documents")
    return module, fake_client


def test_safe_metadata_converts_nested_values(monkeypatch):
    build_documents, _ = _import_build_documents(monkeypatch)

    class CustomObj:
        def __str__(self):
            return "custom-value"

    value = {
        "a": 1,
        "b": [True, {"c": CustomObj()}],
    }

    assert build_documents._safe_metadata(value) == {
        "a": 1,
        "b": [True, {"c": "custom-value"}],
    }


def test_build_documents_from_paper_includes_text_and_metadata(monkeypatch):
    build_documents, _ = _import_build_documents(monkeypatch)
    monkeypatch.setattr(build_documents.splitter, "split_text", lambda text: [text])

    paper = {
        "id": "paper-1",
        "title": "A Paper",
        "authors": ["Alice"],
        "categories": ["cs.AI"],
        "updated": "2026-08-01",
        "published": "2026-07-01",
        "abstract": "Abstract text",
        "sections": [
            {
                "section_id": 0,
                "text": "Section text",
                "tables": {"t1": "|a|b|"},
                "images": {"i1": "base64"},
            }
        ],
    }

    documents = build_documents._build_documents_from_paper(paper)

    assert len(documents) == 2
    assert documents[0].text == "Abstract text"
    assert documents[0].metadata["source_field"] == "abstract"
    assert documents[0].metadata["paper_id"] == "paper-1"

    assert documents[1].text == "Section text"
    assert documents[1].metadata["source_field"] == "sections.text"
    assert documents[1].metadata["section_id"] == 0
    assert documents[1].metadata["tables"] == {"t1": "|a|b|"}
    assert documents[1].metadata["images"] == {"i1": "base64"}


def test_build_all_documents_paginates_s3_results(monkeypatch):
    build_documents, fake_client = _import_build_documents(monkeypatch)

    pages = [
        {
            "Contents": [{"Key": "paper-1.json"}],
            "IsTruncated": True,
            "NextContinuationToken": "token-1",
        },
        {
            "Contents": [{"Key": "paper-2.json"}],
            "IsTruncated": False,
        },
    ]

    payloads = {
        "paper-1.json": json.dumps({"abstract": "A", "sections": []}).encode("utf-8"),
        "paper-2.json": json.dumps({"abstract": "B", "sections": []}).encode("utf-8"),
    }

    fake_client.list_objects_v2 = lambda **kwargs: pages.pop(0)

    def get_object(Bucket, Key):
        return {"Body": BytesIO(payloads[Key])}

    fake_client.get_object = get_object

    monkeypatch.setattr(build_documents, "_build_documents_from_paper", lambda data: [types.SimpleNamespace(text=data["abstract"])])

    documents = build_documents.build_all_documents()

    assert [doc.text for doc in documents] == ["A", "B"]
