"""Tests for embedding generation and image decoding."""

import base64
import json
import importlib
import sys
import types
from io import BytesIO
from contextlib import nullcontext

from PIL import Image


def _install_dependency_stubs(monkeypatch):
    class FakeEmbedding:
        def get_text_embedding(self, text):
            return [float(len(text))]

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, image, return_tensors="pt"):
            return types.SimpleNamespace(to=lambda device: {"pixel_values": "stub"})

        def decode(self, output_ids, skip_special_tokens=True):
            return "caption"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def to(self, device):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTorch(types.SimpleNamespace):
        def __init__(self):
            super().__init__(
                cuda=types.SimpleNamespace(is_available=lambda: False),
                no_grad=lambda: nullcontext(),
            )

    fake_llama_index = types.ModuleType("llama_index")
    fake_embeddings = types.ModuleType("llama_index.embeddings")
    fake_openai = types.ModuleType("llama_index.embeddings.openai")
    fake_openai.OpenAIEmbedding = lambda: FakeEmbedding()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.BlipProcessor = FakeProcessor
    fake_transformers.BlipForConditionalGeneration = FakeModel

    monkeypatch.setitem(sys.modules, "llama_index", fake_llama_index)
    monkeypatch.setitem(sys.modules, "llama_index.embeddings", fake_embeddings)
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.openai", fake_openai)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())


def _import_embed(monkeypatch):
    _install_dependency_stubs(monkeypatch)

    build_documents_stub = types.ModuleType("rag.ingestion.build_documents")
    build_documents_stub.build_all_documents = lambda: []
    build_documents_stub.iter_all_documents = lambda: iter([])
    monkeypatch.setitem(sys.modules, "rag.ingestion.build_documents", build_documents_stub)

    module = importlib.import_module("rag.ingestion.embed")
    module._clear_checkpoint()
    return module


def _make_document(text, metadata):
    return types.SimpleNamespace(
        text=text,
        metadata=metadata,
        get_content=lambda metadata_mode=None: text,
    )


def _make_base64_png():
    img = Image.new("RGB", (1, 1), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_base64_image(monkeypatch):
    embed = _import_embed(monkeypatch)
    data = _make_base64_png()
    image = embed._decode_base64_image(base64.b64encode(data).decode("utf-8"))

    assert image.size == (1, 1)
    assert image.mode == "RGB"


def test_decode_base64_image_handles_data_uri_prefix(monkeypatch):
    embed = _import_embed(monkeypatch)
    data = _make_base64_png()
    data_uri = "data:image/png;base64," + base64.b64encode(data).decode("utf-8")

    image = embed._decode_base64_image(data_uri)

    assert image.size == (1, 1)
    assert image.mode == "RGB"


def test_decode_base64_image_handles_whitespace_and_urlsafe_padding(monkeypatch):
    embed = _import_embed(monkeypatch)
    data = _make_base64_png()
    encoded = base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")
    encoded = "  \n" + encoded[:10] + " \n" + encoded[10:] + "  "

    image = embed._decode_base64_image(encoded)

    assert image.size == (1, 1)
    assert image.mode == "RGB"


def test_documents_to_embeddings_emits_text_tables_and_image_caption(monkeypatch):
    embed = _import_embed(monkeypatch)

    docs = [
        _make_document(
            "section text",
            {
                "source_field": "sections.text",
                "tables": {"table-1": "|a|b|"},
                "images": {"image-1": "ignored-base64"},
            },
        )
    ]

    monkeypatch.setattr(embed, "iter_all_documents", lambda: iter(docs))
    monkeypatch.setattr(embed, "_get_blip_components", lambda: ("model", "processor", "cpu"))
    monkeypatch.setattr(embed, "_generate_caption", lambda *args: "generated caption")
    monkeypatch.setattr(embed.embedding_model, "get_text_embedding", lambda text: [len(text)])
    monkeypatch.setattr(embed, "_decode_base64_image", lambda data: Image.new("RGB", (1, 1)))

    result = embed.documents_to_embeddings()

    assert [item[0] for item in result] == [
        "section text",
        "generated caption",
    ]
    assert result[1][2]["image_id"] == "image-1"


def test_documents_to_embeddings_does_not_duplicate_sidecar_embeddings(monkeypatch):
    embed = _import_embed(monkeypatch)

    docs = [
        _make_document("abstract text", {"source_field": "abstract", "tables": {"table-1": "|a|b|"}, "images": {"image-1": "ignored-base64"}}),
        _make_document("section text", {"source_field": "sections.text", "tables": {"table-1": "|a|b|"}, "images": {"image-1": "ignored-base64"}}),
        _make_document("|a|b|", {"source_field": "sections.tables", "table_id": "table-1", "tables": {"table-1": "|a|b|"}}),
    ]

    monkeypatch.setattr(embed, "iter_all_documents", lambda: iter(docs))
    monkeypatch.setattr(embed, "_get_blip_components", lambda: ("model", "processor", "cpu"))
    monkeypatch.setattr(embed, "_generate_caption", lambda *args: "generated caption")
    monkeypatch.setattr(embed.embedding_model, "get_text_embedding", lambda text: [len(text)])
    monkeypatch.setattr(embed, "_decode_base64_image", lambda data: Image.new("RGB", (1, 1)))

    result = embed.documents_to_embeddings()

    assert [item[0] for item in result] == [
        "abstract text",
        "section text",
        "generated caption",
        "|a|b|",
    ]


def test_documents_to_embeddings_deduplicates_images_across_split_section_chunks(monkeypatch):
    embed = _import_embed(monkeypatch)

    docs = [
        _make_document("section chunk one", {"paper_id": "paper-1", "section_id": "1", "source_field": "sections.text", "images": {"image-1": "ignored-base64"}}),
        _make_document("section chunk two", {"paper_id": "paper-1", "section_id": "1", "source_field": "sections.text", "images": {"image-1": "ignored-base64"}}),
    ]

    monkeypatch.setattr(embed, "iter_all_documents", lambda: iter(docs))
    monkeypatch.setattr(embed, "_get_blip_components", lambda: ("model", "processor", "cpu"))
    monkeypatch.setattr(embed, "_generate_caption", lambda *args: "generated caption")
    monkeypatch.setattr(embed.embedding_model, "get_text_embedding", lambda text: [len(text)])
    monkeypatch.setattr(embed, "_decode_base64_image", lambda data: Image.new("RGB", (1, 1)))

    result = embed.documents_to_embeddings()

    assert [item[0] for item in result] == [
        "section chunk one",
        "generated caption",
        "section chunk two",
    ]


def test_documents_to_embeddings_skips_unreadable_images(monkeypatch):
    embed = _import_embed(monkeypatch)

    docs = [
        _make_document(
            "section text",
            {
                "paper_id": "paper-1",
                "section_id": "1",
                "source_field": "sections.text",
                "images": {"image-1": "not-a-valid-image"},
            },
        )
    ]

    monkeypatch.setattr(embed, "iter_all_documents", lambda: iter(docs))
    monkeypatch.setattr(embed.embedding_model, "get_text_embedding", lambda text: [len(text)])

    result = embed.documents_to_embeddings()

    assert [item[0] for item in result] == ["section text"]


def test_documents_to_embeddings_writes_and_resumes_from_checkpoint(monkeypatch, tmp_path):
    embed = _import_embed(monkeypatch)
    checkpoint_path = tmp_path / ".embedding_checkpoint.json"
    monkeypatch.setattr(embed, "EMBEDDING_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(embed, "BATCH_SIZE", 1)

    docs = [
        _make_document("doc one", {"source_field": "abstract"}),
        _make_document("doc two", {"source_field": "abstract"}),
    ]

    monkeypatch.setattr(embed, "iter_all_documents", lambda: iter(docs))
    monkeypatch.setattr(embed.embedding_model, "get_text_embedding", lambda text: [len(text)])

    first_result = embed.documents_to_embeddings()
    assert [item[0] for item in first_result] == ["doc one", "doc two"]

    saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert saved["next_task_index"] == 2
    assert saved["embeddings"] == []

    calls = []

    def tracking_embed(text):
        calls.append(text)
        return [len(text)]

    monkeypatch.setattr(embed.embedding_model, "get_text_embedding", tracking_embed)
    second_result = embed.documents_to_embeddings()

    assert second_result == []
    assert calls == []


def test_documents_to_embeddings_uses_batch_embedding_when_available(monkeypatch):
    embed = _import_embed(monkeypatch)

    docs = [
        _make_document("doc one", {"source_field": "abstract"}),
        _make_document("doc two", {"source_field": "abstract"}),
    ]

    monkeypatch.setattr(embed, "iter_all_documents", lambda: iter(docs))

    batch_calls = []

    def batch_embed(texts):
        batch_calls.append(list(texts))
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(embed.embedding_model, "get_text_embedding_batch", batch_embed, raising=False)

    result = embed.documents_to_embeddings()

    assert [item[0] for item in result] == ["doc one", "doc two"]
    assert batch_calls == [["doc one", "doc two"]]
