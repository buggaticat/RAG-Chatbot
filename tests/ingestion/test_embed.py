import base64
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

    build_documents_stub = types.ModuleType("src.ingestion.build_documents")
    build_documents_stub.build_all_documents = lambda: []
    monkeypatch.setitem(sys.modules, "src.ingestion.build_documents", build_documents_stub)

    module = importlib.import_module("src.ingestion.embed")
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


def test_documents_to_embeddings_emits_text_tables_and_image_caption(monkeypatch):
    embed = _import_embed(monkeypatch)

    docs = [
        _make_document(
            "section text",
            {
                "tables": {"table-1": "|a|b|"},
                "images": {"image-1": "ignored-base64"},
            },
        )
    ]

    monkeypatch.setattr(embed, "build_all_documents", lambda: docs)
    monkeypatch.setattr(embed, "_get_blip_components", lambda: ("model", "processor", "cpu"))
    monkeypatch.setattr(embed, "_generate_caption", lambda *args: "generated caption")
    monkeypatch.setattr(embed.embedding_model, "get_text_embedding", lambda text: [len(text)])
    monkeypatch.setattr(embed, "_decode_base64_image", lambda data: Image.new("RGB", (1, 1)))

    result = embed.documents_to_embeddings()

    assert [item[0] for item in result] == [
        "section text",
        "|a|b|",
        "generated caption",
    ]
    assert result[1][2]["table_id"] == "table-1"
    assert result[2][2]["image_id"] == "image-1"
