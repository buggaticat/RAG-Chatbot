"""Tests for translation normalization and fallback handling."""

import importlib
import sys
import types


def _import_translate(monkeypatch, response=None, raises=False):
    class FakeTranslate:
        def text(self, payload):
            if raises:
                raise RuntimeError("boom")
            return response

    class FakeJigsawStack:
        def __init__(self, api_key=None):
            self.translate = FakeTranslate()

    fake_module = types.ModuleType("jigsawstack")
    fake_module.JigsawStack = FakeJigsawStack
    monkeypatch.setitem(sys.modules, "jigsawstack", fake_module)
    fake_config = types.ModuleType("rag.translation.config")
    fake_config.JIGSAW_APIKEY = "dummy-key"
    fake_config.TARGET_LANGUAGE = "en"
    monkeypatch.setitem(sys.modules, "rag.translation.config", fake_config)
    sys.modules.pop("rag.translation.translate", None)
    return importlib.import_module("rag.translation.translate")


def test_translate_user_query_returns_plain_string_from_dict(monkeypatch):
    translate = _import_translate(monkeypatch, response={"text": "hello"})

    assert translate.translate_user_query("hola") == "hello"


def test_translate_user_query_falls_back_to_original_text(monkeypatch):
    translate = _import_translate(monkeypatch, raises=True)

    assert translate.translate_user_query("hola") == "hola"


def test_translate_user_query_returns_original_when_client_is_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "jigsawstack", raising=False)
    fake_config = types.ModuleType("rag.translation.config")
    fake_config.JIGSAW_APIKEY = None
    fake_config.TARGET_LANGUAGE = "en"
    monkeypatch.setitem(sys.modules, "rag.translation.config", fake_config)
    sys.modules.pop("rag.translation.translate", None)
    translate = importlib.import_module("rag.translation.translate")

    assert translate.translate_user_query("hola") == "hola"


def test_translate_user_query_prefers_text_attribute(monkeypatch):
    translate = _import_translate(monkeypatch, response=types.SimpleNamespace(text="bonjour"))

    assert translate.translate_user_query("hola") == "bonjour"
