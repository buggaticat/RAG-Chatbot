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
    sys.modules.pop("rag.translation.translate", None)
    return importlib.import_module("rag.translation.translate")


def test_translate_user_query_returns_plain_string_from_dict(monkeypatch):
    translate = _import_translate(monkeypatch, response={"text": "hello"})

    assert translate.translate_user_query("hola") == "hello"


def test_translate_user_query_falls_back_to_original_text(monkeypatch):
    translate = _import_translate(monkeypatch, raises=True)

    assert translate.translate_user_query("hola") == "hola"
