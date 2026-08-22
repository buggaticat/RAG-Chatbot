from __future__ import annotations

from types import SimpleNamespace

from chatbot.prompt import decompose_rewritten_query, normalize_query_language, rewrite_user_query


def test_normalize_query_language_uses_translation_for_any_query():
    seen = {}

    def fake_translate(text: str) -> str:
        seen["text"] = text
        return "how does this work"

    query = "\u00bfC\u00f3mo funciona esto?"
    result = normalize_query_language(query, translator=fake_translate)

    assert seen["text"] == query
    assert result.original == query
    assert result.normalized == "how does this work"


def test_rewrite_user_query_uses_model_when_available():
    class FakeModel:
        def invoke(self, messages):
            assert len(messages) == 2
            assert "retrieval-optimized" in messages[1].content
            return SimpleNamespace(content="clean retrieval query")

    assert rewrite_user_query("What is the model?", model=FakeModel()) == "clean retrieval query"


def test_decompose_rewritten_query_parses_json_lists():
    class FakeModel:
        def invoke(self, messages):
            return SimpleNamespace(content='{"subquestions": ["first", "second", "first"]}')

    assert decompose_rewritten_query("Rewrite", model=FakeModel()) == ["first", "second"]


def test_decompose_rewritten_query_parses_fenced_json_lists():
    class FakeModel:
        def invoke(self, messages):
            return SimpleNamespace(content='```json\n{"subquestions": ["first", "second"]}\n```')

    assert decompose_rewritten_query("Rewrite", model=FakeModel()) == ["first", "second"]


def test_decompose_rewritten_query_prompt_discourages_unnecessary_splitting():
    seen = {}

    class FakeModel:
        def invoke(self, messages):
            seen["system"] = messages[0].content
            seen["human"] = messages[1].content
            return SimpleNamespace(content="[]")

    decompose_rewritten_query("Rewrite", model=FakeModel())

    assert "do not decompose it" in seen["system"]
    assert "multiple parts/questions" in seen["system"]
    assert "If no decomposition is needed, return an empty list." in seen["system"]
