from __future__ import annotations

from types import SimpleNamespace

from rag.context_assembly import format_context


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def test_format_context_keeps_whole_chunks_under_token_budget():
    response = SimpleNamespace(
        source_nodes=[
            SimpleNamespace(metadata={"doc_id": "doc-1", "chunk_id": "c1", "title": "First"}, text="alpha beta gamma"),
            SimpleNamespace(metadata={"doc_id": "doc-2", "chunk_id": "c2", "title": "Second"}, text="delta epsilon zeta eta"),
        ]
    )

    context = format_context(response, max_tokens=10, tokenizer=FakeTokenizer())

    assert "[Chunk 1] doc=doc-1 chunk=c1" in context
    assert "alpha beta gamma" in context
    assert "doc=doc-2" not in context


def test_format_context_accepts_plain_chunk_lists():
    chunks = [
        SimpleNamespace(metadata={"doc_id": "doc-1", "chunk_id": "c1", "title": "First"}, text="hello world"),
    ]

    context = format_context(chunks)

    assert "[Chunk 1] doc=doc-1 chunk=c1" in context
    assert "Title: First" in context
    assert "hello world" in context


def test_format_context_handles_repo_source_chunk_dicts():
    chunks = [
        {
            "doc_id": "doc-1",
            "chunk_id": "c1",
            "title": "First",
            "content": "alpha beta",
            "score": 0.9876,
        }
    ]

    context = format_context(chunks)

    assert "[Chunk 1] doc=doc-1 chunk=c1 score=0.99" in context
    assert "Title: First" in context
    assert "alpha beta" in context


def test_format_context_reads_nested_node_text():
    chunks = [
        SimpleNamespace(
            metadata={"paper_id": "paper-1", "chunk_index": 2, "title": "Nested"},
            node=SimpleNamespace(text="nested text"),
        )
    ]

    context = format_context(chunks)

    assert "[Chunk 1] doc=paper-1 chunk=2" in context
    assert "Title: Nested" in context
    assert "nested text" in context
