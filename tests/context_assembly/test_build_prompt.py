from __future__ import annotations

from rag.context_assembly import SCHEMA, build_grounded_prompt


def test_build_grounded_prompt_includes_grounding_and_instructions():
    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

        def decode(self, tokens: list[str]) -> str:
            return " ".join(tokens)

    prompt = build_grounded_prompt(
        system_prompt="System text",
        instructions="Follow the rules",
        query="What happened?",
        context="Relevant context here",
        tokenizer=FakeTokenizer(),
        max_tokens=100,
    )

    assert "System:\nSystem text" in prompt
    assert "Instructions:\nFollow the rules" in prompt
    assert "You are an enterprise assistant. Use only the context provided below." in prompt
    assert "Return your answer as valid JSON" in prompt
    assert SCHEMA in prompt


def test_build_grounded_prompt_handles_tiny_budget():
    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

        def decode(self, tokens: list[str]) -> str:
            return " ".join(tokens)

    prompt = build_grounded_prompt(
        system_prompt="System text that is long",
        instructions="Follow the rules carefully",
        query="What happened?",
        context="Relevant context here",
        tokenizer=FakeTokenizer(),
        max_tokens=3,
    )

    assert "System:\nSystem text that is long" in prompt
    assert "Context:\n" in prompt


def test_build_grounded_prompt_trims_context_when_needed():
    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

        def decode(self, tokens: list[str]) -> str:
            return " ".join(tokens)

    fixed_prompt = build_grounded_prompt(
        system_prompt="System text",
        instructions="Follow the rules",
        query="What happened?",
        context="",
        tokenizer=FakeTokenizer(),
        max_tokens=1000,
    )
    fixed_tokens = len(FakeTokenizer().encode(fixed_prompt))

    prompt = build_grounded_prompt(
        system_prompt="System text",
        instructions="Follow the rules",
        query="What happened?",
        context="one two three four five six seven eight nine ten",
        tokenizer=FakeTokenizer(),
        max_tokens=fixed_tokens + 3,
    )

    assert "one two three four five six seven eight nine ten" not in prompt
    assert "one two three" in prompt
