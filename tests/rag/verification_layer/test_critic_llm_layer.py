"""Tests for the critic LLM verification layer."""

from __future__ import annotations

import json

from rag.validation_layers.critic_llm_layer import build_critic_prompt, verify_with_critic


def test_build_critic_prompt_includes_context_and_answer_json():
    """Ensure the critic prompt includes both the context and JSON payload."""

    prompt = build_critic_prompt("context here", {"answer": "yes"})

    assert "You are a verifier." in prompt
    assert "Retrieved context:\ncontext here" in prompt
    assert "Assistant answer JSON:" in prompt
    assert '"answer": "yes"' in prompt


def test_verify_with_critic_parses_json_verdict():
    """Ensure the critic wrapper returns a parsed verdict dictionary."""

    def fake_critic(prompt: str) -> str:
        assert "Retrieved context:" in prompt
        return json.dumps(
            {
                "valid": False,
                "unsupported_claims": ["claim 1"],
                "missing_context": ["detail 2"],
            }
        )

    verdict = verify_with_critic("context here", {"answer": "yes"}, fake_critic)

    assert verdict["valid"] is False
    assert verdict["unsupported_claims"] == ["claim 1"]
    assert verdict["missing_context"] == ["detail 2"]


def test_verify_with_critic_accepts_fenced_json():
    """Ensure the critic wrapper tolerates fenced JSON output."""

    def fake_critic(prompt: str) -> str:
        return "```json\n{\"valid\": true, \"unsupported_claims\": [], \"missing_context\": []}\n```"

    verdict = verify_with_critic("context here", {"answer": "yes"}, fake_critic)

    assert verdict["valid"] is True


def test_verify_with_critic_rejects_missing_valid_field():
    """Ensure missing validity metadata raises a useful error."""

    def fake_critic(prompt: str) -> str:
        return json.dumps({"unsupported_claims": [], "missing_context": []})

    try:
        verify_with_critic("context here", {"answer": "yes"}, fake_critic)
    except ValueError as exc:
        assert "valid" in str(exc)
    else:
        raise AssertionError("Expected ValueError when 'valid' is missing")


def test_verify_with_critic_rejects_non_list_fields():
    """Ensure malformed list fields are rejected."""

    def fake_critic(prompt: str) -> str:
        return json.dumps({"valid": True, "unsupported_claims": "claim", "missing_context": []})

    try:
        verify_with_critic("context here", {"answer": "yes"}, fake_critic)
    except ValueError as exc:
        assert "unsupported_claims" in str(exc)
    else:
        raise AssertionError("Expected ValueError when 'unsupported_claims' is not a list")
