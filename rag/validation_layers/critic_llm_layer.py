"""Prompt-based critic layer for grounding and contradiction verification."""

from __future__ import annotations

import json
from typing import Any, Callable
from .config import CRITIC_SYSTEM_PROMPT


def build_critic_prompt(context: str, answer_json: Any) -> str:
    """Build the critic prompt from retrieved context and the assistant JSON answer."""

    if isinstance(answer_json, str):
        answer_payload = answer_json
    else:
        answer_payload = json.dumps(answer_json, ensure_ascii=False, indent=2)

    return (
        f"{CRITIC_SYSTEM_PROMPT.strip()}\n\n"
        f"Retrieved context:\n{context.strip()}\n\n"
        f"Assistant answer JSON:\n{answer_payload}\n"
    )


def verify_with_critic(context: str, answer_json: Any, critic_llm: Callable[[str], str]) -> dict[str, Any]:
    """Call a critic LLM and parse the returned verdict JSON."""

    prompt = build_critic_prompt(context, answer_json)
    resp = critic_llm(prompt)
    verdict = json.loads(resp)

    if not isinstance(verdict, dict):
        raise ValueError("Critic response must be a JSON object.")

    if "valid" not in verdict:
        raise ValueError("Critic response must include a 'valid' field.")
    if "unsupported_claims" not in verdict:
        verdict["unsupported_claims"] = []
    if "missing_context" not in verdict:
        verdict["missing_context"] = []

    if not isinstance(verdict["unsupported_claims"], list):
        raise ValueError("'unsupported_claims' must be a list.")
    if not isinstance(verdict["missing_context"], list):
        raise ValueError("'missing_context' must be a list.")

    return verdict
