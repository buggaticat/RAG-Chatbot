"""Construct grounded prompts for arXiv context answering."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .config import GROUNDING_TEMPLATE, INSTRUCTIONS, SYSTEM_PROMPT, get_tokenizer

SCHEMA = json.dumps(
    {
        "answer": "<short answer or 'Not found in context'>",
        "citations": [
            {
                "doc_id": "<doc_id>",
                "chunk_id": "<chunk_id>",
            }
        ],
        "confidence": "<low|medium|high>",
    },
    indent=2,
)


def _estimate_tokens(text: str, tokenizer: Any | None) -> int:
    """Estimate token count for text using either a tokenizer or word split."""

    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))


def _trim_tokens(tokens: Sequence[Any], allowed_tokens: int) -> Sequence[Any]:
    """Trim a token sequence to the requested maximum length."""

    if allowed_tokens <= 0:
        return tokens[:0]
    return tokens[:allowed_tokens]


def _render_prompt(system_prompt: str, instructions: str, query: str, context: str) -> str:
    """Render the full grounded prompt body with the provided context."""

    return (
        GROUNDING_TEMPLATE.format(
            system_prompt=system_prompt,
            instructions=instructions,
            context=context,
            schema=SCHEMA,
        )
        + f"\n\nUser query:\n{query}\n"
    )


def build_grounded_prompt(
    system_prompt: str | None,
    instructions: str | None,
    query: str,
    context: str,
    tokenizer: Any | None,
    max_tokens: int,
) -> str:
    """Build a grounded prompt while trimming context to fit the token budget."""

    if tokenizer is None:
        tokenizer = get_tokenizer()

    if not system_prompt:
        system_prompt = SYSTEM_PROMPT
    if not instructions:
        instructions = INSTRUCTIONS

    fixed_prompt = _render_prompt(system_prompt, instructions, query, "")
    fixed_tokens = _estimate_tokens(fixed_prompt, tokenizer)

    if fixed_tokens >= max_tokens:
        return fixed_prompt

    if _estimate_tokens(_render_prompt(system_prompt, instructions, query, context), tokenizer) <= max_tokens:
        return _render_prompt(system_prompt, instructions, query, context)

    if tokenizer is not None:
        context_tokens = tokenizer.encode(context)
    else:
        context_tokens = context.split()

    allowed_context_tokens = max_tokens - fixed_tokens
    trimmed_tokens = _trim_tokens(context_tokens, allowed_context_tokens)
    if tokenizer is not None:
        context_trimmed = tokenizer.decode(trimmed_tokens)
    else:
        context_trimmed = " ".join(trimmed_tokens)

    return _render_prompt(system_prompt, instructions, query, context_trimmed)
