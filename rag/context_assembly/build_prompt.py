from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


SYSTEM_PROMPT = """You are a careful scientific RAG assistant for arXiv PDF documents.
Use only the provided context and do not guess beyond the evidence.
The context may include section text, tables, and image descriptions or placeholders.
Prefer exact, concise answers grounded in the retrieved material.
If the answer is not supported by the context, say "Not found in context".
When the evidence spans multiple chunks, synthesize only what is explicitly supported.
"""

INSTRUCTIONS = """- Answer the user's question directly and briefly.
- Treat tables, figure captions, and nearby text as relevant evidence.
- If the query needs a table or image and the context does not include it, say so.
- Preserve technical names, numbers, equations, and section-specific details exactly when possible.
- Return only the answer content needed by the task.
"""

GROUNDING_TEMPLATE = """You are an enterprise assistant. Use only the context provided below.

System:
{system_prompt}

Instructions:
{instructions}

Context:
{context}

If the requested information is not present in the context, reply exactly:
"Not found in context"

Return your answer as valid JSON with this schema:

{schema}
"""

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
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))


def _trim_tokens(tokens: Sequence[Any], allowed_tokens: int) -> Sequence[Any]:
    if allowed_tokens <= 0:
        return tokens[:0]
    return tokens[:allowed_tokens]


def _render_prompt(system_prompt: str, instructions: str, query: str, context: str) -> str:
    return (
        GROUNDING_TEMPLATE.format(
            system_prompt=system_prompt,
            instructions=instructions,
            context=context,
            schema=SCHEMA,
        )
        + f"\n\nUser query:\n{query}\n"
    )


def build_grounded_prompt(system_prompt, instructions, query, context, tokenizer, max_tokens):
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
