"""Validation-layer configuration for grounding checks."""

VALID_CONFIDENCE_LEVELS = {"low", "medium", "high"}
NOT_FOUND_SENTINEL = "Not found in context"
MAX_NOT_FOUND_WORDS = 8
CRITIC_SYSTEM_PROMPT = """You are a verifier.

You receive:
1) Retrieved context
2) The assistant's answer in JSON

Your job:
- Check whether each factual statement in the answer is directly supported by the context.
- If any statement is not supported, or contradicts the context, mark the answer INVALID.

Return strict JSON:
{
  "valid": true or false,
  "unsupported_claims": ["..."],
  "missing_context": ["..."]
}
"""
