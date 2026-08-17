"""Shared helpers for eval metrics and reports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    """Collapse whitespace and trim a string."""

    return " ".join(text.strip().split())


def estimate_tokens(text: str) -> int:
    """Estimate token count when an upstream model does not expose usage."""

    cleaned = normalize_text(text)
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / 4))


def message_to_text(message: Any) -> str:
    """Render a prompt message-like object as text."""

    if isinstance(message, str):
        return message

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text

    if isinstance(message, dict):
        for key in ("content", "text", "output", "result"):
            value = message.get(key)
            if isinstance(value, str):
                return value

    return str(message)


def render_payload_text(payload: Any) -> str:
    """Convert a model payload into plain text for estimation and logging."""

    if isinstance(payload, str):
        return payload
    if isinstance(payload, (list, tuple)):
        rendered = []
        for item in payload:
            role = getattr(item, "type", "message")
            rendered.append(f"{role.upper()}:\n{message_to_text(item)}")
        return "\n\n".join(rendered)
    return message_to_text(payload)


def extract_usage_from_response(response: Any) -> tuple[int | None, int | None]:
    """Pull token usage out of a response object when it is exposed."""

    if response is None:
        return None, None

    def _coerce_pair(candidate: Any) -> tuple[int | None, int | None] | None:
        """Extract token counts from a nested response payload."""

        if candidate is None:
            return None

        if isinstance(candidate, dict):
            direct_pairs = (
                ("prompt_tokens", "completion_tokens"),
                ("input_tokens", "output_tokens"),
                ("input", "output"),
            )
            for input_key, output_key in direct_pairs:
                input_tokens = candidate.get(input_key)
                output_tokens = candidate.get(output_key)
                if input_tokens is not None or output_tokens is not None:
                    return (
                        int(input_tokens) if input_tokens is not None else None,
                        int(output_tokens) if output_tokens is not None else None,
                    )

            for nested_key in ("usage", "usage_metadata", "response_metadata", "llm_output", "additional_kwargs", "token_usage"):
                nested = candidate.get(nested_key)
                if nested is None:
                    continue
                extracted = _coerce_pair(nested)
                if extracted is not None:
                    return extracted
            return None

        for input_key, output_key in (
            ("prompt_tokens", "completion_tokens"),
            ("input_tokens", "output_tokens"),
        ):
            input_tokens = getattr(candidate, input_key, None)
            output_tokens = getattr(candidate, output_key, None)
            if input_tokens is not None or output_tokens is not None:
                return (
                    int(input_tokens) if input_tokens is not None else None,
                    int(output_tokens) if output_tokens is not None else None,
                )

        for attr_name in ("usage", "usage_metadata", "response_metadata", "llm_output", "additional_kwargs", "token_usage"):
            nested = getattr(candidate, attr_name, None)
            if nested is None:
                continue
            extracted = _coerce_pair(nested)
            if extracted is not None:
                return extracted
        return None

    extracted = _coerce_pair(response)
    if extracted is not None:
        return extracted
    return None, None


def ensure_serializable(value: Any) -> Any:
    """Convert common dataclasses and objects into JSON-safe values."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass
    if dataclass_is_instance(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: ensure_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dataclass_is_instance(value: Any) -> bool:
    """Return True when a value is an instantiated dataclass."""

    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def summarize_numbers(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize a numeric sample with percentile statistics."""

    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[int(position)]
        lower_value = ordered[lower]
        upper_value = ordered[upper]
        return lower_value + (upper_value - lower_value) * (position - lower)

    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "std": pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def bucket_timestamp(timestamp: datetime, granularity: str = "day") -> str:
    """Bucket a timestamp into a stable time-series key."""

    normalized = timestamp.astimezone(timezone.utc)
    if granularity == "hour":
        return normalized.strftime("%Y-%m-%dT%H:00:00Z")
    return normalized.strftime("%Y-%m-%d")


@dataclass
class EvaluationReport:
    """Top-level container for the eval sections."""

    latency_and_reliability: Any
    cost_and_usage: Any
    embedding_and_index_health: Any
    retrieval_and_answer_quality: Any = None
    generated_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""

        return ensure_serializable(
            {
                "generated_at": self.generated_at,
                "latency_and_reliability": self.latency_and_reliability,
                "cost_and_usage": self.cost_and_usage,
                "embedding_and_index_health": self.embedding_and_index_health,
                "retrieval_and_answer_quality": self.retrieval_and_answer_quality,
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
