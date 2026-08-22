"""Shared chatbot result and state types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from rag.validation_layers import DeterministicValidationResult

from .prompt import NormalizedQuery


@dataclass
class SubQuestionResult:
    """Execution trace for one decomposed sub-question."""

    subquestion: str
    retrieval_query: str
    retrieval_response: Any
    context: str
    prompt: str
    answer_raw: Any
    deterministic_result: DeterministicValidationResult
    critic_verdict: dict[str, Any] | None
    accepted: bool
    retried_context: bool = False
    retried_retrieval: bool = False
    failure_reason: str | None = None
    retrieval_feedback: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ChatbotRunResult:
    """Overall result from processing one user prompt."""

    original_query: str
    normalized_query: NormalizedQuery
    rewritten_query: str
    subquestions: list[str]
    subquestion_results: list[SubQuestionResult]
    final_answer: str
    accepted: bool
    retry_count: int = 0
    feedback: str | None = None
    component_timings: list[dict[str, Any]] = field(default_factory=list)


class SubQuestionState(TypedDict, total=False):
    """State payload for the per-subquestion LangGraph."""

    subquestion: str
    retrieval_query: str
    retrieval_response: Any
    reranked_response: Any
    context: str
    prompt: str
    answer_raw: Any
    deterministic_result: DeterministicValidationResult
    critic_verdict: dict[str, Any]
    subquestion_result: SubQuestionResult
    accepted: bool
    strict_context: bool
    context_retry_count: int
    retried_context: bool
    retried_retrieval: bool
    needs_rewrite_retry: bool
    rewrite_feedback: str
    retrieval_feedback: str
    notes: list[str]
    failure_reason: str | None


class ChatbotState(TypedDict, total=False):
    """State payload for the top-level LangGraph."""

    original_query: str
    normalized_query: NormalizedQuery
    rewrite_feedback: str
    rewritten_query: str
    subquestions: list[str]
    subquestion_results: list[SubQuestionResult]
    final_answer: str
    accepted: bool
    retry_count: int
    needs_rewrite_retry: bool
    component_timings: list[dict[str, Any]]
