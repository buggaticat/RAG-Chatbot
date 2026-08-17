"""Retrieval and answer quality evaluation built around Ragas metrics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..utils import summarize_numbers

try:  # pragma: no cover - optional dependency
    from ragas import SingleTurnSample
except Exception:  # pragma: no cover - optional dependency
    try:
        from ragas.dataset_schema import SingleTurnSample
    except Exception:  # pragma: no cover - optional dependency
        SingleTurnSample = None

try:  # pragma: no cover - optional dependency
    from ragas.metrics import (
        IDBasedContextPrecision,
        IDBasedContextRecall,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        Faithfulness,
        ResponseGroundedness,
    )
except Exception:  # pragma: no cover - optional dependency
    IDBasedContextPrecision = None
    IDBasedContextRecall = None
    LLMContextPrecisionWithReference = None
    LLMContextRecall = None
    Faithfulness = None
ResponseGroundedness = None


def _score_value(result: Any) -> float:
    """Normalize the variety of Ragas return types into a float."""

    if hasattr(result, "value"):
        return float(result.value)
    return float(result)


async def _evaluate_metric(
    metric: Any,
    *,
    sample: Any | None = None,
    sample_factory: Any | None = None,
    **kwargs: Any,
) -> float:
    """Evaluate one Ragas metric using whichever async API it exposes."""

    if metric is None:
        raise RuntimeError("Metric is not configured.")

    if hasattr(metric, "ascore"):
        if kwargs:
            try:
                return _score_value(await metric.ascore(**kwargs))
            except TypeError:
                pass
        if sample is not None:
            try:
                return _score_value(await metric.ascore(sample))
            except TypeError:
                pass

    if hasattr(metric, "single_turn_ascore"):
        if sample is None and sample_factory is not None:
            sample = sample_factory()
        if sample is None:
            raise RuntimeError("single_turn_ascore requires a SingleTurnSample.")
        return _score_value(await metric.single_turn_ascore(sample))

    if hasattr(metric, "score"):
        if kwargs:
            try:
                return _score_value(metric.score(**kwargs))
            except TypeError:
                pass
        if sample is not None:
            try:
                return _score_value(metric.score(sample))
            except TypeError:
                pass

    raise TypeError(f"Unsupported metric interface for {metric!r}.")


def _run_coroutine(coro: Any) -> Any:
    """Run an async coroutine from synchronous code."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("evaluate() cannot be called from a running event loop; use evaluate_async().")


@dataclass
class RetrievalAndAnswerQualitySample:
    """One offline RAG evaluation example."""

    user_input: str
    retrieved_contexts: list[str] = field(default_factory=list)
    response: str | None = None
    reference: str | None = None
    retrieved_context_ids: list[str | int] = field(default_factory=list)
    reference_context_ids: list[str | int] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)

    def to_single_turn_sample(self) -> Any:
        """Convert into the legacy Ragas sample shape when available."""

        if SingleTurnSample is None:
            raise RuntimeError("ragas.SingleTurnSample is not available in this environment.")

        payload: dict[str, Any] = {
            "user_input": self.user_input,
            "retrieved_contexts": list(self.retrieved_contexts),
        }
        if self.response is not None:
            payload["response"] = self.response
        if self.reference is not None:
            payload["reference"] = self.reference
        if self.retrieved_context_ids:
            payload["retrieved_context_ids"] = list(self.retrieved_context_ids)
        if self.reference_context_ids:
            payload["reference_context_ids"] = list(self.reference_context_ids)
        if self.reference_contexts:
            payload["reference_contexts"] = list(self.reference_contexts)
        return SingleTurnSample(**payload)


@dataclass
class MetricSummary:
    """Generic score summary for a metric across many samples."""

    name: str
    total_samples: int
    evaluated_samples: int
    scores: list[float]
    summary: dict[str, float | int]


@dataclass
class RetrievalAndAnswerQualityReport:
    """Bundle retrieval precision/recall and answer-grounding metrics."""

    context_precision_with_reference: MetricSummary | None
    id_based_context_precision: MetricSummary | None
    context_recall: MetricSummary | None
    id_based_context_recall: MetricSummary | None
    faithfulness: MetricSummary | None
    response_groundedness: MetricSummary | None
    hallucination_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report into plain data."""

        def _metric_to_dict(metric: MetricSummary | None) -> dict[str, Any] | None:
            if metric is None:
                return None
            return {
                "name": metric.name,
                "total_samples": metric.total_samples,
                "evaluated_samples": metric.evaluated_samples,
                "scores": metric.scores,
                "summary": metric.summary,
            }

        return {
            "context_precision_with_reference": _metric_to_dict(self.context_precision_with_reference),
            "id_based_context_precision": _metric_to_dict(self.id_based_context_precision),
            "context_recall": _metric_to_dict(self.context_recall),
            "id_based_context_recall": _metric_to_dict(self.id_based_context_recall),
            "faithfulness": _metric_to_dict(self.faithfulness),
            "response_groundedness": _metric_to_dict(self.response_groundedness),
            "hallucination_rate": self.hallucination_rate,
        }
class RetrievalAndAnswerQualityEvaluator:
    """Evaluate retrieval and grounding metrics with optional Ragas objects."""

    def __init__(
        self,
        *,
        evaluator_llm: Any | None = None,
        context_precision_metric: Any | None = None,
        id_based_context_precision_metric: Any | None = None,
        context_recall_metric: Any | None = None,
        id_based_context_recall_metric: Any | None = None,
        faithfulness_metric: Any | None = None,
        response_groundedness_metric: Any | None = None,
        hallucination_threshold: float = 0.8,
    ) -> None:
        self.evaluator_llm = evaluator_llm
        self.hallucination_threshold = float(hallucination_threshold)

        self.context_precision_metric = context_precision_metric or self._build_context_precision_metric()
        self.id_based_context_precision_metric = id_based_context_precision_metric or self._build_id_based_context_precision_metric()
        self.context_recall_metric = context_recall_metric or self._build_context_recall_metric()
        self.id_based_context_recall_metric = id_based_context_recall_metric or self._build_id_based_context_recall_metric()
        self.faithfulness_metric = faithfulness_metric or self._build_faithfulness_metric()
        self.response_groundedness_metric = response_groundedness_metric or self._build_response_groundedness_metric()

    def _build_context_precision_metric(self) -> Any | None:
        if self.evaluator_llm is None or LLMContextPrecisionWithReference is None:
            return None
        return LLMContextPrecisionWithReference(llm=self.evaluator_llm)

    def _build_id_based_context_precision_metric(self) -> Any | None:
        if IDBasedContextPrecision is None:
            return None
        return IDBasedContextPrecision()

    def _build_context_recall_metric(self) -> Any | None:
        if self.evaluator_llm is None or LLMContextRecall is None:
            return None
        return LLMContextRecall(llm=self.evaluator_llm)

    def _build_id_based_context_recall_metric(self) -> Any | None:
        if IDBasedContextRecall is None:
            return None
        return IDBasedContextRecall()

    def _build_faithfulness_metric(self) -> Any | None:
        if self.evaluator_llm is None or Faithfulness is None:
            return None
        return Faithfulness(llm=self.evaluator_llm)

    def _build_response_groundedness_metric(self) -> Any | None:
        if self.evaluator_llm is None or ResponseGroundedness is None:
            return None
        return ResponseGroundedness(llm=self.evaluator_llm)

    async def _score_content_precision(self, sample: RetrievalAndAnswerQualitySample) -> float:
        if self.context_precision_metric is None:
            raise RuntimeError("Context precision metric is not configured. Pass evaluator_llm or inject a metric.")
        return await _evaluate_metric(
            self.context_precision_metric,
            sample_factory=sample.to_single_turn_sample,
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )

    async def _score_id_precision(self, sample: RetrievalAndAnswerQualitySample) -> float:
        if self.id_based_context_precision_metric is None:
            raise RuntimeError("ID-based context precision metric is not configured.")
        if not sample.retrieved_context_ids or not sample.reference_context_ids:
            raise ValueError("ID-based precision requires retrieved_context_ids and reference_context_ids.")
        return await _evaluate_metric(
            self.id_based_context_precision_metric,
            sample_factory=sample.to_single_turn_sample,
            retrieved_context_ids=sample.retrieved_context_ids,
            reference_context_ids=sample.reference_context_ids,
        )

    async def _score_content_recall(self, sample: RetrievalAndAnswerQualitySample) -> float:
        if self.context_recall_metric is None:
            raise RuntimeError("Context recall metric is not configured. Pass evaluator_llm or inject a metric.")
        return await _evaluate_metric(
            self.context_recall_metric,
            sample_factory=sample.to_single_turn_sample,
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )

    async def _score_id_recall(self, sample: RetrievalAndAnswerQualitySample) -> float:
        if self.id_based_context_recall_metric is None:
            raise RuntimeError("ID-based context recall metric is not configured.")
        if not sample.retrieved_context_ids or not sample.reference_context_ids:
            raise ValueError("ID-based recall requires retrieved_context_ids and reference_context_ids.")
        return await _evaluate_metric(
            self.id_based_context_recall_metric,
            sample_factory=sample.to_single_turn_sample,
            retrieved_context_ids=sample.retrieved_context_ids,
            reference_context_ids=sample.reference_context_ids,
        )

    async def _score_faithfulness(self, sample: RetrievalAndAnswerQualitySample) -> float:
        if self.faithfulness_metric is None:
            raise RuntimeError("Faithfulness metric is not configured. Pass evaluator_llm or inject a metric.")
        if sample.response is None:
            raise ValueError("Faithfulness requires a generated response.")
        return await _evaluate_metric(
            self.faithfulness_metric,
            sample_factory=sample.to_single_turn_sample,
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )

    async def _score_response_groundedness(self, sample: RetrievalAndAnswerQualitySample) -> float:
        if self.response_groundedness_metric is None:
            raise RuntimeError("Response groundedness metric is not configured. Pass evaluator_llm or inject a metric.")
        if sample.response is None:
            raise ValueError("Response groundedness requires a generated response.")
        return await _evaluate_metric(
            self.response_groundedness_metric,
            sample_factory=sample.to_single_turn_sample,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )

    @staticmethod
    def _summary_for(name: str, scores: list[float], total_samples: int) -> MetricSummary | None:
        if not scores:
            return None
        return MetricSummary(
            name=name,
            total_samples=total_samples,
            evaluated_samples=len(scores),
            scores=scores,
            summary=summarize_numbers(scores),
        )

    async def evaluate_async(self, samples: Sequence[RetrievalAndAnswerQualitySample]) -> RetrievalAndAnswerQualityReport:
        """Score a batch of samples with all configured metrics."""

        needs_content_precision = any(sample.reference is not None for sample in samples)
        needs_id_precision = any(sample.retrieved_context_ids and sample.reference_context_ids for sample in samples)
        needs_content_recall = any(sample.reference is not None for sample in samples)
        needs_id_recall = any(sample.retrieved_context_ids and sample.reference_context_ids for sample in samples)
        needs_faithfulness = any(sample.response is not None for sample in samples)

        if needs_content_precision and self.context_precision_metric is None:
            raise RuntimeError("Context precision metric is not configured. Pass evaluator_llm or inject a metric.")
        if needs_id_precision and self.id_based_context_precision_metric is None:
            raise RuntimeError("ID-based context precision metric is not configured.")
        if needs_content_recall and self.context_recall_metric is None:
            raise RuntimeError("Context recall metric is not configured. Pass evaluator_llm or inject a metric.")
        if needs_id_recall and self.id_based_context_recall_metric is None:
            raise RuntimeError("ID-based context recall metric is not configured.")
        if needs_faithfulness and self.faithfulness_metric is None:
            raise RuntimeError("Faithfulness metric is not configured. Pass evaluator_llm or inject a metric.")
        if needs_faithfulness and self.response_groundedness_metric is None:
            raise RuntimeError("Response groundedness metric is not configured. Pass evaluator_llm or inject a metric.")

        content_precision_scores: list[float] = []
        id_precision_scores: list[float] = []
        content_recall_scores: list[float] = []
        id_recall_scores: list[float] = []
        faithfulness_scores: list[float] = []
        groundedness_scores: list[float] = []

        for sample in samples:
            if sample.reference is not None:
                content_precision_scores.append(await self._score_content_precision(sample))
            if sample.retrieved_context_ids and sample.reference_context_ids:
                id_precision_scores.append(await self._score_id_precision(sample))
            if sample.reference is not None:
                content_recall_scores.append(await self._score_content_recall(sample))
            if sample.retrieved_context_ids and sample.reference_context_ids:
                id_recall_scores.append(await self._score_id_recall(sample))
            if sample.response is not None:
                faithfulness_scores.append(await self._score_faithfulness(sample))
            if sample.response is not None:
                groundedness_scores.append(await self._score_response_groundedness(sample))

        faithfulness_summary = self._summary_for("faithfulness", faithfulness_scores, len(samples))
        hallucination_rate = None
        if faithfulness_summary is not None:
            hallucination_rate = 1.0 - float(faithfulness_summary.summary["mean"])

        return RetrievalAndAnswerQualityReport(
            context_precision_with_reference=self._summary_for("context_precision_with_reference", content_precision_scores, len(samples)),
            id_based_context_precision=self._summary_for("id_based_context_precision", id_precision_scores, len(samples)),
            context_recall=self._summary_for("context_recall", content_recall_scores, len(samples)),
            id_based_context_recall=self._summary_for("id_based_context_recall", id_recall_scores, len(samples)),
            faithfulness=faithfulness_summary,
            response_groundedness=self._summary_for("response_groundedness", groundedness_scores, len(samples)),
            hallucination_rate=hallucination_rate,
        )

    def evaluate(self, samples: Sequence[RetrievalAndAnswerQualitySample]) -> RetrievalAndAnswerQualityReport:
        """Synchronously score a batch of samples."""

        return _run_coroutine(self.evaluate_async(samples))
