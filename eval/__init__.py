"""Evaluation layer for the RAG chatbot."""

from __future__ import annotations

from .utils import EvaluationReport
from .cost_and_usage import CostUsageRecorder, UsageTrackingModelProxy
from .embedding_and_index_health import (
    EmbeddingDriftReport,
    EmbeddingSnapshot,
    IndexEventLedger,
    IndexHealthInspector,
    IndexHealthReport,
    QuerySimilarityReport,
)
from .latency_and_reliability import LatencyRecorder, LatencyReport
from .retrieval_and_answer_quality import (
    MetricSummary,
    RetrievalAndAnswerQualityEvaluator,
    RetrievalAndAnswerQualityReport,
    RetrievalAndAnswerQualitySample,
)
from .runner import RAGEvaluationSuite

__all__ = [
    "CostUsageRecorder",
    "EmbeddingDriftReport",
    "EmbeddingSnapshot",
    "EvaluationReport",
    "IndexEventLedger",
    "IndexHealthInspector",
    "IndexHealthReport",
    "LatencyRecorder",
    "LatencyReport",
    "RetrievalAndAnswerQualityEvaluator",
    "RetrievalAndAnswerQualityReport",
    "RetrievalAndAnswerQualitySample",
    "MetricSummary",
    "RAGEvaluationSuite",
    "QuerySimilarityReport",
    "UsageTrackingModelProxy",
]
