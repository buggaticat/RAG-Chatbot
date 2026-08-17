"""High-level orchestration for the RAG evaluation layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import EvaluationReport
from .cost_and_usage import CostUsageRecorder
from .embedding_and_index_health import EmbeddingDriftReport, IndexHealthReport, QuerySimilarityReport
from .latency_and_reliability import LatencyRecorder
from .retrieval_and_answer_quality import RetrievalAndAnswerQualityReport


@dataclass
class RAGEvaluationSuite:
    """Bundle all requested eval sections into one report."""

    latency_recorder: LatencyRecorder | None = None
    cost_recorder: CostUsageRecorder | None = None
    embedding_drift_report: EmbeddingDriftReport | None = None
    query_similarity_report: QuerySimilarityReport | None = None
    index_health_report: IndexHealthReport | None = None
    retrieval_and_answer_quality_report: RetrievalAndAnswerQualityReport | None = None

    def build_report(self) -> EvaluationReport:
        """Build a single report containing all configured eval sections."""

        latency_report = self.latency_recorder.summarize() if self.latency_recorder is not None else None
        cost_report = self.cost_recorder.summarize() if self.cost_recorder is not None else None
        embedding_health = {
            "embedding_drift": None if self.embedding_drift_report is None else self.embedding_drift_report.to_dict(),
            "query_similarity": None if self.query_similarity_report is None else self.query_similarity_report.to_dict(),
            "index_health": None if self.index_health_report is None else self.index_health_report.to_dict(),
        }
        return EvaluationReport(
            latency_and_reliability=latency_report,
            cost_and_usage=cost_report,
            embedding_and_index_health=embedding_health,
            retrieval_and_answer_quality=(
                None
                if self.retrieval_and_answer_quality_report is None
                else self.retrieval_and_answer_quality_report.to_dict()
            ),
        )
