"""Helpers for building CLI eval reports from local files and runtime inputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from eval import (
    CostUsageRecorder,
    EmbeddingDriftReport,
    EvaluationReport,
    IndexEventLedger,
    IndexHealthInspector,
    IndexHealthReport,
    LatencyRecorder,
    QuerySimilarityReport,
    RAGEvaluationSuite,
    RetrievalAndAnswerQualityEvaluator,
    RetrievalAndAnswerQualityReport,
    RetrievalAndAnswerQualitySample,
)
from eval.embedding_and_index_health.metrics import DEFAULT_INDEX_EVENT_LOG
from eval.embedding_and_index_health.metrics import DEFAULT_EMBEDDING_DRIFT_PATH
from eval.utils import ensure_serializable
from rag.retrieval.config import COLLECTION_NAME, QDRANT_APIKEY, QDRANT_CLUSTER_ENDPOINT
from eval.cost_and_usage.metrics import ModelPricing


REPORT_CHOICES = (
    "latency",
    "cost",
    "embedding-drift",
    "query-similarity",
    "index-health",
    "retrieval-answer-quality",
)

DEFAULT_LOG_PATH = Path(os.getenv("CHATBOT_CLI_LOG_PATH", ".chatbot_cli_runs.jsonl"))

GOLDEN_DATASET_DIR = Path(
    os.getenv("GOLDEN_DATASET_DIR", str(Path(__file__).resolve().parent.parent / "eval" / "golden_dataset"))
)


@dataclass
class BuiltEvalReports:
    """Container for all CLI-built eval report objects."""

    report: EvaluationReport
    latency_report: Any | None = None
    cost_report: Any | None = None
    embedding_drift_report: EmbeddingDriftReport | None = None
    query_similarity_report: QuerySimilarityReport | None = None
    index_health_report: IndexHealthReport | None = None
    retrieval_and_answer_quality_report: RetrievalAndAnswerQualityReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the entire report bundle into JSON-safe data."""

        return ensure_serializable(
            {
                "report": self.report,
                "latency_report": self.latency_report,
                "cost_report": self.cost_report,
                "embedding_drift_report": self.embedding_drift_report,
                "query_similarity_report": self.query_similarity_report,
                "index_health_report": self.index_health_report,
                "retrieval_and_answer_quality_report": self.retrieval_and_answer_quality_report,
            }
        )


def load_json_or_jsonl(path: str | Path) -> list[Any]:
    """Load JSON or JSONL data from disk."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    items: list[Any] = []
    lines = [raw_line.strip() for raw_line in text.splitlines() if raw_line.strip()]
    if len(lines) > 1:
        for line in lines:
            items.append(json.loads(line))
        return items

    if text.startswith("[") or text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        return [payload]

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def load_trace_records(path: str | Path) -> list[dict[str, Any]]:
    """Load chatbot run traces from the local JSONL ledger."""

    records: list[dict[str, Any]] = []
    for item in load_json_or_jsonl(path):
        if isinstance(item, dict):
            records.append(item)
    return records


def load_golden_queries(dataset_dir: str | Path = GOLDEN_DATASET_DIR) -> list[dict[str, Any]]:
    """Load the golden queries used to generate retrieval-quality traces."""

    _, query_records, _ = _load_named_payloads(dataset_dir)
    return query_records


def load_embedding_vectors(path: str | Path) -> list[list[float]]:
    """Load embedding vectors from JSON or JSONL."""

    vectors: list[list[float]] = []
    for item in load_json_or_jsonl(path):
        candidate = item
        if isinstance(candidate, dict):
            for key in ("vector", "embedding", "values"):
                value = candidate.get(key)
                if isinstance(value, list):
                    candidate = value
                    break
        if isinstance(candidate, list):
            vector = [float(value) for value in candidate]
            if vector:
                vectors.append(vector)
    return vectors


def _normalize_key(value: Any) -> str:
    """Collapse a matching key into a stable lowercase string."""

    return " ".join(str(value).strip().split()).lower()


def _extract_record_value(record: Any, *keys: str) -> Any:
    """Return the first populated value across a list of possible keys."""

    if not isinstance(record, dict):
        return None
    for key in keys:
        value = record.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _load_named_payloads(directory: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load qrels, queries, and answers payloads from a dataset directory."""

    base = Path(directory)
    qrels_path = base / "qrels.json"
    queries_path = base / "queries.json"
    answers_path = base / "answers.json"
    if not qrels_path.exists() or not queries_path.exists() or not answers_path.exists():
        raise FileNotFoundError(
            f"Golden dataset files are missing under {base}. Expected qrels.json, queries.json, and answers.json."
        )

    def _as_records(path: Path) -> list[dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "items", "queries", "answers", "qrels", "examples", "records"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return [item for item in candidate if isinstance(item, dict)]
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    return _as_records(qrels_path), _as_records(queries_path), _as_records(answers_path)


def _build_retrieval_samples_from_defaults(
    trace_records: Sequence[dict[str, Any]],
    *,
    dataset_dir: str | Path = GOLDEN_DATASET_DIR,
) -> list[RetrievalAndAnswerQualitySample]:
    """Build retrieval-quality samples from the golden dataset and local chatbot traces."""

    qrels_records, query_records, answer_records = _load_named_payloads(dataset_dir)

    trace_by_query_id: dict[str, dict[str, Any]] = {}
    trace_by_query_text: dict[str, dict[str, Any]] = {}
    for record in trace_records:
        query_id = _normalize_key(_extract_record_value(record, "query_id", "id", "run_id"))
        if query_id:
            trace_by_query_id[query_id] = record
        query_text = _normalize_key(_extract_record_value(record, "user_query", "query", "prompt"))
        if query_text:
            trace_by_query_text[query_text] = record

    answer_by_query_id: dict[str, Any] = {}
    answer_by_query_text: dict[str, Any] = {}
    for record in answer_records:
        query_id = _normalize_key(_extract_record_value(record, "query_id", "id"))
        if query_id:
            answer_by_query_id[query_id] = _extract_record_value(record, "answer", "reference", "gold_answer", "output")
        question_text = _normalize_key(_extract_record_value(record, "question", "query", "user_input", "prompt"))
        if question_text:
            answer_by_query_text[question_text] = _extract_record_value(record, "answer", "reference", "gold_answer", "output")

    qrels_by_query_id: dict[str, list[str]] = {}
    qrels_by_query_text: dict[str, list[str]] = {}
    for record in qrels_records:
        query_id = _normalize_key(_extract_record_value(record, "query_id", "id"))
        relevant = _extract_record_value(record, "relevant_context_ids", "doc_ids", "context_ids", "retrieved_context_ids")
        if isinstance(relevant, list):
            relevant_ids = [str(value) for value in relevant if value not in (None, "")]
        elif relevant not in (None, "", [], {}):
            relevant_ids = [str(relevant)]
        else:
            relevant_ids = []
        if query_id:
            qrels_by_query_id[query_id] = relevant_ids
        query_text = _normalize_key(_extract_record_value(record, "question", "query", "user_input", "prompt"))
        if query_text:
            qrels_by_query_text[query_text] = relevant_ids

    samples: list[RetrievalAndAnswerQualitySample] = []
    for record in query_records:
        query_id = _normalize_key(_extract_record_value(record, "query_id", "id"))
        question = _extract_record_value(record, "question", "query", "user_input", "prompt", "text")
        if question in (None, ""):
            continue

        trace = trace_by_query_id.get(query_id) or trace_by_query_text.get(_normalize_key(question))
        if trace is None:
            continue

        reference = answer_by_query_id.get(query_id) or answer_by_query_text.get(_normalize_key(question))
        if reference in (None, ""):
            continue

        retrieved_contexts = [
            str(chunk.get("text", "")).strip()
            for chunk in trace.get("retrieved_chunks", []) or []
            if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
        ]
        retrieved_context_ids = [
            str(chunk.get("chunk_id") or chunk.get("doc_id") or chunk.get("paper_id") or idx)
            for idx, chunk in enumerate(trace.get("retrieved_chunks", []) or [], start=1)
            if isinstance(chunk, dict)
        ]
        reference_context_ids = qrels_by_query_id.get(query_id) or qrels_by_query_text.get(_normalize_key(question)) or []

        samples.append(
            RetrievalAndAnswerQualitySample(
                user_input=str(question),
                retrieved_contexts=retrieved_contexts,
                response=str(trace.get("final_answer", "")),
                reference=str(reference),
                retrieved_context_ids=retrieved_context_ids,
                reference_context_ids=reference_context_ids,
                reference_contexts=[str(reference)],
            )
        )

    return samples


def _record_scores_from_trace(record: dict[str, Any]) -> list[float]:
    """Extract retrieved-chunk scores from one trace record."""

    scores: list[float] = []
    for chunk in record.get("retrieved_chunks", []) or []:
        score = chunk.get("score") if isinstance(chunk, dict) else None
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    return scores


def build_latency_and_cost_recorders(records: Sequence[dict[str, Any]]) -> tuple[LatencyRecorder, CostUsageRecorder]:
    """Build the latency and cost recorders from stored chatbot traces."""

    latency_recorder = LatencyRecorder()
    cost_recorder = CostUsageRecorder(
        pricing={
            "chatbot_cli": ModelPricing(
                input_cost_per_1k=0.00075,
                output_cost_per_1k=0.0045,
            )
        }
    )

    for record in records:
        duration_s = float(record.get("duration_s", 0.0) or 0.0)
        latency_recorder.record_request(
            duration_s,
            success=True,
            accepted=bool(record.get("accepted", False)),
            metadata={
                "run_id": record.get("run_id"),
                "retry_count": record.get("retry_count", 0),
                "retrieved_chunk_count": len(record.get("retrieved_chunks", []) or []),
            },
        )

        for sample in record.get("component_timings", []) or []:
            if not isinstance(sample, dict):
                continue
            component = str(sample.get("component", "")).strip()
            if not component:
                continue
            try:
                component_duration = float(sample.get("duration_s", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            latency_recorder.record_component(
                component,
                component_duration,
                metadata={"run_id": record.get("run_id")},
            )

        prompt_tokens = int(record.get("prompt_tokens_estimate", 0) or 0)
        answer_tokens = int(record.get("answer_tokens_estimate", 0) or 0)
        cost_recorder.record(
            "chatbot_cli",
            prompt_tokens,
            answer_tokens,
            metadata={
                "run_id": record.get("run_id"),
                "accepted": bool(record.get("accepted", False)),
            },
        )

    return latency_recorder, cost_recorder


def build_latency_and_cost_report(records: Sequence[dict[str, Any]]) -> tuple[Any | None, Any | None, EvaluationReport]:
    """Build latency and cost reporting from stored chatbot traces."""

    latency_recorder, cost_recorder = build_latency_and_cost_recorders(records)
    suite = RAGEvaluationSuite(latency_recorder=latency_recorder, cost_recorder=cost_recorder)
    report = suite.build_report()
    return report.latency_and_reliability, report.cost_and_usage, report


def build_query_similarity_report(records: Sequence[dict[str, Any]]) -> QuerySimilarityReport:
    """Build a query similarity distribution from trace chunk scores."""

    query_scores = [_record_scores_from_trace(record) for record in records]
    return QuerySimilarityReport.from_query_scores(query_scores)


def build_embedding_drift_report(
    *,
    baseline_path: str | Path | None = None,
) -> EmbeddingDriftReport:
    """Build the embedding drift report from the local checkpoint and baseline snapshot."""

    from eval.embedding_and_index_health.metrics import EmbeddingSnapshot
    from rag.ingestion.embed import _load_checkpoint, get_checkpoint_embeddings

    checkpoint = _load_checkpoint()
    vectors = [vector for _, vector, _ in get_checkpoint_embeddings(checkpoint)]
    current_snapshot = EmbeddingSnapshot.from_vectors(vectors)
    if current_snapshot.count == 0:
        fallback_snapshot = EmbeddingSnapshot.load_current_from_file(DEFAULT_EMBEDDING_DRIFT_PATH)
        if fallback_snapshot is not None:
            current_snapshot = fallback_snapshot
    if baseline_path is None:
        baseline = EmbeddingSnapshot.load_from_file(DEFAULT_EMBEDDING_DRIFT_PATH)
    else:
        baseline = EmbeddingSnapshot.load_from_file(baseline_path)
    return EmbeddingDriftReport.compare(current_snapshot, baseline)


def build_index_health_report(
    *,
    collection_name: str = COLLECTION_NAME,
    qdrant_url: str | None = QDRANT_CLUSTER_ENDPOINT,
    qdrant_api_key: str | None = QDRANT_APIKEY,
    index_event_log: str | Path | None = DEFAULT_INDEX_EVENT_LOG,
) -> IndexHealthReport:
    """Inspect a Qdrant collection and combine it with local index telemetry."""

    from qdrant_client import QdrantClient

    ledger = IndexEventLedger(path=Path(index_event_log) if index_event_log is not None else IndexEventLedger().path)
    inspector = IndexHealthInspector(collection_name=collection_name, ledger=ledger)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    return inspector.inspect(client)


def build_retrieval_and_answer_quality_report(
    *,
    evaluator_llm: Any,
    trace_records: Sequence[dict[str, Any]],
    dataset_dir: str | Path = GOLDEN_DATASET_DIR,
) -> RetrievalAndAnswerQualityReport:
    """Build retrieval and grounded-answer quality metrics from local traces and golden data."""

    samples = _build_retrieval_samples_from_defaults(trace_records, dataset_dir=dataset_dir)
    evaluator = RetrievalAndAnswerQualityEvaluator(evaluator_llm=evaluator_llm)
    return evaluator.evaluate(samples)


def parse_selected_reports(requested: Sequence[str] | None) -> list[str]:
    """Normalize report selection arguments into a stable ordered list."""

    if not requested:
        return []

    requested_set = {value.strip().lower() for value in requested if value and value.strip()}
    if "all" in requested_set:
        return list(REPORT_CHOICES)

    selected: list[str] = []
    for choice in REPORT_CHOICES:
        if choice in requested_set:
            selected.append(choice)
    return selected


def estimate_trace_tokens(records: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Estimate prompt and completion token usage from stored traces."""

    prompt_tokens = 0
    answer_tokens = 0
    for record in records:
        prompt_tokens += int(record.get("prompt_tokens_estimate", 0) or 0)
        answer_tokens += int(record.get("answer_tokens_estimate", 0) or 0)
    return prompt_tokens, answer_tokens


def build_query_similarity_text(report: QuerySimilarityReport) -> str:
    """Render a concise human-readable summary for query similarity."""

    data = report.to_dict()
    return (
        f"Queries: {data['total_queries']}\n"
        f"Scores: {data['total_scores']}\n"
        f"Score mean: {data['score_summary']['mean']:.4f}\n"
        f"Top-1 mean: {data['top1_summary']['mean']:.4f}\n"
        f"High similarity rate: {data['high_similarity_rate']:.2%}"
    )
