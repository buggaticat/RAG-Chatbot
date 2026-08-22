"""Embedding drift, query similarity, and index health evaluation tools."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..utils import summarize_numbers, utc_now


DEFAULT_INDEX_EVENT_LOG = Path(os.getenv("EVAL_INDEX_EVENT_LOG", ".eval_index_events.jsonl"))
DEFAULT_EMBEDDING_DRIFT_PATH = Path(os.getenv("EMBEDDING_DRIFT_PATH", ".embedding_drift.json"))


def _vector_norm(vector: Sequence[float]) -> float:
    """Return the L2 norm of a vector."""

    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Compute the mean vector for a collection of embeddings."""

    if not vectors:
        return []
    width = min(len(vector) for vector in vectors)
    if width == 0:
        return []
    return [
        sum(float(vector[index]) for vector in vectors) / len(vectors)
        for index in range(width)
    ]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Compute cosine similarity for two vectors when dimensions match."""

    if not left or not right or len(left) != len(right):
        return None
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        return None
    dot_product = sum(float(l) * float(r) for l, r in zip(left, right))
    return dot_product / (left_norm * right_norm)


@dataclass
class EmbeddingSnapshot:
    """Summary statistics for a set of embeddings."""

    count: int
    mean_norm: float
    std_norm: float
    median_norm: float
    p95_norm: float
    min_norm: float
    max_norm: float
    centroid: list[float]
    label: str = "current"
    generated_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_vectors(cls, vectors: Sequence[Sequence[float]], *, label: str = "current") -> "EmbeddingSnapshot":
        """Build a snapshot from raw embedding vectors."""

        norms = [_vector_norm(vector) for vector in vectors]
        stats = summarize_numbers(norms)
        return cls(
            count=int(stats["count"]),
            mean_norm=float(stats["mean"]),
            std_norm=float(stats["std"]),
            median_norm=float(stats["p50"]),
            p95_norm=float(stats["p95"]),
            min_norm=float(stats["min"]),
            max_norm=float(stats["max"]),
            centroid=_centroid(vectors),
            label=label,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, label: str = "baseline") -> "EmbeddingSnapshot":
        """Build a snapshot from a persisted JSON payload."""

        return cls(
            count=int(payload.get("count", 0)),
            mean_norm=float(payload.get("mean_norm", 0.0)),
            std_norm=float(payload.get("std_norm", 0.0)),
            median_norm=float(payload.get("median_norm", 0.0)),
            p95_norm=float(payload.get("p95_norm", 0.0)),
            min_norm=float(payload.get("min_norm", 0.0)),
            max_norm=float(payload.get("max_norm", 0.0)),
            centroid=[float(value) for value in payload.get("centroid", []) or []],
            label=label,
            generated_at=utc_now(),
        )

    @classmethod
    def load_from_file(cls, path: str | Path = DEFAULT_EMBEDDING_DRIFT_PATH) -> "EmbeddingSnapshot | None":
        """Load a baseline snapshot from disk."""

        file_path = Path(path)
        if not file_path.exists():
            return None
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None

        for key in ("golden_eval", "baseline"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                return cls.from_payload(candidate, label=key)
        return None

    @classmethod
    def load_current_from_file(cls, path: str | Path = DEFAULT_EMBEDDING_DRIFT_PATH) -> "EmbeddingSnapshot | None":
        """Load the most recent current snapshot from disk when the live checkpoint is gone."""

        file_path = Path(path)
        if not file_path.exists():
            return None
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None

        candidate = payload.get("current")
        if isinstance(candidate, dict) and candidate:
            return cls.from_payload(candidate, label="current")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Render the snapshot as JSON-safe data."""

        return {
            "count": self.count,
            "mean_norm": self.mean_norm,
            "std_norm": self.std_norm,
            "median_norm": self.median_norm,
            "p95_norm": self.p95_norm,
            "min_norm": self.min_norm,
            "max_norm": self.max_norm,
            "centroid": self.centroid,
            "label": self.label,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class EmbeddingDriftReport:
    """Compare a current embedding snapshot against a baseline."""

    current: EmbeddingSnapshot
    baseline: EmbeddingSnapshot | None
    count_delta: int
    mean_norm_delta: float
    std_norm_delta: float
    median_norm_delta: float
    p95_norm_delta: float
    min_norm_delta: float
    max_norm_delta: float
    centroid_cosine_similarity: float | None
    centroid_cosine_distance: float | None
    drift_score: float

    @classmethod
    def compare(cls, current: EmbeddingSnapshot, baseline: EmbeddingSnapshot | None) -> "EmbeddingDriftReport":
        """Compute the embedding drift metrics for two snapshots."""

        if baseline is None:
            return cls(
                current=current,
                baseline=None,
                count_delta=current.count,
                mean_norm_delta=current.mean_norm,
                std_norm_delta=current.std_norm,
                median_norm_delta=current.median_norm,
                p95_norm_delta=current.p95_norm,
                min_norm_delta=current.min_norm,
                max_norm_delta=current.max_norm,
                centroid_cosine_similarity=None,
                centroid_cosine_distance=None,
                drift_score=0.0,
            )

        centroid_similarity = _cosine_similarity(current.centroid, baseline.centroid)
        centroid_distance = None if centroid_similarity is None else 1.0 - centroid_similarity
        drift_components = [
            abs(current.mean_norm - baseline.mean_norm),
            abs(current.std_norm - baseline.std_norm),
            abs(current.median_norm - baseline.median_norm),
            abs(current.p95_norm - baseline.p95_norm),
            abs(current.min_norm - baseline.min_norm),
            abs(current.max_norm - baseline.max_norm),
        ]
        drift_score = sum(drift_components) / len(drift_components) if drift_components else 0.0
        if centroid_distance is not None:
            drift_score = (drift_score + centroid_distance) / 2.0

        return cls(
            current=current,
            baseline=baseline,
            count_delta=current.count - baseline.count,
            mean_norm_delta=current.mean_norm - baseline.mean_norm,
            std_norm_delta=current.std_norm - baseline.std_norm,
            median_norm_delta=current.median_norm - baseline.median_norm,
            p95_norm_delta=current.p95_norm - baseline.p95_norm,
            min_norm_delta=current.min_norm - baseline.min_norm,
            max_norm_delta=current.max_norm - baseline.max_norm,
            centroid_cosine_similarity=centroid_similarity,
            centroid_cosine_distance=centroid_distance,
            drift_score=drift_score,
        )

    @classmethod
    def from_vectors(
        cls,
        vectors: Sequence[Sequence[float]],
        *,
        baseline_path: str | Path = DEFAULT_EMBEDDING_DRIFT_PATH,
    ) -> "EmbeddingDriftReport":
        """Convenience constructor for the current embeddings and a persisted baseline."""

        current = EmbeddingSnapshot.from_vectors(vectors)
        baseline = EmbeddingSnapshot.load_from_file(baseline_path)
        return cls.compare(current, baseline)

    def to_dict(self) -> dict[str, Any]:
        """Render the drift report as plain data."""

        return {
            "current": {
                "count": self.current.count,
                "mean_norm": self.current.mean_norm,
                "std_norm": self.current.std_norm,
                "median_norm": self.current.median_norm,
                "p95_norm": self.current.p95_norm,
                "min_norm": self.current.min_norm,
                "max_norm": self.current.max_norm,
                "centroid": self.current.centroid,
                "label": self.current.label,
            },
            "baseline": None
            if self.baseline is None
            else {
                "count": self.baseline.count,
                "mean_norm": self.baseline.mean_norm,
                "std_norm": self.baseline.std_norm,
                "median_norm": self.baseline.median_norm,
                "p95_norm": self.baseline.p95_norm,
                "min_norm": self.baseline.min_norm,
                "max_norm": self.baseline.max_norm,
                "centroid": self.baseline.centroid,
                "label": self.baseline.label,
            },
            "count_delta": self.count_delta,
            "mean_norm_delta": self.mean_norm_delta,
            "std_norm_delta": self.std_norm_delta,
            "median_norm_delta": self.median_norm_delta,
            "p95_norm_delta": self.p95_norm_delta,
            "min_norm_delta": self.min_norm_delta,
            "max_norm_delta": self.max_norm_delta,
            "centroid_cosine_similarity": self.centroid_cosine_similarity,
            "centroid_cosine_distance": self.centroid_cosine_distance,
            "drift_score": self.drift_score,
        }


@dataclass
class QuerySimilarityReport:
    """Describe the distribution of retrieval similarity scores."""

    total_queries: int
    total_scores: int
    score_summary: dict[str, float | int]
    top1_summary: dict[str, float | int]
    top1_top2_margin_summary: dict[str, float | int]
    high_similarity_rate: float
    per_query_score_summary: list[dict[str, float | int]]

    @classmethod
    def from_query_scores(
        cls,
        query_scores: Sequence[Sequence[float]],
        *,
        high_similarity_threshold: float = 0.75,
    ) -> "QuerySimilarityReport":
        """Build a distribution report from per-query similarity lists."""

        flattened = [float(score) for scores in query_scores for score in scores]
        top1_scores: list[float] = []
        top1_top2_margins: list[float] = []
        per_query_score_summary: list[dict[str, float | int]] = []

        for scores in query_scores:
            score_list = sorted((float(score) for score in scores), reverse=True)
            if not score_list:
                per_query_score_summary.append(
                    {
                        "count": 0,
                        "mean": 0.0,
                        "p50": 0.0,
                        "p95": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }
                )
                continue
            top1_scores.append(score_list[0])
            if len(score_list) > 1:
                top1_top2_margins.append(score_list[0] - score_list[1])
            per_query_score_summary.append(
                {
                    "count": len(score_list),
                    "mean": sum(score_list) / len(score_list),
                    "p50": summarize_numbers(score_list)["p50"],
                    "p95": summarize_numbers(score_list)["p95"],
                    "min": min(score_list),
                    "max": max(score_list),
                }
            )

        score_summary = summarize_numbers(flattened)
        top1_summary = summarize_numbers(top1_scores)
        margin_summary = summarize_numbers(top1_top2_margins)
        high_similarity_rate = (
            sum(1 for scores in query_scores if scores and max(float(score) for score in scores) >= high_similarity_threshold)
            / len(query_scores)
            if query_scores
            else 0.0
        )

        return cls(
            total_queries=len(query_scores),
            total_scores=len(flattened),
            score_summary=score_summary,
            top1_summary=top1_summary,
            top1_top2_margin_summary=margin_summary,
            high_similarity_rate=high_similarity_rate,
            per_query_score_summary=per_query_score_summary,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the similarity distribution."""

        return {
            "total_queries": self.total_queries,
            "total_scores": self.total_scores,
            "score_summary": self.score_summary,
            "top1_summary": self.top1_summary,
            "top1_top2_margin_summary": self.top1_top2_margin_summary,
            "high_similarity_rate": self.high_similarity_rate,
            "per_query_score_summary": self.per_query_score_summary,
        }


@dataclass
class IndexHealthReport:
    """Snapshot of vector index size and operational health."""

    collection_name: str
    point_count: int
    vector_count: int
    indexed_vector_count: int
    segments_count: int
    vector_size: int | None
    estimated_vector_mb: float | None
    status: str
    optimizer_status: str
    fragmentation_ratio: float
    error_event_count: int
    error_rate: float
    error_events: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the index health report."""

        return {
            "collection_name": self.collection_name,
            "point_count": self.point_count,
            "vector_count": self.vector_count,
            "indexed_vector_count": self.indexed_vector_count,
            "segments_count": self.segments_count,
            "vector_size": self.vector_size,
            "estimated_vector_mb": self.estimated_vector_mb,
            "status": self.status,
            "optimizer_status": self.optimizer_status,
            "fragmentation_ratio": self.fragmentation_ratio,
            "error_event_count": self.error_event_count,
            "error_rate": self.error_rate,
            "error_events": self.error_events,
            "notes": self.notes,
        }


@dataclass
class IndexEventLedger:
    """Persist and summarize index health events for later evaluation."""

    path: Path = DEFAULT_INDEX_EVENT_LOG

    def record_event(
        self,
        event_type: str,
        *,
        success: bool,
        stage: str | None = None,
        message: str | None = None,
        error_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an operational event to the ledger."""

        event = {
            "timestamp": utc_now().isoformat(),
            "event_type": event_type,
            "stage": stage,
            "success": bool(success),
            "message": message,
            "error_type": error_type,
            "metadata": metadata or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def load_events(self) -> list[dict[str, Any]]:
        """Read all stored events from disk."""

        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except Exception:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def error_rate(self, *, event_type: str | None = None) -> float:
        """Calculate the failure rate for the stored events."""

        events = self.load_events()
        if event_type is not None:
            events = [event for event in events if event.get("event_type") == event_type]
        if not events:
            return 0.0
        failures = sum(1 for event in events if not bool(event.get("success", False)))
        return failures / len(events)


class IndexHealthInspector:
    """Inspect Qdrant collection health and combine it with local error telemetry."""

    def __init__(
        self,
        *,
        collection_name: str,
        ledger: IndexEventLedger | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.ledger = ledger or IndexEventLedger()

    @staticmethod
    def _read_nested_value(root: Any, *keys: str) -> Any:
        """Read a nested attribute or mapping value."""

        current = root
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    @classmethod
    def _extract_collection_info(cls, collection_info: Any) -> Any:
        """Handle both raw and wrapped Qdrant collection info payloads."""

        return getattr(collection_info, "result", collection_info)

    @classmethod
    def _extract_vector_size(cls, collection_info: Any) -> int | None:
        """Extract the configured vector size when it is available."""

        payload = cls._extract_collection_info(collection_info)
        config = cls._read_nested_value(payload, "config")
        params = cls._read_nested_value(config, "params")
        vectors = cls._read_nested_value(params, "vectors")

        if isinstance(vectors, dict):
            if "size" in vectors:
                return int(vectors["size"])
            for value in vectors.values():
                size = cls._read_nested_value(value, "size")
                if size is not None:
                    return int(size)
        size = cls._read_nested_value(vectors, "size")
        return int(size) if size is not None else None

    @classmethod
    def _extract_health_fields(cls, collection_info: Any) -> dict[str, Any]:
        """Pull the collection health metrics out of a Qdrant response."""

        payload = cls._extract_collection_info(collection_info)
        point_count = int(cls._read_nested_value(payload, "points_count") or 0)
        vector_count = int(cls._read_nested_value(payload, "vectors_count") or point_count or 0)
        indexed_vector_count = int(cls._read_nested_value(payload, "indexed_vectors_count") or 0)
        segments_count = int(cls._read_nested_value(payload, "segments_count") or 0)
        status = str(cls._read_nested_value(payload, "status") or "unknown")
        optimizer_status = str(cls._read_nested_value(payload, "optimizer_status") or "unknown")
        vector_size = cls._extract_vector_size(collection_info)
        estimated_vector_mb = None
        if vector_size is not None and vector_count > 0:
            estimated_vector_mb = (vector_count * vector_size * 4) / (1024 * 1024)

        if vector_count > 0 and indexed_vector_count > 0:
            fragmentation_ratio = max(0.0, 1.0 - (indexed_vector_count / vector_count))
        elif point_count > 0 and vector_count > 0:
            fragmentation_ratio = max(0.0, 1.0 - (vector_count / point_count))
        else:
            fragmentation_ratio = 0.0

        return {
            "point_count": point_count,
            "vector_count": vector_count,
            "indexed_vector_count": indexed_vector_count,
            "segments_count": segments_count,
            "vector_size": vector_size,
            "estimated_vector_mb": estimated_vector_mb,
            "status": status,
            "optimizer_status": optimizer_status,
            "fragmentation_ratio": fragmentation_ratio,
        }

    def inspect(self, client: Any) -> IndexHealthReport:
        """Inspect the configured collection and aggregate local error telemetry."""

        collection_info = client.get_collection(collection_name=self.collection_name)
        fields = self._extract_health_fields(collection_info)
        events = self.ledger.load_events()
        event_type = "qdrant_ingestion"
        error_events = [
            event
            for event in events
            if event.get("event_type") == event_type and not bool(event.get("success", False))
        ]
        error_rate = self.ledger.error_rate(event_type=event_type)

        notes: list[str] = []
        if fields["status"] != "green" and fields["status"] != "yellow" and fields["status"] != "unknown":
            notes.append(f"Collection status reported as {fields['status']}.")
        if fields["fragmentation_ratio"] > 0.2:
            notes.append("Fragmentation proxy is elevated; consider compaction or reindexing.")
        if error_rate > 0:
            notes.append("Index error telemetry contains failures.")

        return IndexHealthReport(
            collection_name=self.collection_name,
            point_count=fields["point_count"],
            vector_count=fields["vector_count"],
            indexed_vector_count=fields["indexed_vector_count"],
            segments_count=fields["segments_count"],
            vector_size=fields["vector_size"],
            estimated_vector_mb=fields["estimated_vector_mb"],
            status=fields["status"],
            optimizer_status=fields["optimizer_status"],
            fragmentation_ratio=fields["fragmentation_ratio"],
            error_event_count=len(error_events),
            error_rate=error_rate,
            error_events=error_events,
            notes=notes,
        )
