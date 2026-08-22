from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from eval.embedding_and_index_health import (
    EmbeddingDriftReport,
    EmbeddingSnapshot,
    IndexEventLedger,
    IndexHealthInspector,
    QuerySimilarityReport,
)


def test_embedding_drift_metrics_compare_against_baseline(tmp_path: Path):
    baseline_path = tmp_path / ".embedding_drift.json"
    baseline_payload = {
        "golden_eval": {
            "count": 2,
            "mean_norm": 1.0,
            "std_norm": 0.0,
            "median_norm": 1.0,
            "p95_norm": 1.0,
            "min_norm": 1.0,
            "max_norm": 1.0,
            "centroid": [1.0, 0.0],
        }
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    report = EmbeddingDriftReport.from_vectors([[0.8, 0.6], [0.6, 0.8]], baseline_path=baseline_path)

    assert report.current.count == 2
    assert report.baseline is not None
    assert report.baseline.label == "golden_eval"
    assert report.count_delta == 0
    assert report.centroid_cosine_similarity is not None
    assert report.drift_score >= 0


def test_embedding_drift_metrics_without_baseline_return_none(tmp_path: Path):
    baseline_path = tmp_path / ".embedding_drift.json"
    baseline_path.write_text(
        json.dumps(
            {
                "current": {
                    "count": 2,
                    "mean_norm": 1.0,
                    "std_norm": 0.0,
                    "median_norm": 1.0,
                    "p95_norm": 1.0,
                    "min_norm": 1.0,
                    "max_norm": 1.0,
                    "centroid": [1.0, 0.0],
                },
                "golden_eval": {},
            }
        ),
        encoding="utf-8",
    )

    report = EmbeddingDriftReport.from_vectors([[0.8, 0.6], [0.6, 0.8]], baseline_path=baseline_path)

    assert report.baseline is None


def test_query_similarity_distribution_summarizes_scores():
    report = QuerySimilarityReport.from_query_scores([[0.9, 0.8, 0.1], [0.7, 0.6], []])

    assert report.total_queries == 3
    assert report.total_scores == 5
    assert len(report.per_query_score_summary) == 3
    assert report.per_query_score_summary[2]["count"] == 0
    assert report.score_summary["count"] == 5
    assert report.top1_summary["count"] == 2
    assert report.top1_top2_margin_summary["count"] == 2
    assert 0 <= report.high_similarity_rate <= 1


def test_index_health_inspector_uses_collection_info_and_error_ledger(tmp_path: Path):
    ledger_path = tmp_path / "index_events.jsonl"
    ledger = IndexEventLedger(path=ledger_path)
    ledger.record_event("qdrant_ingestion", success=True, stage="sync")
    ledger.record_event("qdrant_ingestion", success=False, stage="sync", message="boom", error_type="RuntimeError")

    inspector = IndexHealthInspector(collection_name="embedding_collection", ledger=ledger)

    fake_collection = SimpleNamespace(
        points_count=100,
        vectors_count=100,
        indexed_vectors_count=90,
        segments_count=4,
        status="green",
        optimizer_status="ok",
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1536))),
    )
    fake_client = SimpleNamespace(get_collection=lambda collection_name: fake_collection)

    report = inspector.inspect(fake_client)

    assert report.collection_name == "embedding_collection"
    assert report.point_count == 100
    assert report.vector_size == 1536
    assert report.estimated_vector_mb is not None
    assert round(report.fragmentation_ratio, 3) == 0.1
    assert report.error_event_count == 1
    assert report.error_rate == 0.5
