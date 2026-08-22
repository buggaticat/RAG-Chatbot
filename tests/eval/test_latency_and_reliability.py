from __future__ import annotations

from eval.latency_and_reliability import LatencyRecorder


def test_latency_recorder_summarizes_e2e_and_components():
    recorder = LatencyRecorder()
    recorder.record_request(0.10, success=True, accepted=True, tenant_id="tenant-a")
    recorder.record_request(0.25, success=True, accepted=False, tenant_id="tenant-a")
    recorder.record_request(0.40, success=False, timeout=True, accepted=False, tenant_id="tenant-b")

    recorder.record_component("Vector search", 0.05)
    recorder.record_component("BM25", 0.03)
    recorder.record_component("Reranker", 0.02)
    recorder.record_component("LLM inference", 0.18)

    report = recorder.summarize()

    assert report.end_to_end["count"] == 3
    assert round(float(report.end_to_end["p50"]), 3) == 0.25
    assert round(float(report.end_to_end["p95"]), 3) == 0.385
    assert report.reliability.total_requests == 3
    assert report.reliability.success_rate == 2 / 3
    assert report.reliability.accepted_rate == 1 / 3
    assert report.reliability.timeout_rate == 1 / 3
    assert [item.component for item in report.component_breakdown] == [
        "bm25",
        "llm_inference",
        "reranker",
        "vector_search",
    ]


def test_latency_tracking_proxy_records_component_duration():
    recorder = LatencyRecorder()

    def retriever(query: str) -> str:
        return f"retrieved: {query}"

    proxy = recorder.wrap_component("Vector search", retriever, request_id="req-1", tenant_id="tenant-a")
    assert proxy("hello") == "retrieved: hello"

    report = recorder.summarize()
    assert report.component_breakdown[0].component == "vector_search"
    assert report.component_breakdown[0].summary["count"] == 1


def test_latency_tracking_proxy_prefers_predict_when_predict_is_available():
    calls = {"predict": 0, "invoke": 0}

    class PredictComponent:
        def predict(self, payload):
            calls["predict"] += 1
            return f"predict:{payload}"

        def invoke(self, payload):
            calls["invoke"] += 1
            return f"invoke:{payload}"

    recorder = LatencyRecorder()
    proxy = recorder.wrap_component("Reranker", PredictComponent())

    assert proxy.predict("hello") == "predict:hello"
    assert calls == {"predict": 1, "invoke": 0}
