from __future__ import annotations

from eval.cost_and_usage import CostUsageRecorder
from eval.cost_and_usage.metrics import ModelPricing


class FakeModel:
    def invoke(self, payload):
        return {
            "text": "grounded answer",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }


def test_usage_proxy_records_tokens_cost_and_tenant_breakdown():
    recorder = CostUsageRecorder(
        pricing={"FakeModel": ModelPricing(input_cost_per_1k=1.0, output_cost_per_1k=2.0)}
    )
    proxy = recorder.wrap_model(FakeModel(), model_name="FakeModel", tenant_id="tenant-a")

    response = proxy.invoke("What happened?")
    assert response["text"] == "grounded answer"

    report = recorder.summarize()

    assert report.total_input_tokens == 10
    assert report.total_output_tokens == 4
    assert round(report.total_cost_usd, 3) == 0.018
    assert report.by_model["FakeModel"].request_count == 1
    assert report.by_tenant["tenant-a"].total_tokens == 14
    assert round(float(report.per_request_over_time[0]["cost_usd"]), 3) == 0.018


def test_usage_proxy_prefers_predict_when_predict_is_available():
    calls = {"predict": 0, "invoke": 0}

    class PredictModel:
        def predict(self, payload):
            calls["predict"] += 1
            return {"text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 2}}

        def invoke(self, payload):
            calls["invoke"] += 1
            return {"text": "wrong"}

    recorder = CostUsageRecorder()
    proxy = recorder.wrap_model(PredictModel(), model_name="PredictModel")
    proxy.predict("hello")

    assert calls == {"predict": 1, "invoke": 0}


def test_usage_recorder_falls_back_to_estimated_tokens():
    recorder = CostUsageRecorder(default_pricing=ModelPricing(input_cost_per_1k=0.5, output_cost_per_1k=1.0))

    class NoUsageModel:
        def invoke(self, payload):
            return "short answer"

    proxy = recorder.wrap_model(NoUsageModel(), model_name="NoUsageModel")
    proxy.invoke("A much longer prompt that should be estimated.")

    report = recorder.summarize()

    assert report.by_model["NoUsageModel"].request_count == 1
    assert report.total_tokens > 0
    assert report.total_cost_usd > 0
