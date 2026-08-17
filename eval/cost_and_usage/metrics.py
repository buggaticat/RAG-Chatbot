"""Cost and token usage evaluation primitives."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping

from ..utils import (
    bucket_timestamp,
    estimate_tokens,
    extract_usage_from_response,
    render_payload_text,
    summarize_numbers,
    utc_now,
)


@dataclass(frozen=True)
class ModelPricing:
    """Simple token pricing model in USD per 1k tokens."""

    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total request cost from token counts."""

        return (input_tokens / 1000.0) * self.input_cost_per_1k + (output_tokens / 1000.0) * self.output_cost_per_1k


@dataclass
class TokenUsageSample:
    """Token usage observed for a single model invocation."""

    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: str | None = None
    tenant_id: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Return the total token count for the sample."""

        return self.input_tokens + self.output_tokens


@dataclass
class ModelUsageSummary:
    """Aggregated token and cost usage for a model or tenant."""

    name: str
    request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_cost_usd: float


@dataclass
class CostUsageReport:
    """Aggregated cost and token usage report."""

    per_request_over_time: list[dict[str, Any]]
    by_model: dict[str, ModelUsageSummary]
    by_tenant: dict[str, ModelUsageSummary]
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    samples: list[TokenUsageSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report payload."""

        return {
            "per_request_over_time": self.per_request_over_time,
            "by_model": {
                key: {
                    "name": value.name,
                    "request_count": value.request_count,
                    "input_tokens": value.input_tokens,
                    "output_tokens": value.output_tokens,
                    "total_tokens": value.total_tokens,
                    "cost_usd": value.cost_usd,
                    "mean_input_tokens": value.mean_input_tokens,
                    "mean_output_tokens": value.mean_output_tokens,
                    "mean_cost_usd": value.mean_cost_usd,
                }
                for key, value in self.by_model.items()
            },
            "by_tenant": {
                key: {
                    "name": value.name,
                    "request_count": value.request_count,
                    "input_tokens": value.input_tokens,
                    "output_tokens": value.output_tokens,
                    "total_tokens": value.total_tokens,
                    "cost_usd": value.cost_usd,
                    "mean_input_tokens": value.mean_input_tokens,
                    "mean_output_tokens": value.mean_output_tokens,
                    "mean_cost_usd": value.mean_cost_usd,
                }
                for key, value in self.by_tenant.items()
            },
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }


class CostUsageRecorder:
    """Collect token usage and cost estimates for chatbot requests."""

    def __init__(
        self,
        *,
        pricing: Mapping[str, ModelPricing] | None = None,
        default_pricing: ModelPricing | None = None,
    ) -> None:
        self.pricing = dict(pricing or {})
        self.default_pricing = default_pricing or ModelPricing()
        self.samples: list[TokenUsageSample] = []

    def _pricing_for(self, model_name: str) -> ModelPricing:
        """Resolve the token pricing for a specific model."""

        return self.pricing.get(model_name, self.default_pricing)

    def record(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TokenUsageSample:
        """Store one model usage sample."""

        pricing = self._pricing_for(model_name)
        sample = TokenUsageSample(
            model_name=model_name,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost_usd=pricing.cost(int(input_tokens), int(output_tokens)),
            request_id=request_id,
            tenant_id=tenant_id,
            metadata=dict(metadata or {}),
        )
        self.samples.append(sample)
        return sample

    def record_from_payload(
        self,
        model_name: str,
        payload: Any,
        response: Any,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TokenUsageSample:
        """Record usage from a model payload and response."""

        input_tokens, output_tokens = extract_usage_from_response(response)
        prompt_text = render_payload_text(payload)
        response_text = render_payload_text(response)
        if input_tokens is None:
            input_tokens = estimate_tokens(prompt_text)
        if output_tokens is None:
            output_tokens = estimate_tokens(response_text)
        return self.record(
            model_name,
            input_tokens,
            output_tokens,
            request_id=request_id,
            tenant_id=tenant_id,
            metadata=metadata,
        )

    def wrap_model(
        self,
        model: Any,
        *,
        model_name: str | None = None,
        tenant_id: str | Callable[[], str] | None = None,
        request_id: str | Callable[[], str] | None = None,
    ) -> "UsageTrackingModelProxy":
        """Return a proxy that records every invocation of the model."""

        return UsageTrackingModelProxy(
            model=model,
            recorder=self,
            model_name=model_name or model.__class__.__name__,
            tenant_id=tenant_id,
            request_id=request_id,
        )

    def summarize(self) -> CostUsageReport:
        """Aggregate all tracked usage samples."""

        per_request_over_time = [
            {
                "timestamp": sample.started_at.isoformat(),
                "bucket": bucket_timestamp(sample.started_at),
                "model_name": sample.model_name,
                "tenant_id": sample.tenant_id,
                "request_id": sample.request_id,
                "input_tokens": sample.input_tokens,
                "output_tokens": sample.output_tokens,
                "total_tokens": sample.total_tokens,
                "cost_usd": sample.cost_usd,
            }
            for sample in sorted(self.samples, key=lambda item: item.started_at)
        ]

        by_model_groups: dict[str, list[TokenUsageSample]] = defaultdict(list)
        by_tenant_groups: dict[str, list[TokenUsageSample]] = defaultdict(list)
        for sample in self.samples:
            by_model_groups[sample.model_name].append(sample)
            if sample.tenant_id:
                by_tenant_groups[sample.tenant_id].append(sample)

        def _summarize(name: str, items: list[TokenUsageSample]) -> ModelUsageSummary:
            input_tokens = sum(sample.input_tokens for sample in items)
            output_tokens = sum(sample.output_tokens for sample in items)
            cost_usd = sum(sample.cost_usd for sample in items)
            total_tokens = input_tokens + output_tokens
            count = len(items)
            return ModelUsageSummary(
                name=name,
                request_count=count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                mean_input_tokens=(input_tokens / count) if count else 0.0,
                mean_output_tokens=(output_tokens / count) if count else 0.0,
                mean_cost_usd=(cost_usd / count) if count else 0.0,
            )

        by_model = {name: _summarize(name, items) for name, items in by_model_groups.items()}
        by_tenant = {tenant_id: _summarize(tenant_id, items) for tenant_id, items in by_tenant_groups.items()}

        total_input_tokens = sum(sample.input_tokens for sample in self.samples)
        total_output_tokens = sum(sample.output_tokens for sample in self.samples)
        total_cost_usd = sum(sample.cost_usd for sample in self.samples)

        return CostUsageReport(
            per_request_over_time=per_request_over_time,
            by_model=by_model,
            by_tenant=by_tenant,
            total_cost_usd=total_cost_usd,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            samples=list(self.samples),
        )


class UsageTrackingModelProxy:
    """Proxy a model while recording token usage for each invocation."""

    def __init__(
        self,
        *,
        model: Any,
        recorder: CostUsageRecorder,
        model_name: str,
        tenant_id: str | Callable[[], str] | None = None,
        request_id: str | Callable[[], str] | None = None,
    ) -> None:
        self._model = model
        self._recorder = recorder
        self.model_name = model_name
        self._tenant_id = tenant_id
        self._request_id = request_id

    def _resolve_context_value(self, value: str | Callable[[], str] | None) -> str | None:
        """Return a static or callable context value."""

        if callable(value):
            try:
                return value()
            except Exception:
                return None
        return value

    def _record(self, payload: Any, response: Any, *, started_at: float, metadata: dict[str, Any] | None = None) -> Any:
        """Store a usage sample after a successful call."""

        duration_s = time.perf_counter() - started_at
        sample = self._recorder.record_from_payload(
            self.model_name,
            payload,
            response,
            request_id=self._resolve_context_value(self._request_id),
            tenant_id=self._resolve_context_value(self._tenant_id),
            metadata={"duration_s": duration_s, **(metadata or {})},
        )
        return response

    def _invoke_underlying(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped model using whichever interface it exposes."""

        if hasattr(self._model, "invoke"):
            return self._model.invoke(payload, *args, **kwargs)
        if hasattr(self._model, "predict"):
            return self._model.predict(payload, *args, **kwargs)
        if callable(self._model):
            return self._model(payload, *args, **kwargs)
        raise TypeError("Wrapped model must be callable or expose invoke/predict.")

    def _predict_underlying(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped model preferring its predict interface."""

        if hasattr(self._model, "predict"):
            return self._model.predict(payload, *args, **kwargs)
        if hasattr(self._model, "invoke"):
            return self._model.invoke(payload, *args, **kwargs)
        if callable(self._model):
            return self._model(payload, *args, **kwargs)
        raise TypeError("Wrapped model must be callable or expose invoke/predict.")

    def invoke(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped model and record token usage."""

        started_at = time.perf_counter()
        response = self._invoke_underlying(payload, *args, **kwargs)
        return self._record(payload, response, started_at=started_at)

    def predict(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped model's predict method and record usage."""

        started_at = time.perf_counter()
        response = self._predict_underlying(payload, *args, **kwargs)
        return self._record(payload, response, started_at=started_at)

    def __call__(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped model directly when it is a plain callable."""

        started_at = time.perf_counter()
        response = self._invoke_underlying(payload, *args, **kwargs)
        return self._record(payload, response, started_at=started_at)

    def __getattr__(self, name: str) -> Any:
        """Delegate any missing attribute to the wrapped model."""

        return getattr(self._model, name)
