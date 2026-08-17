"""Latency and reliability evaluation primitives."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from ..utils import summarize_numbers, utc_now


COMPONENT_ALIASES = {
    "vector": "vector_search",
    "vector_search": "vector_search",
    "bm25": "bm25",
    "sparse_search": "bm25",
    "reranker": "reranker",
    "rerank": "reranker",
    "llm": "llm_inference",
    "llm_inference": "llm_inference",
    "answer_model": "llm_inference",
    "critic_model": "llm_inference",
}

COMPONENT_LABELS = {
    "vector_search": "Vector search",
    "bm25": "BM25",
    "reranker": "Reranker",
    "llm_inference": "LLM inference",
}


def _canonical_component_name(name: str) -> str:
    """Normalize a stage name into a canonical component label."""

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return COMPONENT_ALIASES.get(normalized, normalized)


@dataclass
class LatencySample:
    """End-to-end latency sample for one chatbot request."""

    duration_s: float
    success: bool = True
    accepted: bool | None = None
    timeout: bool = False
    request_id: str | None = None
    tenant_id: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyComponentSample:
    """Latency sample for a specific pipeline component."""

    component: str
    duration_s: float
    request_id: str | None = None
    tenant_id: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyComponentSummary:
    """Percentile summary for a component."""

    component: str
    label: str
    summary: dict[str, float | int]


@dataclass
class ReliabilitySummary:
    """Success and failure summary for chatbot runs."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    accepted_requests: int
    timeout_requests: int
    success_rate: float
    accepted_rate: float
    failure_rate: float
    timeout_rate: float
    by_tenant: dict[str, dict[str, float | int]] = field(default_factory=dict)


@dataclass
class LatencyReport:
    """Aggregated latency and reliability report."""

    end_to_end: dict[str, float | int]
    component_breakdown: list[LatencyComponentSummary]
    reliability: ReliabilitySummary
    samples: list[LatencySample] = field(default_factory=list)
    component_samples: list[LatencyComponentSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the report as plain data."""

        return {
            "end_to_end": self.end_to_end,
            "component_breakdown": [
                {
                    "component": item.component,
                    "label": item.label,
                    "summary": item.summary,
                }
                for item in self.component_breakdown
            ],
            "reliability": {
                "total_requests": self.reliability.total_requests,
                "successful_requests": self.reliability.successful_requests,
                "failed_requests": self.reliability.failed_requests,
                "accepted_requests": self.reliability.accepted_requests,
                "timeout_requests": self.reliability.timeout_requests,
                "success_rate": self.reliability.success_rate,
                "accepted_rate": self.reliability.accepted_rate,
                "failure_rate": self.reliability.failure_rate,
                "timeout_rate": self.reliability.timeout_rate,
                "by_tenant": self.reliability.by_tenant,
            },
        }


class LatencyRecorder:
    """Collect end-to-end and component latency samples for the RAG chatbot."""

    def __init__(self) -> None:
        self.samples: list[LatencySample] = []
        self.component_samples: list[LatencyComponentSample] = []

    def record_request(
        self,
        duration_s: float,
        *,
        success: bool = True,
        accepted: bool | None = None,
        timeout: bool = False,
        request_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LatencySample:
        """Store one end-to-end latency observation."""

        sample = LatencySample(
            duration_s=float(duration_s),
            success=bool(success),
            accepted=accepted,
            timeout=bool(timeout),
            request_id=request_id,
            tenant_id=tenant_id,
            metadata=dict(metadata or {}),
        )
        self.samples.append(sample)
        return sample

    def record_component(
        self,
        component: str,
        duration_s: float,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LatencyComponentSample:
        """Store a component-level latency observation."""

        sample = LatencyComponentSample(
            component=_canonical_component_name(component),
            duration_s=float(duration_s),
            request_id=request_id,
            tenant_id=tenant_id,
            metadata=dict(metadata or {}),
        )
        self.component_samples.append(sample)
        return sample

    def wrap_component(
        self,
        component: str,
        target: Any,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "LatencyTrackingProxy":
        """Return a proxy that times calls to the wrapped component."""

        return LatencyTrackingProxy(
            target=target,
            recorder=self,
            component=component,
            request_id=request_id,
            tenant_id=tenant_id,
            metadata=metadata,
        )

    @contextmanager
    def track_request(
        self,
        *,
        request_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Measure a single request block and record it automatically."""

        started = time.perf_counter()
        error: Exception | None = None
        try:
            yield
        except Exception as exc:
            error = exc
            raise
        finally:
            elapsed = time.perf_counter() - started
            request_metadata = dict(metadata or {})
            if error is not None:
                request_metadata["error"] = str(error)
            self.record_request(
                elapsed,
                success=error is None,
                request_id=request_id,
                tenant_id=tenant_id,
                metadata=request_metadata,
            )

    def summarize(self) -> LatencyReport:
        """Build the aggregated latency and reliability report."""

        end_to_end = summarize_numbers([sample.duration_s for sample in self.samples])

        component_groups: dict[str, list[float]] = defaultdict(list)
        for sample in self.component_samples:
            component_groups[sample.component].append(sample.duration_s)

        component_breakdown: list[LatencyComponentSummary] = []
        for component in sorted(component_groups):
            component_breakdown.append(
                LatencyComponentSummary(
                    component=component,
                    label=COMPONENT_LABELS.get(component, component.replace("_", " ").title()),
                    summary=summarize_numbers(component_groups[component]),
                )
            )

        by_tenant: dict[str, dict[str, float | int]] = {}
        tenant_groups: dict[str, list[LatencySample]] = defaultdict(list)
        for sample in self.samples:
            if sample.tenant_id:
                tenant_groups[sample.tenant_id].append(sample)

        for tenant_id, tenant_samples in tenant_groups.items():
            durations = [sample.duration_s for sample in tenant_samples]
            accepted_count = sum(1 for sample in tenant_samples if sample.accepted is True)
            by_tenant[tenant_id] = {
                "count": len(tenant_samples),
                "success_rate": sum(1 for sample in tenant_samples if sample.success) / len(tenant_samples),
                "accepted_rate": accepted_count / len(tenant_samples),
                "mean_latency_s": sum(durations) / len(durations),
                "p95_latency_s": summarize_numbers(durations)["p95"],
            }

        total_requests = len(self.samples)
        successful_requests = sum(1 for sample in self.samples if sample.success)
        failed_requests = total_requests - successful_requests
        accepted_requests = sum(1 for sample in self.samples if sample.accepted is True)
        timeout_requests = sum(1 for sample in self.samples if sample.timeout)
        reliability = ReliabilitySummary(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            accepted_requests=accepted_requests,
            timeout_requests=timeout_requests,
            success_rate=(successful_requests / total_requests) if total_requests else 0.0,
            accepted_rate=(accepted_requests / total_requests) if total_requests else 0.0,
            failure_rate=(failed_requests / total_requests) if total_requests else 0.0,
            timeout_rate=(timeout_requests / total_requests) if total_requests else 0.0,
            by_tenant=by_tenant,
        )

        return LatencyReport(
            end_to_end=end_to_end,
            component_breakdown=component_breakdown,
            reliability=reliability,
            samples=list(self.samples),
            component_samples=list(self.component_samples),
        )


class LatencyTrackingProxy:
    """Proxy any callable/invokable object and record component latency."""

    def __init__(
        self,
        *,
        target: Any,
        recorder: LatencyRecorder,
        component: str,
        request_id: str | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._target = target
        self._recorder = recorder
        self.component = _canonical_component_name(component)
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.metadata = dict(metadata or {})

    def _invoke_underlying(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped object using the interface it exposes."""

        if hasattr(self._target, "invoke"):
            return self._target.invoke(payload, *args, **kwargs)
        if hasattr(self._target, "predict"):
            return self._target.predict(payload, *args, **kwargs)
        if callable(self._target):
            return self._target(payload, *args, **kwargs)
        raise TypeError("Wrapped component must be callable or expose invoke/predict.")

    def _predict_underlying(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped object preferring its predict interface."""

        if hasattr(self._target, "predict"):
            return self._target.predict(payload, *args, **kwargs)
        if hasattr(self._target, "invoke"):
            return self._target.invoke(payload, *args, **kwargs)
        if callable(self._target):
            return self._target(payload, *args, **kwargs)
        raise TypeError("Wrapped component must be callable or expose invoke/predict.")

    def _record(self, started_at: float) -> None:
        """Store the elapsed time once the wrapped call completes."""

        self._recorder.record_component(
            self.component,
            time.perf_counter() - started_at,
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            metadata=self.metadata,
        )

    def invoke(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped component's invoke method."""

        started_at = time.perf_counter()
        try:
            return self._invoke_underlying(payload, *args, **kwargs)
        finally:
            self._record(started_at)

    def predict(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the wrapped component's predict method."""

        started_at = time.perf_counter()
        try:
            return self._predict_underlying(payload, *args, **kwargs)
        finally:
            self._record(started_at)

    def __call__(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped component directly."""

        started_at = time.perf_counter()
        try:
            return self._invoke_underlying(payload, *args, **kwargs)
        finally:
            self._record(started_at)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped component."""

        return getattr(self._target, name)
