"""Terminal chatbot CLI with chat and report commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from openai import OpenAI

from chatbot import RAGChatbotGraph
from chatbot.utils import extract_chunks
from eval import LatencyRecorder, RAGEvaluationSuite
from eval.utils import ensure_serializable, estimate_tokens

from .eval_reports import (
    BuiltEvalReports,
    DEFAULT_LOG_PATH,
    GOLDEN_DATASET_DIR,
    REPORT_CHOICES,
    build_embedding_drift_report,
    build_index_health_report,
    build_latency_and_cost_recorders,
    build_query_similarity_report,
    build_retrieval_and_answer_quality_report,
    load_trace_records,
    load_golden_queries,
    parse_selected_reports,
)


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class OpenAITextModel:
    """Small adapter that exposes OpenAI Responses API calls as a text model."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    @staticmethod
    def _render_payload(payload: Any) -> str:
        """Convert either plain text or chat messages into prompt text."""

        if isinstance(payload, (list, tuple)):
            from chatbot.utils import messages_to_prompt_text

            return messages_to_prompt_text(list(payload))
        return str(payload)

    @staticmethod
    def _response_text(response: Any) -> str:
        """Extract text from an OpenAI Responses payload."""

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = getattr(response, "output", None)
        if isinstance(output, list):
            rendered: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for block in content:
                        text = getattr(block, "text", None)
                        if isinstance(text, str) and text.strip():
                            rendered.append(text.strip())
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text.strip():
                        rendered.append(text.strip())
            if rendered:
                return "\n".join(rendered).strip()

        return str(response).strip()

    def invoke(self, payload: Any, *args: Any, **kwargs: Any) -> str:
        """Run the model through the OpenAI Responses API."""

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self._render_payload(payload),
        }
        if self.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        response = self._client.responses.create(
            **request_kwargs,
        )
        return self._response_text(response)

    def predict(self, payload: Any, *args: Any, **kwargs: Any) -> str:
        """Support the prompt helpers' predict-style fallback."""

        return self.invoke(payload, *args, **kwargs)


def _default_openai_model() -> str:
    """Read the configured default OpenAI model lazily from the environment."""

    return os.getenv("CHATBOT_MODEL", DEFAULT_OPENAI_MODEL)


def _get_openai_api_key() -> str:
    """Read the OpenAI API key from the common chatbot env var names."""

    for name in ("CHATBOT_APIKEY", "CHATBOT_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _get_openai_client() -> OpenAI:
    """Build an OpenAI client from the configured terminal environment key."""

    api_key = _get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "No OpenAI API key was found. Set CHATBOT_APIKEY, CHATBOT_API_KEY, or OPENAI_API_KEY."
        )
    return OpenAI(api_key=api_key)


def create_chatbot_graph() -> RAGChatbotGraph:
    """Create the production chatbot graph using OpenAI-backed text models."""

    client = _get_openai_client()
    model_name = _default_openai_model()
    latency_recorder = LatencyRecorder()
    answer_model = OpenAITextModel(client, model=model_name, max_output_tokens=120, temperature=0.1)
    critic_model = OpenAITextModel(client, model=model_name, max_output_tokens=96, temperature=0.0)
    rewrite_model = OpenAITextModel(client, model=model_name, max_output_tokens=48, temperature=0.0)
    decomposition_model = OpenAITextModel(client, model=model_name, max_output_tokens=96, temperature=0.0)

    return RAGChatbotGraph(
        answer_model=answer_model,
        critic_model=critic_model,
        rewrite_model=rewrite_model,
        decomposition_model=decomposition_model,
        max_subquestions=3,
        max_context_retries=2,
        max_retrieval_rounds=1,
        latency_recorder=latency_recorder,
        status_callback=print,
    )


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    """Convert a retrieved chunk-like object into a compact log record."""

    node = getattr(chunk, "node", None)
    metadata = getattr(chunk, "metadata", None) or getattr(node, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}

    text = (
        getattr(chunk, "text", None)
        or getattr(chunk, "content", None)
        or getattr(node, "text", None)
        or getattr(node, "content", None)
        or ""
    )
    score = getattr(chunk, "score", None)
    if score is None and node is not None:
        score = getattr(node, "score", None)

    return {
        "doc_id": metadata.get("doc_id") or metadata.get("paper_id") or metadata.get("source"),
        "chunk_id": metadata.get("chunk_id") or metadata.get("chunk_index"),
        "section_id": metadata.get("section_id"),
        "paper_id": metadata.get("paper_id"),
        "score": score,
        "text": str(text).strip(),
        "metadata": metadata,
    }


def _subquestion_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a sub-question trace into JSON-safe data."""

    retrieved_chunks = [_chunk_to_dict(chunk) for chunk in extract_chunks(getattr(result, "retrieval_response", None))]
    deterministic_result = getattr(result, "deterministic_result", None)

    return ensure_serializable(
        {
            "subquestion": getattr(result, "subquestion", ""),
            "retrieval_query": getattr(result, "retrieval_query", ""),
            "context": getattr(result, "context", ""),
            "prompt": getattr(result, "prompt", ""),
            "answer_raw": getattr(result, "answer_raw", None),
            "deterministic_result": deterministic_result,
            "critic_verdict": getattr(result, "critic_verdict", None),
            "accepted": getattr(result, "accepted", False),
            "retried_context": getattr(result, "retried_context", False),
            "retried_retrieval": getattr(result, "retried_retrieval", False),
            "failure_reason": getattr(result, "failure_reason", None),
            "retrieval_feedback": getattr(result, "retrieval_feedback", None),
            "notes": list(getattr(result, "notes", []) or []),
            "retrieved_chunks": retrieved_chunks,
        }
    )


def _collect_retrieved_chunks(result: Any) -> list[dict[str, Any]]:
    """Gather all retrieved chunks from a chatbot result."""

    chunks: list[dict[str, Any]] = []
    for subquestion_result in getattr(result, "subquestion_results", []) or []:
        chunks.extend(_subquestion_result_to_dict(subquestion_result)["retrieved_chunks"])
    return chunks


def _build_run_trace(result: Any, *, duration_s: float) -> dict[str, Any]:
    """Create a JSON-safe trace record from a chatbot run result."""

    prompt_text = "\n".join(
        part
        for part in [
            getattr(result, "original_query", ""),
            getattr(getattr(result, "normalized_query", None), "normalized", ""),
            getattr(result, "rewritten_query", ""),
            "\n".join(getattr(result, "subquestions", []) or []),
            "\n\n".join(getattr(subquestion, "prompt", "") for subquestion in getattr(result, "subquestion_results", []) or []),
        ]
        if part
    )
    answer_text = getattr(result, "final_answer", "")
    normalized_query = getattr(result, "normalized_query", None)
    trace = {
        "schema_version": 1,
        "created_at": time.time(),
        "user_query": getattr(result, "original_query", ""),
        "normalized_query": None if normalized_query is None else ensure_serializable(normalized_query),
        "rewritten_query": getattr(result, "rewritten_query", ""),
        "subquestions": list(getattr(result, "subquestions", []) or []),
        "subquestion_results": [_subquestion_result_to_dict(subquestion) for subquestion in getattr(result, "subquestion_results", []) or []],
        "final_answer": answer_text,
        "accepted": bool(getattr(result, "accepted", False)),
        "retry_count": int(getattr(result, "retry_count", 0)),
        "feedback": getattr(result, "feedback", None),
        "duration_s": float(duration_s),
        "prompt_tokens_estimate": estimate_tokens(prompt_text),
        "answer_tokens_estimate": estimate_tokens(answer_text),
        "retrieved_chunks": _collect_retrieved_chunks(result),
        "component_timings": list(getattr(result, "component_timings", []) or []),
    }
    return ensure_serializable(trace)


def _append_run_log(log_path: Path, trace: dict[str, Any]) -> None:
    """Append one run trace to the JSONL log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _build_eval_bundle(args: Any) -> BuiltEvalReports:
    """Build the selected eval reports from CLI inputs."""

    selected = parse_selected_reports(getattr(args, "report", None))
    selected_set = set(selected)

    log_path = Path(getattr(args, "log_path", DEFAULT_LOG_PATH))
    trace_records: list[dict[str, Any]] = []
    if {"latency", "cost", "query-similarity", "retrieval-answer-quality"} & selected_set:
        try:
            trace_records = load_trace_records(log_path)
        except FileNotFoundError:
            trace_records = []

    latency_recorder = None
    cost_recorder = None
    if {"latency", "cost"} & selected_set:
        latency_recorder, cost_recorder = build_latency_and_cost_recorders(trace_records)

    embedding_drift_report = None
    if "embedding-drift" in selected_set:
        embedding_drift_report = build_embedding_drift_report()

    query_similarity_report = None
    if "query-similarity" in selected_set:
        query_similarity_report = build_query_similarity_report(trace_records)

    index_health_report = None
    if "index-health" in selected_set:
        index_health_report = build_index_health_report()

    retrieval_quality_report = None
    if "retrieval-answer-quality" in selected_set:
        if not getattr(args, "evaluator_llm", None):
            raise ValueError("retrieval-answer-quality requires an evaluator LLM.")
        retrieval_quality_report = build_retrieval_and_answer_quality_report(
            evaluator_llm=args.evaluator_llm,
            trace_records=trace_records,
        )

    suite = RAGEvaluationSuite(
        latency_recorder=latency_recorder,
        cost_recorder=cost_recorder,
        embedding_drift_report=embedding_drift_report,
        query_similarity_report=query_similarity_report,
        index_health_report=index_health_report,
        retrieval_and_answer_quality_report=retrieval_quality_report,
    )
    evaluation_report = suite.build_report()
    return BuiltEvalReports(
        report=evaluation_report,
        latency_report=evaluation_report.latency_and_reliability,
        cost_report=evaluation_report.cost_and_usage,
        embedding_drift_report=embedding_drift_report,
        query_similarity_report=query_similarity_report,
        index_health_report=index_health_report,
        retrieval_and_answer_quality_report=retrieval_quality_report,
    )


def _run_golden_queries(graph: RAGChatbotGraph, *, dataset_dir: str | Path, log_path: Path) -> int:
    """Run the golden queries through the chatbot and persist traces for eval."""

    query_records = load_golden_queries(dataset_dir)
    if not query_records:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    executed = 0
    with log_path.open("a", encoding="utf-8") as handle:
        for record in query_records:
            prompt = str(record.get("question") or record.get("query") or record.get("prompt") or record.get("text") or "").strip()
            if not prompt:
                continue
            started = time.perf_counter()
            result = graph.run(prompt)
            trace = _build_run_trace(result, duration_s=time.perf_counter() - started)
            trace["query_id"] = record.get("query_id") or record.get("id")
            trace["golden_query"] = prompt
            handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
            executed += 1
    return executed


def _format_eval_bundle_text(bundle: BuiltEvalReports, selected: Sequence[str]) -> str:
    """Render the selected eval reports as readable terminal text."""
    
    selected_set = set(selected)
    sections: list[str] = []

    if "latency" in selected_set and bundle.latency_report is not None:
        latency = bundle.latency_report.to_dict()
        reliability = latency["reliability"]
        component_breakdown = latency.get("component_breakdown", [])
        sections.append(
            "\n".join(
                [
                    "Latency and Reliability",
                    f"Requests evaluated: {reliability['total_requests']}",
                    f"Successful requests: {reliability['successful_requests']}",
                    f"Failed requests: {reliability['failed_requests']}",
                    f"Accepted requests: {reliability['accepted_requests']}",
                    f"Timed out requests: {reliability['timeout_requests']}",
                    f"Success rate: {reliability['success_rate']:.2%}",
                    f"Accepted rate: {reliability['accepted_rate']:.2%}",
                    f"Failure rate: {reliability['failure_rate']:.2%}",
                    f"Timeout rate: {reliability['timeout_rate']:.2%}",
                    f"End-to-end latency mean per request: {latency['end_to_end']['mean']:.3f}s",
                    f"End-to-end latency p50 per request: {latency['end_to_end']['p50']:.3f}s",
                    f"End-to-end latency p95 per request: {latency['end_to_end']['p95']:.3f}s",
                    f"End-to-end latency p99 per request: {latency['end_to_end']['p99']:.3f}s",
                    f"End-to-end latency min per request: {latency['end_to_end']['min']:.3f}s",
                    f"End-to-end latency max per request: {latency['end_to_end']['max']:.3f}s",
                    "",
                    "Component Breakdown",
                ]
                + [
                    f"- {item['label']}: requests={item['summary'].get('count', 0)}, mean per request={item['summary'].get('mean', 0.0):.3f}s, p50={item['summary'].get('p50', 0.0):.3f}s, p95={item['summary'].get('p95', 0.0):.3f}s, p99={item['summary'].get('p99', 0.0):.3f}s, min={item['summary'].get('min', 0.0):.3f}s, max={item['summary'].get('max', 0.0):.3f}s"
                    for item in component_breakdown
                ]
            )
        )

    if "cost" in selected_set and bundle.cost_report is not None:
        cost = bundle.cost_report.to_dict()
        by_model = cost.get("by_model", {})
        sections.append(
            "\n".join(
                [
                    "Cost and Usage",
                    f"Input tokens counted: {cost['total_input_tokens']}",
                    f"Output tokens counted: {cost['total_output_tokens']}",
                    f"Total tokens counted: {cost['total_tokens']}",
                    f"Estimated cost from token pricing: ${cost['total_cost_usd']:.4f}",
                    f"Mean input tokens per request: {cost['total_input_tokens'] / max(1, len(bundle.cost_report.samples)):.2f}" if bundle.cost_report.samples else "Mean input tokens per request: 0.00",
                    f"Mean output tokens per request: {cost['total_output_tokens'] / max(1, len(bundle.cost_report.samples)):.2f}" if bundle.cost_report.samples else "Mean output tokens per request: 0.00",
                    f"Mean cost USD per request: ${cost['total_cost_usd'] / max(1, len(bundle.cost_report.samples)):.4f}" if bundle.cost_report.samples else "Mean cost USD per request: $0.0000",
                    "",
                    "By Model",
                ]
                + [
                    f"- {name}: requests={item['request_count']}, input={item['input_tokens']}, output={item['output_tokens']}, total={item['total_tokens']}, cost=${item['cost_usd']:.4f}"
                    for name, item in by_model.items()
                ]
            )
        )

    if "embedding-drift" in selected_set and bundle.embedding_drift_report is not None:
        drift = bundle.embedding_drift_report.to_dict()
        sections.append(
            "\n".join(
                [
                    "Embedding Drift",
                    f"Current embedding count: {drift['current']['count']}",
                    f"Baseline embedding count: {drift['baseline']['count'] if drift['baseline'] else 0}",
                    f"Embedding count delta: {drift['count_delta']}",
                    f"Mean embedding norm delta: {drift['mean_norm_delta']:.4f}",
                    f"Std embedding norm delta: {drift['std_norm_delta']:.4f}",
                    f"Median embedding norm delta: {drift['median_norm_delta']:.4f}",
                    f"p95 embedding norm delta: {drift['p95_norm_delta']:.4f}",
                    f"Min embedding norm delta: {drift['min_norm_delta']:.4f}",
                    f"Max embedding norm delta: {drift['max_norm_delta']:.4f}",
                    f"Centroid cosine similarity: {drift['centroid_cosine_similarity'] if drift['centroid_cosine_similarity'] is not None else 'n/a'}",
                    f"Centroid cosine distance: {drift['centroid_cosine_distance'] if drift['centroid_cosine_distance'] is not None else 'n/a'}",
                    f"Drift score: {drift['drift_score']:.4f}",
                ]
            )
        )

    if "query-similarity" in selected_set and bundle.query_similarity_report is not None:
        similarity = bundle.query_similarity_report.to_dict()
        per_query = similarity.get("per_query_score_summary", [])
        sections.append(
            "\n".join(
                [
                    "Query Similarity",
                    f"Queries evaluated: {similarity['total_queries']}",
                    f"Retrieved chunk scores counted: {similarity['total_scores']}",
                    f"All retrieved chunk scores mean: {similarity['score_summary']['mean']:.4f}",
                    f"All retrieved chunk scores std: {similarity['score_summary']['std']:.4f}",
                    f"All retrieved chunk scores p50: {similarity['score_summary']['p50']:.4f}",
                    f"All retrieved chunk scores p95: {similarity['score_summary']['p95']:.4f}",
                    f"All retrieved chunk scores p99: {similarity['score_summary']['p99']:.4f}",
                    f"All retrieved chunk scores min: {similarity['score_summary']['min']:.4f}",
                    f"All retrieved chunk scores max: {similarity['score_summary']['max']:.4f}",
                    f"Best retrieval score per query mean: {similarity['top1_summary']['mean']:.4f}",
                    f"Best retrieval score per query p50: {similarity['top1_summary']['p50']:.4f}",
                    f"Best retrieval score per query p95: {similarity['top1_summary']['p95']:.4f}",
                    f"Best retrieval score per query min: {similarity['top1_summary']['min']:.4f}",
                    f"Best retrieval score per query max: {similarity['top1_summary']['max']:.4f}",
                    f"Top-1 minus top-2 margin mean: {similarity['top1_top2_margin_summary']['mean']:.4f}",
                    f"Top-1 minus top-2 margin p50: {similarity['top1_top2_margin_summary']['p50']:.4f}",
                    f"Top-1 minus top-2 margin p95: {similarity['top1_top2_margin_summary']['p95']:.4f}",
                    f"Top-1 minus top-2 margin min: {similarity['top1_top2_margin_summary']['min']:.4f}",
                    f"Top-1 minus top-2 margin max: {similarity['top1_top2_margin_summary']['max']:.4f}",
                    f"High similarity rate (best score >= 0.75): {similarity['high_similarity_rate']:.2%}",
                    "",
                    "Per Query",
                ]
                + [
                    f"- query {idx + 1}: score count={item['count']}, score mean={item['mean']:.4f}, score p50={item['p50']:.4f}, score p95={item['p95']:.4f}, score min={item['min']:.4f}, score max={item['max']:.4f}"
                    for idx, item in enumerate(per_query)
                ]
            )
        )

    if "index-health" in selected_set and bundle.index_health_report is not None:
        index_health = bundle.index_health_report.to_dict()
        notes = index_health.get("notes", [])
        error_events = index_health.get("error_events", [])
        index_lines = [
                    "Index Health",
                    f"Collection: {index_health['collection_name']}",
                    f"Points stored: {index_health['point_count']}",
                    f"Vectors stored: {index_health['vector_count']}",
                    f"Indexed vectors: {index_health['indexed_vector_count']}",
                    f"Segments: {index_health['segments_count']}",
                    f"Vector size: {index_health['vector_size']}",
                    f"Estimated vector storage MB: {index_health['estimated_vector_mb']}",
                    f"Status: {index_health['status']}",
                    f"Optimizer status: {index_health['optimizer_status']}",
                    f"Fragmentation ratio: {index_health['fragmentation_ratio']:.2%}",
                    f"Error events: {index_health['error_event_count']}",
                    f"Error rate across qdrant_ingestion events: {index_health['error_rate']:.2%}",
                ]
        if notes:
            index_lines.extend(["", "Notes"])
            index_lines.extend(f"- {note}" for note in notes)
        if error_events:
            index_lines.extend(["", "Error Events"])
            index_lines.extend(
                f"- {event.get('timestamp', '')} | {event.get('stage', '')} | {event.get('message', '')}"
                for event in error_events
            )
        sections.append("\n".join(index_lines))

    if "retrieval-answer-quality" in selected_set and bundle.retrieval_and_answer_quality_report is not None:
        quality = bundle.retrieval_and_answer_quality_report.to_dict()
        def _metric_lines(name: str, metric: dict[str, Any] | None) -> list[str]:
            if not metric:
                return [f"{name}: n/a"]
            summary = metric["summary"]
            return [
                f"{name}",
                f"  total samples: {metric['total_samples']}",
                f"  evaluated samples: {metric['evaluated_samples']}",
                f"  mean: {summary['mean']:.4f}",
                f"  p95: {summary['p95']:.4f}",
                f"  min: {summary['min']:.4f}",
                f"  max: {summary['max']:.4f}",
            ]

        sections.append(
            "\n".join(
                [
                    "Retrieval and Answer Quality",
                    f"Hallucination rate (1 - faithfulness mean): {quality['hallucination_rate']:.2%}" if quality["hallucination_rate"] is not None else "Hallucination rate (1 - faithfulness mean): n/a",
                ]
                + ["", *_metric_lines("Context precision with reference", quality["context_precision_with_reference"])]
                + ["", *_metric_lines("ID-based context precision", quality["id_based_context_precision"])]
                + ["", *_metric_lines("Context recall", quality["context_recall"])]
                + ["", *_metric_lines("ID-based context recall", quality["id_based_context_recall"])]
                + ["", *_metric_lines("Faithfulness", quality["faithfulness"])]
                + ["", *_metric_lines("Response groundedness", quality["response_groundedness"])]
            )
        )

    return "\n\n".join(sections) if sections else "No reports were built."


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the eval-report arguments to a subcommand parser."""

    parser.add_argument(
        "--report",
        action="append",
        choices=REPORT_CHOICES + ("all",),
        help="Select one or more reports. Defaults to the trace-backed reports.",
    )
    parser.add_argument(
        "--run-golden-queries",
        action="store_true",
        help="Run eval/golden_dataset queries through the chatbot before building retrieval-quality reports.",
    )
    parser.add_argument(
        "--trace-log",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to the chatbot trace log used by eval.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report bundle as JSON.")


def _chat_loop(graph: RAGChatbotGraph) -> None:
    """Run an interactive terminal chat loop."""

    print("RAG Chatbot: ready.")
    while True:
        try:
            prompt = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        started = time.perf_counter()
        result = graph.run(prompt)
        trace = _build_run_trace(result, duration_s=time.perf_counter() - started)
        _append_run_log(DEFAULT_LOG_PATH, trace)
        answer = trace["final_answer"].strip()
        print(answer if answer else "No grounded answer was produced.")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(prog="chatbot-cli", description="Terminal chatbot with persisted eval traces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("chat", help="Start an interactive terminal chat.")

    eval_parser = subparsers.add_parser("eval", help="Build one or more eval reports from local data.")
    _add_eval_arguments(eval_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "eval":
        selected = parse_selected_reports(args.report)
        if args.run_golden_queries:
            graph = create_chatbot_graph()
            executed = _run_golden_queries(graph, dataset_dir=GOLDEN_DATASET_DIR, log_path=Path(args.trace_log))
            if executed == 0:
                print(
                    f"eval: no golden queries were found under {GOLDEN_DATASET_DIR}.",
                    file=sys.stderr,
                )
                return 1
        if "retrieval-answer-quality" in selected:
            api_key = _get_openai_api_key()
            if not api_key:
                print(
                    "eval: no OpenAI API key was found for retrieval-answer-quality. "
                    "Set CHATBOT_APIKEY, CHATBOT_API_KEY, or OPENAI_API_KEY.",
                    file=sys.stderr,
                )
                return 1
            evaluator_model = _default_openai_model()
            client = OpenAI(api_key=api_key)
            args.evaluator_llm = OpenAITextModel(client, model=evaluator_model)
        else:
            args.evaluator_llm = None

        try:
            if args.run_golden_queries:
                from cli import eval_reports as cli_eval_reports

                cli_eval_reports.DEFAULT_LOG_PATH = Path(args.trace_log)
            bundle = _build_eval_bundle(args)
        except Exception as exc:
            print(f"eval: {exc}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(_format_eval_bundle_text(bundle, selected))
        return 0

    if args.command == "chat":
        graph = create_chatbot_graph()

        _chat_loop(graph)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":  
    raise SystemExit(main())
