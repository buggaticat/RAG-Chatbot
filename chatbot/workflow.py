"""LangGraph workflow for the chatbot graph."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from langgraph.graph import END, StateGraph

from rag.context_assembly import INSTRUCTIONS, SYSTEM_PROMPT, build_grounded_prompt, format_context
from rag.retrieval.config import DEFAULT_RERANK_MODEL
from rag.validation_layers import DeterministicValidationResult, validate_deterministic_output, verify_with_critic
from eval import LatencyRecorder

from .prompt import NormalizedQuery, decompose_rewritten_query, normalize_query_language, rewrite_user_query
from .state import ChatbotRunResult, ChatbotState, SubQuestionResult, SubQuestionState
from .utils import (
    STRICTER_INSTRUCTIONS,
    copy_notes,
    extract_chunks,
    invoke_text_model,
    normalize_text,
    parse_answer_payload,
)


class RAGChatbotGraph:
    """Coordinate language normalization, retrieval, answering, and verification."""

    def __init__(
        self,
        *,
        answer_model: Any,
        critic_model: Any,
        rewrite_model: Any | None = None,
        decomposition_model: Any | None = None,
        retriever: Any | None = None,
        translator: Any | None = None,
        top_k: int = 5,
        rerank_top_n: int | None = None,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        context_max_tokens: int | None = None,
        max_subquestions: int = 3,
        max_context_retries: int = 2,
        max_retrieval_rounds: int = 1,
        system_prompt: str = SYSTEM_PROMPT,
        status_callback: Any | None = None,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        self.answer_model = answer_model
        self.critic_model = critic_model
        self.rewrite_model = rewrite_model
        self.decomposition_model = decomposition_model
        self.retriever = retriever or self._default_retriever
        self.translator = translator
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.rerank_model = rerank_model
        self.context_max_tokens = context_max_tokens
        self.max_subquestions = max(1, max_subquestions)
        self.max_context_retries = max_context_retries
        self.max_retrieval_rounds = max_retrieval_rounds
        self.system_prompt = system_prompt
        self.status_callback = status_callback
        self.latency_recorder = latency_recorder
        self._component_timings: list[dict[str, Any]] = []

        self.subquestion_graph = self._build_subquestion_graph()
        self.graph = self._build_chatbot_graph()

    def _normalize_query(self, user_query: str) -> NormalizedQuery:
        """Translate queries into the working language before retrieval."""

        if self.translator is None:
            try:
                from rag.translation.translate import translate_user_query as translator  # type: ignore[no-redef]
            except Exception:
                translator = lambda value: value  # type: ignore[assignment]
        else:
            translator = self.translator
        return normalize_query_language(user_query, translator=translator)

    def _build_context(self, retrieved: Any) -> str:
        """Format retrieved chunks into a prompt-ready context block."""

        return format_context(retrieved, max_tokens=self.context_max_tokens)

    def _emit_status(self, message: str) -> None:
        """Send a lightweight progress message to the configured terminal hook."""

        callback = self.status_callback
        if callback is None:
            return
        try:
            callback(message)
        except Exception:
            return

    def _record_component_timing(self, component: str, started_at: float) -> None:
        """Record component timing when a latency recorder is configured."""

        duration_s = time.perf_counter() - started_at
        self._component_timings.append({"component": component, "duration_s": duration_s})
        if self.latency_recorder is None:
            return
        try:
            self.latency_recorder.record_component(component, duration_s)
        except Exception:
            return

    def _build_answer_prompt(self, query: str, context: str, *, strict: bool = False) -> str:
        """Build the grounded answer prompt for a specific sub-question."""

        instructions = STRICTER_INSTRUCTIONS if strict else INSTRUCTIONS
        return build_grounded_prompt(
            self.system_prompt,
            instructions,
            query,
            context,
            tokenizer=None,
            max_tokens=self.context_max_tokens or 1024,
        )

    def _retrieve(self, query: str) -> Any:
        """Run hybrid retrieval with the configured top-k budget."""

        return self.retriever(query, top_k=self.top_k)

    @staticmethod
    def _default_retriever(query: str, **kwargs: Any) -> Any:
        """Import the production retriever lazily so tests can inject a fake one."""

        from rag.retrieval import run_hybrid_search

        return run_hybrid_search(query, **kwargs)

    def _ask_answer_model(self, prompt: str) -> str:
        """Generate a grounded answer with the configured answer model."""

        return invoke_text_model(self.answer_model, prompt)

    def _rerank_retrieved(self, query: str, retrieved: Any) -> Any:
        """Run the configured cross-encoder reranker when enabled."""

        if self.rerank_top_n is None or self.rerank_top_n <= 0:
            return retrieved

        chunks = extract_chunks(retrieved)
        if not chunks:
            return retrieved

        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            from llama_index.core.schema import QueryBundle
        except Exception:
            return retrieved

        try:
            reranker = SentenceTransformerRerank(model=self.rerank_model, top_n=self.rerank_top_n)
            reranked_nodes = reranker.postprocess_nodes(chunks, query_bundle=QueryBundle(query_str=query))
        except Exception:
            return retrieved

        try:
            retrieved.source_nodes = list(reranked_nodes)
            return retrieved
        except Exception:
            return SimpleNamespace(source_nodes=list(reranked_nodes))

    def _build_subquestion_result(
        self,
        state: SubQuestionState,
        *,
        accepted: bool,
        failure_reason: str | None = None,
        retried_retrieval: bool = False,
    ) -> SubQuestionResult:
        """Convert the subquestion graph state into a structured execution trace."""

        deterministic = state.get("deterministic_result") or DeterministicValidationResult(False, ["Answer was not produced."])
        return SubQuestionResult(
            subquestion=normalize_text(state.get("subquestion", "")),
            retrieval_query=normalize_text(state.get("retrieval_query") or state.get("subquestion") or ""),
            retrieval_response=state.get("reranked_response") or state.get("retrieval_response"),
            context=state.get("context", ""),
            prompt=state.get("prompt", ""),
            answer_raw=state.get("answer_raw"),
            deterministic_result=deterministic,
            critic_verdict=state.get("critic_verdict"),
            accepted=accepted,
            retried_context=bool(state.get("context_retry_count", 0)),
            retried_retrieval=retried_retrieval,
            failure_reason=failure_reason,
            retrieval_feedback=state.get("retrieval_feedback"),
            notes=copy_notes(state.get("notes")),
        )

    def _build_subquestion_graph(self):
        """Build the LangGraph that processes one sub-question end to end."""

        workflow: StateGraph[SubQuestionState] = StateGraph(SubQuestionState)
        workflow.add_node("retrieve", self._subgraph_retrieve)
        workflow.add_node("rerank", self._subgraph_rerank)
        workflow.add_node("assemble_context", self._subgraph_assemble_context)
        workflow.add_node("generate_answer", self._subgraph_generate_answer)
        workflow.add_node("deterministic_verify", self._subgraph_deterministic_verify)
        workflow.add_node("strict_context_retry", self._subgraph_strict_context_retry)
        workflow.add_node("critic_verify", self._subgraph_critic_verify)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "assemble_context")
        workflow.add_edge("assemble_context", "generate_answer")
        workflow.add_edge("generate_answer", "deterministic_verify")
        workflow.add_conditional_edges(
            "deterministic_verify",
            self._route_after_deterministic,
            {
                "retry_context": "strict_context_retry",
                "critic_verify": "critic_verify",
                "end": END,
            },
        )
        workflow.add_edge("strict_context_retry", "assemble_context")
        workflow.add_conditional_edges(
            "critic_verify",
            self._route_after_critic,
            {
                "retry_context": "strict_context_retry",
                "end": END,
            },
        )
        return workflow.compile()

    def _build_chatbot_graph(self):
        """Build the top-level LangGraph for the full chatbot run."""

        workflow: StateGraph[ChatbotState] = StateGraph(ChatbotState)
        workflow.add_node("initialize_run", self._graph_initialize_run)
        workflow.add_node("normalize_language", self._graph_normalize_language)
        workflow.add_node("rewrite_query", self._graph_rewrite_query)
        workflow.add_node("decompose_query", self._graph_decompose_query)
        workflow.add_node("queue_subquestions", self._graph_queue_subquestions)
        workflow.add_node("finalize_run", self._graph_finalize_run)
        workflow.set_entry_point("initialize_run")
        workflow.add_edge("initialize_run", "normalize_language")
        workflow.add_edge("normalize_language", "rewrite_query")
        workflow.add_edge("rewrite_query", "decompose_query")
        workflow.add_edge("decompose_query", "queue_subquestions")
        workflow.add_conditional_edges(
            "queue_subquestions",
            self._route_after_queue_subquestions,
            {
                "rewrite_query": "rewrite_query",
                "finalize_run": "finalize_run",
            },
        )
        workflow.add_edge("finalize_run", END)
        return workflow.compile()

    def _route_after_deterministic(self, state: SubQuestionState) -> str:
        """Pick the next subquestion node after deterministic verification."""

        result = state.get("deterministic_result")
        if result and result.is_valid:
            return "critic_verify"
        if state.get("context_retry_count", 0) < self.max_context_retries:
            return "retry_context"
        return "end"

    def _route_after_critic(self, state: SubQuestionState) -> str:
        """Pick the next subquestion node after critic verification."""

        verdict = state.get("critic_verdict") or {}
        if verdict.get("valid") is True:
            return "end"
        if verdict.get("missing_context"):
            if state.get("context_retry_count", 0) < self.max_context_retries:
                return "retry_context"
            return "end"
        if state.get("context_retry_count", 0) < self.max_context_retries and verdict.get("unsupported_claims"):
            return "retry_context"
        return "end"

    def _route_after_queue_subquestions(self, state: ChatbotState) -> str:
        """Route the top-level graph after all queued subquestions finish."""

        if state.get("needs_rewrite_retry") and state.get("retry_count", 0) < self.max_retrieval_rounds:
            return "rewrite_query"
        return "finalize_run"

    @staticmethod
    def _subquestion_key(subquestion: str) -> str:
        """Normalize a subquestion key for merge and reuse checks."""

        return normalize_text(subquestion).lower()

    def _subgraph_retrieve(self, state: SubQuestionState) -> SubQuestionState:
        """Retrieve candidate chunks for one sub-question."""

        query = normalize_text(state.get("retrieval_query") or state.get("subquestion") or "")
        self._emit_status(f"Retrieving chunks for: {query or 'sub-question'}...")
        started_at = time.perf_counter()
        retrieved = self._retrieve(query)
        self._record_component_timing("vector_search", started_at)
        return {
            "retrieval_query": query,
            "retrieval_response": retrieved,
            "notes": copy_notes(state.get("notes")),
        }

    def _subgraph_rerank(self, state: SubQuestionState) -> SubQuestionState:
        """Apply the configured cross-encoder reranker to retrieved chunks."""

        query = normalize_text(state.get("retrieval_query") or state.get("subquestion") or "")
        retrieved = state.get("retrieval_response")
        self._emit_status("Reranking retrieved chunks...")
        started_at = time.perf_counter()
        reranked = self._rerank_retrieved(query, retrieved)
        self._record_component_timing("reranker", started_at)
        return {
            "reranked_response": reranked,
            "notes": copy_notes(state.get("notes")),
        }

    def _subgraph_assemble_context(self, state: SubQuestionState) -> SubQuestionState:
        """Assemble a grounded context block and answer prompt."""

        source = state.get("reranked_response") or state.get("retrieval_response")
        self._emit_status("Assembling grounded context...")
        started_at = time.perf_counter()
        context = self._build_context(source)
        prompt = self._build_answer_prompt(
            state.get("subquestion", ""),
            context,
            strict=bool(state.get("strict_context")),
        )
        self._record_component_timing("prompt_assembly", started_at)
        return {
            "context": context,
            "prompt": prompt,
            "notes": copy_notes(state.get("notes")),
        }

    def _subgraph_generate_answer(self, state: SubQuestionState) -> SubQuestionState:
        """Call the answer model using the assembled prompt."""

        prompt = state.get("prompt", "")
        self._emit_status("Generating answer...")
        started_at = time.perf_counter()
        answer_raw = self._ask_answer_model(prompt)
        self._record_component_timing("answer_model", started_at)
        return {
            "answer_raw": answer_raw,
            "notes": copy_notes(state.get("notes")),
        }

    def _subgraph_deterministic_verify(self, state: SubQuestionState) -> SubQuestionState:
        """Run deterministic grounding checks on the answer."""

        response = state.get("reranked_response") or state.get("retrieval_response")
        self._emit_status("Running deterministic validation layer...")
        started_at = time.perf_counter()
        deterministic_result = validate_deterministic_output(state.get("answer_raw"), response)
        self._record_component_timing("deterministic_validation", started_at)
        notes = copy_notes(state.get("notes"))
        notes.extend(deterministic_result.errors)

        if deterministic_result.is_valid:
            return {
                "deterministic_result": deterministic_result,
                "notes": notes,
            }

        if state.get("context_retry_count", 0) >= self.max_context_retries:
            subquestion_result = self._build_subquestion_result(
                {
                    **state,
                    "deterministic_result": deterministic_result,
                    "notes": notes,
                },
                accepted=False,
                failure_reason="verification_failed",
            )
            return {
                "deterministic_result": deterministic_result,
                "subquestion_result": subquestion_result,
                "accepted": False,
                "notes": notes,
            }

        return {
            "deterministic_result": deterministic_result,
            "notes": notes,
        }

    def _subgraph_strict_context_retry(self, state: SubQuestionState) -> SubQuestionState:
        """Rebuild only the prompt with stricter context-only instructions."""

        retry_count = state.get("context_retry_count", 0) + 1
        notes = copy_notes(state.get("notes"))
        notes.append("Rebuilt the prompt with stricter context-only instructions.")
        return {
            "context_retry_count": retry_count,
            "strict_context": True,
            "notes": notes,
        }

    def _subgraph_critic_verify(self, state: SubQuestionState) -> SubQuestionState:
        """Run the critic LLM verification layer."""

        context = state.get("context", "")
        answer_payload = parse_answer_payload(state.get("answer_raw")) or state.get("answer_raw")
        notes = copy_notes(state.get("notes"))
        self._emit_status("Running critic verification...")

        try:
            started_at = time.perf_counter()
            verdict = verify_with_critic(context, answer_payload, lambda prompt: invoke_text_model(self.critic_model, prompt))
            self._record_component_timing("critic_model", started_at)
        except Exception as exc:
            verdict = {
                "valid": False,
                "unsupported_claims": [f"Critic verification failed: {exc}"],
                "missing_context": [],
            }
            notes.append(f"Critic verification failed: {exc}")

        if verdict.get("valid") is True:
            result = self._build_subquestion_result(
                {
                    **state,
                    "critic_verdict": verdict,
                    "notes": notes,
                },
                accepted=True,
            )
            return {
                "critic_verdict": verdict,
                "subquestion_result": result,
                "accepted": True,
                "notes": notes,
            }

        missing_context = verdict.get("missing_context") or []
        unsupported_claims = verdict.get("unsupported_claims") or []

        if missing_context:
            notes.append("Critic reported missing or irrelevant context.")
            feedback = "; ".join(str(item) for item in missing_context if item) or "Retrieved context was missing or irrelevant."
            result = self._build_subquestion_result(
                {
                    **state,
                    "critic_verdict": verdict,
                    "retrieval_feedback": feedback,
                    "notes": notes,
                },
                accepted=False,
                failure_reason="critic_missing_context",
            )
            return {
                "critic_verdict": verdict,
                "subquestion_result": result,
                "accepted": False,
                "needs_rewrite_retry": True,
                "retrieval_feedback": feedback,
                "notes": notes,
            }

        if unsupported_claims:
            notes.append("Critic detected hallucination or instruction slip.")

        if state.get("context_retry_count", 0) >= self.max_context_retries:
            result = self._build_subquestion_result(
                {
                    **state,
                    "critic_verdict": verdict,
                    "notes": notes,
                },
                accepted=False,
                failure_reason="verification_failed",
            )
            return {
                "critic_verdict": verdict,
                "subquestion_result": result,
                "accepted": False,
                "notes": notes,
            }

        return {
            "critic_verdict": verdict,
            "notes": notes,
        }

    def _graph_initialize_run(self, state: ChatbotState) -> ChatbotState:
        """Initialize the chatbot run state."""

        return {
            "original_query": state.get("original_query", ""),
            "rewrite_feedback": state.get("rewrite_feedback"),
            "retry_count": state.get("retry_count", 0),
            "subquestion_results": [],
            "subquestions": [],
            "final_answer": "",
            "accepted": False,
            "needs_rewrite_retry": False,
        }

    def _graph_normalize_language(self, state: ChatbotState) -> ChatbotState:
        """Translate the user query into the working language."""

        self._emit_status("Normalizing query language...")
        normalized_query = self._normalize_query(state.get("original_query", ""))
        return {"normalized_query": normalized_query}

    def _graph_rewrite_query(self, state: ChatbotState) -> ChatbotState:
        """Rewrite the normalized query into a retrieval-friendly query."""

        self._emit_status("Rewriting query for retrieval...")
        normalized_query = state.get("normalized_query") or self._normalize_query(state.get("original_query", ""))
        rewritten_query = rewrite_user_query(
            normalized_query.normalized,
            model=self.rewrite_model,
            feedback=state.get("rewrite_feedback"),
        )
        return {
            "rewritten_query": rewritten_query,
            "needs_rewrite_retry": False,
        }

    def _graph_decompose_query(self, state: ChatbotState) -> ChatbotState:
        """Break the rewritten query into smaller retrieval objectives."""

        self._emit_status("Decomposing query into sub-questions...")
        rewritten_query = normalize_text(state.get("rewritten_query", ""))
        subquestions = decompose_rewritten_query(
            rewritten_query,
            model=self.decomposition_model,
            feedback=state.get("rewrite_feedback"),
            max_subquestions=self.max_subquestions,
        )
        if not subquestions:
            fallback = rewritten_query or (state.get("normalized_query").normalized if state.get("normalized_query") else "")
            subquestions = [fallback] if fallback else []
        return {"subquestions": subquestions[: self.max_subquestions]}

    def _run_subquestion_graph(self, subquestion: str, *, retrieval_query: str, retried_retrieval: bool) -> SubQuestionResult:
        """Execute the per-subquestion LangGraph and return the structured trace."""

        result_state = self.subquestion_graph.invoke(
            {
                "subquestion": subquestion,
                "retrieval_query": retrieval_query,
                "strict_context": False,
                "context_retry_count": 0,
                "notes": [],
            }
        )
        result = result_state.get("subquestion_result")
        if not isinstance(result, SubQuestionResult):
            deterministic_result = result_state.get("deterministic_result") or DeterministicValidationResult(False, ["Answer was not produced."])
            result = SubQuestionResult(
                subquestion=normalize_text(subquestion),
                retrieval_query=normalize_text(retrieval_query),
                retrieval_response=result_state.get("reranked_response") or result_state.get("retrieval_response"),
                context=result_state.get("context", ""),
                prompt=result_state.get("prompt", ""),
                answer_raw=result_state.get("answer_raw"),
                deterministic_result=deterministic_result,
                critic_verdict=result_state.get("critic_verdict"),
                accepted=bool(result_state.get("accepted")),
                retried_context=bool(result_state.get("context_retry_count", 0)),
                retried_retrieval=retried_retrieval,
                failure_reason=result_state.get("failure_reason"),
                notes=copy_notes(result_state.get("notes")),
            )
        else:
            result.retried_retrieval = retried_retrieval
        return result

    def _graph_queue_subquestions(self, state: ChatbotState) -> ChatbotState:
        """Queue each sub-question through retrieval, generation, and verification."""

        normalized_query = state.get("normalized_query") or self._normalize_query(state.get("original_query", ""))
        rewritten_query = normalize_text(state.get("rewritten_query") or normalized_query.normalized)
        subquestions = list(state.get("subquestions") or [])
        if not subquestions:
            subquestions = [rewritten_query] if rewritten_query else []

        retry_count = state.get("retry_count", 0)
        previous_results = list(state.get("subquestion_results") or [])
        previous_by_key: dict[str, SubQuestionResult] = {
            self._subquestion_key(result.subquestion): result for result in previous_results if result.subquestion
        }
        results_by_key: dict[str, SubQuestionResult] = {}
        ordered_keys: list[str] = []
        seen_keys: set[str] = set()
        needs_rewrite_retry = False
        rewrite_feedback = state.get("rewrite_feedback")
        missing_context_feedbacks: list[str] = []

        for subquestion in subquestions:
            retrieval_query = normalize_text(subquestion or rewritten_query)
            key = self._subquestion_key(subquestion or retrieval_query)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_keys.append(key)
            self._emit_status(f"Processing sub-question: {subquestion or retrieval_query}...")

            previous_result = previous_by_key.get(key)
            if previous_result and previous_result.accepted:
                results_by_key[key] = previous_result
                continue

            result = self._run_subquestion_graph(
                subquestion,
                retrieval_query=retrieval_query,
                retried_retrieval=retry_count > 0 or bool(previous_result),
            )
            results_by_key[key] = result

            if result.failure_reason == "critic_missing_context":
                feedback = result.retrieval_feedback or "Retrieved context was missing or irrelevant."
                missing_context_feedbacks.append(feedback)
                if retry_count + 1 < self.max_retrieval_rounds:
                    needs_rewrite_retry = True
                continue

        if missing_context_feedbacks:
            rewrite_feedback = " ".join(dict.fromkeys(missing_context_feedbacks))
            if retry_count + 1 < self.max_retrieval_rounds:
                retry_count += 1

        results = [results_by_key[key] for key in ordered_keys if key in results_by_key]
        accepted = bool(results) and all(result.accepted for result in results)
        return {
            "subquestion_results": results,
            "needs_rewrite_retry": needs_rewrite_retry,
            "rewrite_feedback": rewrite_feedback,
            "retry_count": retry_count,
            "accepted": accepted,
        }

    def _graph_finalize_run(self, state: ChatbotState) -> ChatbotState:
        """Assemble the final answer from the per-subquestion results."""

        results = list(state.get("subquestion_results") or [])
        return {
            "final_answer": self._assemble_final_answer(results),
            "accepted": bool(results) and all(result.accepted for result in results),
        }

    def run(
        self,
        user_prompt: str,
        *,
        rewrite_feedback: str | None = None,
    ) -> ChatbotRunResult:
        """Run the full chatbot graph for a single user prompt."""

        self._component_timings = []
        result_state = self.graph.invoke(
            {
                "original_query": user_prompt,
                "rewrite_feedback": rewrite_feedback,
                "retry_count": 0,
                "subquestion_results": [],
                "subquestions": [],
                "final_answer": "",
                "accepted": False,
                "needs_rewrite_retry": False,
            }
        )

        normalized_query = result_state.get("normalized_query") or self._normalize_query(user_prompt)
        subquestion_results = list(result_state.get("subquestion_results") or [])
        return ChatbotRunResult(
            original_query=user_prompt,
            normalized_query=normalized_query,
            rewritten_query=result_state.get("rewritten_query", ""),
            subquestions=list(result_state.get("subquestions") or []),
            subquestion_results=subquestion_results,
            final_answer=result_state.get("final_answer", ""),
            accepted=bool(result_state.get("accepted")) and bool(subquestion_results),
            retry_count=int(result_state.get("retry_count", 0)),
            feedback=result_state.get("rewrite_feedback"),
            component_timings=list(self._component_timings),
        )

    def _assemble_final_answer(self, subquestion_results: list[SubQuestionResult]) -> str:
        """Combine accepted sub-answers into a single response payload."""

        answers: list[str] = []
        for result in subquestion_results:
            if not result.accepted:
                continue
            payload = parse_answer_payload(result.answer_raw)
            answer_text = payload.get("answer")
            if isinstance(answer_text, str) and answer_text.strip():
                answers.append(answer_text.strip())

        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]
        return "\n".join(f"- {answer}" for answer in answers)
