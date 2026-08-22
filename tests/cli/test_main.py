from __future__ import annotations

import json
import importlib
from types import SimpleNamespace

from cli.main import create_chatbot_graph, main as cli_main_entry

cli_main = importlib.import_module("cli.main")


class FakeGraph:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def run(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(final_answer="The model achieved 92.5% accuracy.")


def test_chat_command_prints_answer(monkeypatch, capsys):
    graph = FakeGraph()
    monkeypatch.setattr(cli_main, "create_chatbot_graph", lambda: graph)
    inputs = iter(["What is the model accuracy?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    exit_code = cli_main_entry(["chat"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "92.5%" in output
    assert graph.prompts == ["What is the model accuracy?"]


def test_create_chatbot_graph_uses_chatbot_apikey_and_single_model(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text="ok"))

    class FakeGraph:
        def __init__(self, **kwargs):
            captured["graph_kwargs"] = kwargs

    monkeypatch.setenv("CHATBOT_APIKEY", "secret-key")
    monkeypatch.setattr(cli_main, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(cli_main, "RAGChatbotGraph", FakeGraph)
    monkeypatch.setenv("CHATBOT_MODEL", "gpt-5.4-mini")

    graph = create_chatbot_graph()

    assert isinstance(graph, FakeGraph)
    assert captured["api_key"] == "secret-key"
    assert captured["graph_kwargs"]["answer_model"].model == "gpt-5.4-mini"
    assert captured["graph_kwargs"]["critic_model"].model == "gpt-5.4-mini"
    assert captured["graph_kwargs"]["rewrite_model"].model == "gpt-5.4-mini"
    assert captured["graph_kwargs"]["decomposition_model"].model == "gpt-5.4-mini"


def test_create_chatbot_graph_accepts_chatbot_api_key_alias(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text="ok"))

    class FakeGraph:
        def __init__(self, **kwargs):
            captured["graph_kwargs"] = kwargs

    monkeypatch.delenv("CHATBOT_APIKEY", raising=False)
    monkeypatch.setenv("CHATBOT_API_KEY", "alias-key")
    monkeypatch.setattr(cli_main, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(cli_main, "RAGChatbotGraph", FakeGraph)

    graph = create_chatbot_graph()

    assert isinstance(graph, FakeGraph)
    assert captured["api_key"] == "alias-key"


def test_create_chatbot_graph_accepts_openai_api_key_fallback(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text="ok"))

    class FakeGraph:
        def __init__(self, **kwargs):
            captured["graph_kwargs"] = kwargs

    monkeypatch.delenv("CHATBOT_APIKEY", raising=False)
    monkeypatch.delenv("CHATBOT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(cli_main, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(cli_main, "RAGChatbotGraph", FakeGraph)

    graph = create_chatbot_graph()

    assert isinstance(graph, FakeGraph)
    assert captured["api_key"] == "openai-key"


def test_eval_command_uses_trace_log(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "runs.jsonl"
    log_path.write_text(
        json.dumps({"user_query": "What is the model accuracy?", "final_answer": "The model achieved 92.5% accuracy.", "retrieved_chunks": [], "duration_s": 0.1, "accepted": True, "retry_count": 0, "prompt_tokens_estimate": 10, "answer_tokens_estimate": 4}) + "\n",
        encoding="utf-8",
    )

    from cli import eval_reports as cli_eval_reports
    monkeypatch.setattr(cli_eval_reports, "DEFAULT_LOG_PATH", log_path)

    exit_code = cli_main_entry(["eval", "--json"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "\"query_similarity_report\"" in output
    assert "\"latency_report\"" in output


def test_eval_command_can_run_golden_queries_before_report(tmp_path, monkeypatch, capsys):
    dataset_dir = tmp_path / "golden_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "qrels.json").write_text(json.dumps([{"query_id": "q1", "relevant_context_ids": ["c1"]}]), encoding="utf-8")
    (dataset_dir / "queries.json").write_text(json.dumps([{"query_id": "q1", "question": "What is the model accuracy?"}]), encoding="utf-8")
    (dataset_dir / "answers.json").write_text(json.dumps([{"query_id": "q1", "answer": "The model achieved 92.5% accuracy."}]), encoding="utf-8")

    trace_log = tmp_path / "runs.jsonl"

    class FakeGraph:
        def run(self, prompt: str):
            return SimpleNamespace(
                final_answer="The model achieved 92.5% accuracy.",
                retrieved_chunks=[],
                duration_s=0.1,
                accepted=True,
                retry_count=0,
                prompt_tokens_estimate=10,
                answer_tokens_estimate=4,
            )

    monkeypatch.setattr(cli_main, "create_chatbot_graph", lambda: FakeGraph())
    monkeypatch.setattr(cli_main, "GOLDEN_DATASET_DIR", dataset_dir)
    monkeypatch.setattr(cli_main, "DEFAULT_LOG_PATH", trace_log)
    monkeypatch.setattr(cli_main, "_get_openai_api_key", lambda: "test-key")

    class FakeOpenAI:
        def __init__(self, *, api_key):
            self.responses = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output_text="ok"))

    monkeypatch.setattr(cli_main, "OpenAI", FakeOpenAI)

    from cli import eval_reports as cli_eval_reports
    monkeypatch.setattr(cli_eval_reports, "DEFAULT_LOG_PATH", trace_log)
    monkeypatch.setattr(cli_eval_reports, "GOLDEN_DATASET_DIR", dataset_dir)

    exit_code = cli_main_entry(["eval", "--report", "retrieval-answer-quality", "--run-golden-queries", "--json"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert trace_log.exists()
    assert "\"retrieval_and_answer_quality_report\"" in output


def test_eval_report_selection_supports_all_reports():
    from cli.eval_reports import REPORT_CHOICES, parse_selected_reports

    assert parse_selected_reports(["all"]) == list(REPORT_CHOICES)
