# RAG Chatbot

We build a retrieval-augmented chatbot that turns a user question into a grounded answer using a multi-step workflow:

1. Normalize the user query into a clean working form.
2. Rewrite it into a retrieval-friendly search query.
3. Decompose it into smaller sub-questions when that helps retrieval.
4. Retrieve and optionally rerank relevant chunks from the knowledge base.
5. Assemble a grounded context window and generate an answer.
6. Validate the answer with deterministic checks and a critic model before returning it.

The result is a chatbot that is optimized for grounded answers, traceability, and evalability rather than just a single model completion.

## What’s In The Repo

- `chatbot/` contains the main LangGraph workflow.
- `rag/` contains retrieval, translation, ingestion, and context assembly helpers.
- `cli/` contains the terminal interface and eval report commands.
- `eval/` contains offline evaluation utilities and report builders.
- `tests/` contains the automated test suite.

## Prerequisites

- Python 3.11+.
- An OpenAI API key for chat and most eval workflows.
- Qdrant credentials if you want to use the real retrieval index.
- A populated dataset / index if you want grounded answers from your own corpus.

## Environment Variables

Set the variables that match the parts of the system you want to run:

- `CHATBOT_APIKEY`, `CHATBOT_API_KEY`, or `OPENAI_API_KEY`: OpenAI API key used by the CLI.
- `CHATBOT_MODEL`: model name used by the CLI, defaults to `gpt-5.4-mini`.
- `QDRANT_APIKEY`: Qdrant API key.
- `QDRANT_CLUSTER_ENDPOINT`: Qdrant cluster URL.
- `S3_BUCKET_NAME`: bucket used by ingestion and dataset download helpers.
- `HF_TOKEN`: Hugging Face read token used by `load_data.py`.
- `GOLDEN_DATASET_DIR`: optional path to golden eval data.
- `CHATBOT_CLI_LOG_PATH`: optional path for CLI trace logs.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you plan to use Git hooks, install them once:

```powershell
.\scripts\install-hooks.ps1
```

## Run The CLI

The CLI is exposed as a module:

```bash
python -m cli --help
```

### Interactive Chat

Start a terminal chat session:

```bash
python -m cli chat
```

Type a question, press Enter, and the workflow will:

- normalize the query,
- rewrite it for retrieval,
- decompose it if needed,
- search the index,
- assemble context,
- generate the answer,
- and run validation before returning the final response.

Use `quit` or `exit` to leave the session.

### Eval Reports

Build eval reports from local trace logs:

```bash
python -m cli eval
```

Common flags:

- `--report latency`
- `--report cost`
- `--report embedding-drift`
- `--report query-similarity`
- `--report index-health`
- `--report retrieval-answer-quality`
- `--report all`
- `--run-golden-queries`
- `--trace-log path\to\runs.jsonl`
- `--json`

Example:

```bash
python -m cli eval --report latency --report cost
```

To run the golden eval queries before building grounded-answer metrics:

```bash
python -m cli eval --report retrieval-answer-quality --run-golden-queries
```

## Setting Up In GitHub Codespaces

We do not currently ship a checked-in `.devcontainer/` folder, so the quickest way to get working in Codespaces is:

1. Open the repository in a new Codespace.
2. In the Codespace terminal, run:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Add the environment variables you need in the Codespace secrets / environment settings.
4. If you want to chat against a real index, make sure Qdrant is reachable from the Codespace.
5. Run the smoke checks:

   ```bash
   python -m cli --help
   python -m pytest
   ```

If you want a more polished Codespaces experience, the next step is to add a `.devcontainer/devcontainer.json` plus any bootstrap script for dependencies and secrets.

## Data And Ingestion

The repo also includes helper scripts for working with the corpus and embeddings:

- `load_data.py` downloads Open RAG Bench assets from Hugging Face and mirrors them to S3.
- `rag/ingestion/` contains the ingestion pipeline for documents, embeddings, and Qdrant upload.

Those flows expect the same environment variables listed above, plus a populated dataset and vector store.

## Testing

Run the full test suite with:

```bash
pytest
```

## Notes

- Runtime traces are written to `.chatbot_cli_runs.jsonl` by default.
- Ingestion state and embedding checkpoints are stored in `.ingestion_state.json` and `.embedding_checkpoint.json`.
- The repo is designed so tests can mock the models and retrieval layer, which keeps the core workflow easy to validate locally.
