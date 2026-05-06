# Nullwatch Python SDK

Python client for instrumenting LLM and agent runs with spans, evaluations, and
quality signals.

Nullwatch is designed for local-first observability. The SDK keeps the public
API small: create spans around work, ingest explicit evals, run optional
scorers, and query run summaries back from a Nullwatch service.

By default, `NullwatchClient()` connects to `http://127.0.0.1:7710`.

## Install

Core client:

```bash
pip install nullwatch-py
```

Client plus RAG hallucination detection:

```bash
pip install "nullwatch-py[rag]"
```

## Quick Start

```python
from nullwatch import NullwatchClient
from nullwatch.scorers import RAGHallucinationScorer, ToolCallScorer


client = NullwatchClient()

with client.span("run-123", "llm.call", model="gpt-4o") as span:
    response = call_llm(prompt)
    span.input_tokens = response.usage.prompt_tokens
    span.output_tokens = response.usage.completion_tokens
    span.cost_usd = response.usage.total_cost

rag_scorer = RAGHallucinationScorer()
client.ingest_eval(
    rag_scorer.score(
        run_id="run-123",
        contexts=docs,
        question=q,
        answer=response.text,
    )
)

tool_scorer = ToolCallScorer(tools=MY_TOOLS)
client.ingest_eval(
    tool_scorer.score("run-123", tool_calls=response.tool_calls)
)
```

## Core Concepts

### Runs

A run is the top-level unit of work. Use the same `run_id` for every span and
eval that belongs to one agent execution, workflow step, request, or test case.

```python
run_id = "run-123"
```

### Spans

Spans represent timed work inside a run: an LLM call, retrieval step, tool
execution, parser pass, reranker request, or any other operation worth
measuring.

The context manager starts the timer on entry, finishes it on exit, captures
errors, and ingests the span automatically.

```python
with client.span("run-123", "llm.call", model="gpt-4o") as span:
    response = call_llm(prompt)
    span.input_tokens = response.usage.prompt_tokens
    span.output_tokens = response.usage.completion_tokens
    span.cost_usd = response.usage.total_cost
```

Use stable span names. Good names describe the operation, not one specific
implementation detail:

```text
llm.call
retriever.search
tool.execute
workflow.step
parser.extract_json
```

### Evals

Evals attach quality signals to a run. They can come from deterministic checks,
human review, an LLM judge, a RAG hallucination detector, or a custom scorer.

```python
from nullwatch import Eval


client.ingest_eval(
    Eval(
        run_id="run-123",
        eval_key="helpfulness",
        scorer="llm-judge",
        score=0.94,
        verdict="pass",
    )
)
```

Recommended fields:

```text
run_id      Stable run identifier.
eval_key    What is being measured, for example "helpfulness".
scorer      The scorer or system that produced the result.
score       Numeric score when available.
verdict     "pass", "fail", or "warn".
notes       Human-readable details for debugging.
metadata    Structured details for downstream analysis.
```

## RAG Hallucination Detection

`RAGHallucinationScorer` is intended for retrieval-augmented generation
workflows. It compares an answer against retrieved context and returns an eval
that marks unsupported answer spans.

The `rag` extra is designed around
[LettuceDetect](https://pypi.org/project/lettucedetect/), a ModernBERT-based
token classifier for RAG hallucination detection. The large English model is
published as
[`KRLabsOrg/lettucedect-large-modernbert-en-v1`](https://huggingface.co/KRLabsOrg/lettucedect-large-modernbert-en-v1).

```python
from nullwatch.scorers import RAGHallucinationScorer


scorer = RAGHallucinationScorer()

eval_ = scorer.score(
    run_id="run-123",
    contexts=[
        "The capital of France is Paris. Population is 68 million.",
    ],
    question="What is the capital and population of France?",
    answer="The capital is Paris. The population is 80 million.",
)

client.ingest_eval(eval_)

print(eval_.verdict)  # "fail"
print(eval_.notes)    # 'Hallucinated spans detected: "80 million" (conf=0.97)'
```

The scorer should report enough detail to debug the failure without replaying
the whole request:

```text
eval_key: rag_hallucination
scorer:   lettucedetect
verdict:  fail
notes:    unsupported spans and confidence
metadata: spans, offsets, confidence, model name
```

## Tool-Call Validity

`ToolCallScorer` validates LLM-generated tool calls against a declared tool
schema. It catches fabricated tool names, misspelled arguments, missing required
fields, and wrong argument types. It does not require an ML model.

You can pass the compact Nullwatch schema:

```python
from nullwatch.scorers import ToolCallScorer


tool_scorer = ToolCallScorer(
    tools=[
        {
            "name": "search_web",
            "parameters": {
                "query": {"type": "string", "required": True},
            },
        }
    ]
)

eval_ = tool_scorer.score(
    run_id="run-123",
    tool_call={
        "name": "search_web",
        "arguments": {"querY": "zig lang"},
    },
)

print(eval_.verdict)  # "fail"
print(eval_.notes)    # "Unknown argument 'querY' (did you mean: ['query'])?"
```

You can also pass the same OpenAI-style `tools=[...]` JSON schema you send to
the model. This keeps validation close to production behavior and avoids a
second source of truth.

## Querying Runs

Use query methods when you need to build dashboards, inspect failures, or
export slices of evaluation data.

```python
summary = client.get_run("run-123")
print(summary.span_count, summary.eval_count, summary.verdict)

failed_rag_evals = client.list_evals(
    verdict="fail",
    eval_key="rag_hallucination",
)

error_spans = client.list_spans(status="error")
```

Expected query surface:

```text
get_run(run_id)
list_runs(...)
list_spans(run_id=None, status=None, name=None)
list_evals(run_id=None, verdict=None, eval_key=None, scorer=None)
```

## Configuration

```python
client = NullwatchClient(
    base_url="http://127.0.0.1:7710",
    api_key=None,
    timeout=10.0,
)
```

Configuration should also be readable from environment variables:

```bash
export NULLWATCH_URL=http://127.0.0.1:7710
export NULLWATCH_API_KEY=...
```

## Architecture

The SDK should stay small and layered:

```text
nullwatch/
  __init__.py          Public exports.
  client.py            NullwatchClient and high-level API.
  models.py            Span, Eval, RunSummary, typed payloads.
  transport.py         HTTP transport, retries, timeouts, errors.
  spans.py             Span context manager and lifecycle.
  scorers/
    base.py            Scorer protocol and common result helpers.
    rag.py             RAGHallucinationScorer.
    tool_calls.py      ToolCallScorer.
```

Design rules:

```text
Keep data models serializable.
Keep transport replaceable for tests.
Keep scorers optional and dependency-isolated.
Keep network ingestion explicit except for span context manager exit.
Keep deterministic validators free of ML dependencies.
```

## Development

```bash
make install
make lint
make test
```

`make lint` runs Ruff and should report zero errors. `make test` should run
locally without external services; scorer tests should use fixtures or mocked
model outputs unless an integration test is explicitly requested.
