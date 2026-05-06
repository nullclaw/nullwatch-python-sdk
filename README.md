# Nullwatch Python SDK

Python SDK for instrumenting LLM and agent applications with traces, spans,
evals, scorers, and run queries backed by
[`nullwatch`](https://github.com/nullclaw/nullwatch).

`nullwatch-py` is the Python entry point for Nullwatch observability. It records
what happened inside a run, attaches quality signals to that run, and makes the
results queryable for dashboards, CI checks, local debugging, regression suites,
and production monitoring.

By default, `NullwatchClient()` connects to `http://127.0.0.1:7710`.

## What It Does

`nullwatch-py` covers the Python side of the Nullwatch workflow:

- records spans for LLM calls, retrieval, parsing, tool calls, workflow steps,
  retries, fallbacks, and custom application operations
- ingests evals from deterministic checks, human review, LLM judges, test
  suites, RAG hallucination detection, and tool-call validation
- queries runs, spans, evals, summaries, verdicts, costs, latency, token usage,
  and error status from the Nullwatch service
- provides ergonomic Python APIs: context managers, decorators, explicit data
  models, buffered ingestion, test helpers, and a small CLI
- keeps heavyweight scorer dependencies optional so the core client remains
  small and usable in normal application code

The boundary is simple: Python code produces telemetry and evals; `nullwatch`
stores, summarizes, and exposes them.

## Install

Core client:

```bash
pip install nullwatch-py
```

Client plus RAG hallucination detection:

```bash
pip install "nullwatch-py[rag]"
```

Development tools:

```bash
pip install "nullwatch-py[dev]"
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
eval that belongs to one agent execution, workflow step, HTTP request, CLI run,
background job, dataset item, or test case.

```python
run_id = "run-123"
```

### Spans

Spans represent timed work inside a run:

- model calls
- tool invocations
- retrieval and reranking
- memory lookups
- parser passes
- workflow steps
- retries and fallback branches
- custom application operations

The context manager starts the timer on entry, finishes it on exit, captures
errors, and ingests the span automatically.

```python
with client.span("run-123", "llm.call", model="gpt-4o") as span:
    response = call_llm(prompt)
    span.input_tokens = response.usage.prompt_tokens
    span.output_tokens = response.usage.completion_tokens
    span.cost_usd = response.usage.total_cost
```

Stable operation names make dashboards and regression queries easier to read:

```text
llm.call
retriever.search
retriever.rerank
tool.execute
workflow.step
parser.extract_json
memory.lookup
```

### Evals

Evals attach quality signals to a run. They can come from deterministic checks,
human review, an LLM judge, a RAG hallucination detector, a schema validator, a
dataset regression suite, or a custom scorer.

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
dataset     Dataset, environment, or regression suite name.
notes       Human-readable details for debugging.
metadata    Structured details for downstream analysis.
```

## Client API

The client covers the common lifecycle for Python agents and RAG services:

```python
client = NullwatchClient()

client.health()
client.capabilities()

client.ingest_span(span)
client.ingest_spans([span_a, span_b])
client.ingest_eval(eval_)

client.get_run("run-123")
client.list_runs(limit=20)
client.list_spans(run_id="run-123", status="error")
client.list_evals(verdict="fail", eval_key="rag_hallucination")

client.flush()
client.close()
```

The default client sends data immediately. Buffered mode batches span ingest
through `/v1/spans/bulk` and flushes on context-manager exit:

```python
with NullwatchClient(buffered=True, flush_at=100) as client:
    ...
```

Core capabilities:

```text
Connection
  base URL, API token, timeout, retry policy, health check, capabilities query

Ingestion
  single span, bulk spans, single eval, buffered mode, explicit flush,
  graceful close

Queries
  run detail, run list, span list, eval list, summaries and filtered views

Instrumentation
  sync spans, async spans, decorators, parent/child span relationships,
  generated trace/span IDs, automatic error capture

Data hygiene
  redaction hook, metadata normalization, serializable payloads,
  predictable timestamps

Testing
  in-memory transport, assertion helpers, no-server test path
```

## Decorators

Use decorators when application code is already organized into functions.

```python
@client.trace("retriever.search")
def search_docs(run_id: str, query: str) -> list[str]:
    return retriever.search(query)
```

Async functions use the same model:

```python
@client.atrace("llm.call")
async def call_model(run_id: str, prompt: str) -> str:
    return await model.generate(prompt)
```

The decorator reads `run_id` from keyword arguments by default and can fall back
to a generated run ID when no run ID is available.

## RAG Hallucination Detection

`RAGHallucinationScorer` is built for retrieval-augmented generation workflows.
It compares an answer against retrieved context and returns an eval that marks
unsupported answer spans.

The `rag` extra uses
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

RAG scorer output:

```text
eval_key: rag_hallucination
scorer:   lettucedetect
verdict:  pass | fail | warn
score:    confidence-adjusted support score
notes:    unsupported spans and confidence
metadata: spans, offsets, confidence, model name, threshold
```

The scorer is optional and dependency-isolated. Importing `nullwatch` or using
the core client does not import PyTorch, Transformers, or LettuceDetect. Those
dependencies live behind the `rag` extra.

## Tool-Call Validity

`ToolCallScorer` validates LLM-generated tool calls against a declared tool
schema. It catches fabricated tool names, misspelled arguments, missing required
fields, malformed JSON arguments, enum violations, and wrong argument types. It
does not require an ML model.

Compact Nullwatch schema:

You can pass either:

- the compact `nullwatch-py` schema format shown below, or
- the same OpenAI-style `tools=[...]` JSON schema you send to the model

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

The same scorer also accepts OpenAI-style `tools=[...]` JSON schema, so
applications can validate against the exact schema sent to the model.

Tool-call scorer output:

```text
eval_key: tool_call_validity
scorer:   schema
verdict:  pass | fail
score:    1.0 | 0.0
notes:    concise failure summary
metadata: unknown_tools, missing_args, unknown_args, type_errors
```

Supported inputs:

```text
one tool call
multiple tool calls
dict-like provider responses
response objects with tool_calls attributes
compact Nullwatch schema
OpenAI-style tools JSON schema
```

## Provider Helpers

Provider helpers make common LLM usage cheap to instrument without coupling the
SDK to a provider SDK.

```python
with client.span("run-123", "llm.call", model="gpt-4o") as span:
    response = openai_client.chat.completions.create(...)
    span.record_openai_usage(response)
```

Recommended helpers:

```text
record_openai_usage(response)
record_anthropic_usage(response)
record_tokens(input_tokens=..., output_tokens=...)
record_cost(cost_usd=...)
```

These helpers are best-effort adapters over response objects and dictionaries.
They do not require OpenAI, Anthropic, or other provider packages at import time.

## Querying Runs

Use query methods to build dashboards, inspect failures, export evaluation data,
or run regression checks in CI.

```python
summary = client.get_run("run-123")
print(summary.span_count, summary.eval_count, summary.verdict)

failed_rag_evals = client.list_evals(
    verdict="fail",
    eval_key="rag_hallucination",
)

error_spans = client.list_spans(status="error")
```

Query surface:

```text
get_run(run_id)
list_runs(...)
list_spans(run_id=None, status=None, name=None)
list_evals(run_id=None, verdict=None, eval_key=None, scorer=None)
```

Filters mirror the `nullwatch` service:

```text
Runs
  run_id, verdict, source, model, dataset, limit, before/after

Spans
  run_id, trace_id, status, source, operation, model, tool_name,
  prompt_version, limit

Evals
  run_id, eval_key, scorer, verdict, dataset, limit
```

## Configuration

```python
client = NullwatchClient(
    base_url="http://127.0.0.1:7710",
    api_key=None,
    timeout=10.0,
)
```

Environment variables:

```bash
export NULLWATCH_URL=http://127.0.0.1:7710
export NULLWATCH_API_KEY=...
```

## Testing Utilities

Test helpers let application code assert telemetry without running a real
`nullwatch` server.

```python
from nullwatch.testing import MemoryTransport


transport = MemoryTransport()
client = NullwatchClient(transport=transport)

with client.span("run-123", "tool.execute", tool_name="search"):
    pass

assert len(transport.spans) == 1
```

Useful test helpers:

```text
MemoryTransport
assert_no_failed_evals(...)
assert_span_recorded(...)
assert_eval_recorded(...)
```

## Redaction

Production users can remove secrets and sensitive payloads before ingest.

```python
client = NullwatchClient(
    redact=lambda payload: scrub_secrets(payload),
)
```

Redaction runs immediately before transport serialization for spans, evals, and
query metadata. It is deterministic, local, and opt-in.

Recommended redaction targets:

```text
API keys and bearer tokens
passwords and session cookies
raw prompts when an application marks them private
tool arguments that contain credentials or private file paths
provider response payloads before they are copied into metadata
```

## CLI

The SDK includes a small CLI for debugging and shell scripts:

```bash
nullwatch-py ping
nullwatch-py ingest-span span.json
nullwatch-py ingest-eval eval.json
nullwatch-py run run-123
```

This CLI is a convenience wrapper over the Python client. Operational server
commands belong to the `nullwatch` Zig binary.

## Architecture

The package is small and layered:

```text
nullwatch/
  __init__.py          Public exports.
  errors.py            Public exception hierarchy.
  client.py            NullwatchClient and high-level API.
  models.py            Span, Eval, RunSummary, typed payloads.
  transport.py         HTTP transport, retries, timeouts, errors.
  spans.py             Span context manager and lifecycle.
  testing.py           MemoryTransport and assertion helpers.
  cli.py               Small debugging CLI.
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

Package extras:

```text
nullwatch-py          core client, models, transport, tool-call scorer
nullwatch-py[rag]    LettuceDetect-backed RAG hallucination scorer
nullwatch-py[dev]    tests, Ruff, type checking, build tooling
```

## Compatibility With Nullwatch

The SDK maps directly to the `nullwatch` service contract:

```text
POST /v1/spans        ingest one span
POST /v1/spans/bulk   ingest many spans
POST /v1/evals        ingest one eval
GET  /v1/runs         list runs
GET  /v1/runs/{id}    get run detail
GET  /v1/spans        list spans
GET  /v1/evals        list evals
GET  /v1/capabilities inspect server capabilities
GET  /health          health check
```

OTLP ingestion stays on the `nullwatch` service side. The Python SDK provides
helpers for mapping Python spans into the native Nullwatch span model, but it is
not an OpenTelemetry collector.

## Development

```bash
make install
make lint
make test
```

`make lint` runs Ruff and reports zero errors. `make test` runs locally without
external services; scorer tests use fixtures or mocked model outputs unless an
integration test is explicitly requested.
