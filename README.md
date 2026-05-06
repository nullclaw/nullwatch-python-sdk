# Nullwatch Python SDK

Python client for instrumenting LLM and agent runs with spans, evaluations, and
quality signals.

Nullwatch is designed for local-first observability. The SDK keeps the public
API small: create spans around work, ingest explicit evals, run optional
scorers, and query run summaries back from a Nullwatch service.

By default, `NullwatchClient()` connects to `http://127.0.0.1:7710`.

## Status

This repository is being initialized. The README describes the intended public
API, package shape, and design contract for the Python SDK. Implementation
should follow this surface without turning the SDK into a second `nullwatch`
server or a UI layer.

## Repository Scope

This repository is the Python SDK home for `nullwatch`. Its job is to make
Python applications, agent runtimes, RAG services, evaluation scripts, and test
suites easy to connect to the `nullwatch` HTTP API.

The SDK should cover:

- span instrumentation for LLM calls, tool calls, retrieval, parsing, workflow
  steps, retries, and custom operations
- eval ingestion for human review, deterministic checks, LLM judges, regression
  gates, RAG hallucination checks, and tool-call validation
- run queries for summaries, spans, evals, pass/fail status, latency, token
  usage, cost, and error inspection
- ergonomic Python APIs: context managers, decorators, explicit model objects,
  buffered ingestion, and test helpers
- optional scorer packages that can add heavier dependencies without bloating
  the core client

This repository should not contain:

- a `nullwatch` server implementation
- dashboard or UI code
- durable storage engines
- queue ownership, orchestration policy, or scheduling logic
- provider SDK wrappers that hide the underlying OpenAI, Anthropic, or local
  model clients

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

## Quick Start

Target API:

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

## Client Surface

The SDK should cover the common lifecycle for Python agents and RAG services:

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

The default client should send data immediately. Buffered mode should batch span
ingest through `/v1/spans/bulk` and flush on context-manager exit:

```python
with NullwatchClient(buffered=True, flush_at=100) as client:
    ...
```

Core client capabilities:

```text
Connection
  base URL, API token, timeout, retry policy, health check, capabilities query

Ingestion
  single span, bulk spans, single eval, future bulk evals, buffered mode,
  explicit flush, graceful close

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

In addition to explicit span blocks, the SDK should support decorators for code
that is already organized into functions.

```python
@client.trace("retriever.search")
def search_docs(run_id: str, query: str) -> list[str]:
    return retriever.search(query)
```

Async functions should be supported with the same semantics:

```python
@client.atrace("llm.call")
async def call_model(run_id: str, prompt: str) -> str:
    return await model.generate(prompt)
```

The decorator should extract `run_id` from a keyword argument by default and
fall back to a generated run ID only when no run ID is available.

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

The scorer should stay optional and dependency-isolated. Importing
`nullwatch` or using the core client must not import PyTorch, Transformers, or
LettuceDetect. Those dependencies belong behind the `rag` extra.

RAG scorer capabilities:

```text
Input
  run_id, contexts, question, answer, optional dataset, optional metadata

Output
  Eval with eval_key="rag_hallucination", verdict, score, notes, metadata

Metadata
  hallucinated spans, character offsets, confidence, model name, threshold

Behavior
  pass when no unsupported spans are found
  fail when unsupported spans exceed threshold
  warn when the detector cannot produce a reliable result
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

The scorer should return one eval per validation call, with structured metadata
that is easy to inspect in a UI:

```text
eval_key: tool_call_validity
scorer:   schema
verdict:  pass | fail
score:    1.0 | 0.0
notes:    concise failure summary
metadata: unknown_tools, missing_args, unknown_args, type_errors
```

Tool-call validation should support:

```text
Tool schemas
  compact Nullwatch schema
  OpenAI-style tools JSON schema

Tool-call shapes
  one tool call
  multiple tool calls
  dict-like provider responses
  response objects with tool_calls attributes

Checks
  unknown tool name
  missing required argument
  unknown argument with suggestions
  type mismatch
  malformed JSON arguments
  array/object/enum handling where schema provides enough detail
```

## Provider Helpers

Provider helpers should make common LLM usage cheap to instrument without
coupling the SDK to a provider SDK.

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

These helpers should be best-effort adapters over response objects and dicts.
They should never require OpenAI, Anthropic, or other provider packages at import
time.

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

Filtering should mirror the `nullwatch` service where possible:

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

Configuration should also be readable from environment variables:

```bash
export NULLWATCH_URL=http://127.0.0.1:7710
export NULLWATCH_API_KEY=...
```

## Testing Utilities

The SDK should include test helpers so application code can assert telemetry
without running a real `nullwatch` server.

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

Production users need a safe place to remove secrets and sensitive payloads
before ingest.

```python
client = NullwatchClient(
    redact=lambda payload: scrub_secrets(payload),
)
```

Redaction should run immediately before transport serialization for spans,
evals, and query metadata. It should be deterministic, local, and opt-in.

Recommended redaction targets:

```text
API keys and bearer tokens
passwords and session cookies
raw prompts when an application marks them private
tool arguments that contain credentials or private file paths
provider response payloads before they are copied into metadata
```

## CLI

A small SDK CLI is useful for debugging and shell scripts:

```bash
nullwatch-py ping
nullwatch-py ingest-span span.json
nullwatch-py ingest-eval eval.json
nullwatch-py run run-123
```

This CLI should remain a convenience wrapper over the Python client. Operational
server commands belong to the `nullwatch` Zig binary.

## Architecture

The SDK should stay small and layered:

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

The core package should depend only on the Python standard library. Optional
extras can add heavier dependencies:

```text
nullwatch-py          core client, models, transport, tool-call scorer
nullwatch-py[rag]    LettuceDetect-backed RAG hallucination scorer
nullwatch-py[dev]    tests, Ruff, type checking, build tooling
```

## Compatibility With Nullwatch

The SDK should map directly to the `nullwatch` service contract:

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

OTLP ingestion stays on the `nullwatch` service side. The Python SDK may expose
helpers for mapping Python spans into the native Nullwatch span model, but it
should not become an OpenTelemetry collector.

## Documentation Goals

The README should remain the high-level product contract. Once implementation
starts, detailed docs can split into:

```text
docs/api.md           public Python API reference
docs/scorers.md       scorer behavior and output formats
docs/testing.md       MemoryTransport and assertion helpers
docs/integrations.md  provider-specific usage examples
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
