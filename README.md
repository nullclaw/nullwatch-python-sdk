# Nullwatch Python SDK

Python client for sending observability spans and evaluation results to Nullwatch.

Nullwatch helps track agent and LLM runs, record model-call metadata, and attach
quality signals such as RAG hallucination checks or tool-call validation to the
same run ID.

## Installation

Install only the core client when you need span ingestion and basic reporting:

```bash
pip install nullwatch-py
```

Install the RAG extras when you also need hallucination detection for
retrieval-augmented generation workflows:

```bash
pip install "nullwatch-py[rag]"
```

## Usage

```python
from nullwatch import NullwatchClient
from nullwatch.scorers import RAGHallucinationScorer, ToolCallScorer


client = NullwatchClient()

# Create a span with an automatic timer.
with client.span("run-123", "llm.call", model="gpt-4o") as span:
    response = call_llm(prompt)
    span.input_tokens = response.usage.prompt_tokens
    span.output_tokens = response.usage.completion_tokens

# Evaluate hallucinations in a RAG answer.
scorer = RAGHallucinationScorer()
eval_ = scorer.score(
    "run-123",
    contexts=docs,
    question=q,
    answer=response.text,
)
client.ingest_eval(eval_)

# Evaluate whether tool calls are valid for the available tools.
tool_scorer = ToolCallScorer(tools=MY_TOOLS)
client.ingest_eval(
    tool_scorer.score("run-123", tool_calls=response.tool_calls)
)
```

## Validation

```bash
make install
make lint
make test
```

The lint target runs Ruff and should report zero errors. The test suite is
designed to run locally without external services; the current baseline is 33
tests.
