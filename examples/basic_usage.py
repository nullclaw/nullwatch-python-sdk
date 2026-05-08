from nullwatch import NullwatchClient, Span, Eval
from nullwatch.scorers import ToolCallScorer

# 1. Connect to nullwatch
client = NullwatchClient(
    base_url="http://127.0.0.1:7710",
    raise_on_error=False,  # won't raise if server is not running
)

print("Server alive:", client.is_alive())

# 2. Manual span ingestion
span = Span(
    run_id="run-demo-001",
    operation="llm.call",
    model="gpt-4o",
    input_tokens=420,
    output_tokens=96,
    cost_usd=0.018,
)
span.finish()
client.ingest_span(span)
print("Span ingested:", span.span_id)

# 3. Context-manager span (auto-finish + auto-ingest)
with client.span("run-demo-001", "tool.call", tool_name="search_web") as s:
    # simulate work
    import time

    time.sleep(0.05)
    # you can mutate `s` inside the block
    s.status = "ok"

print("Tool span done, duration_ms:", s.duration_ms)

# 4. Manual eval ingestion
eval_ = Eval(
    run_id="run-demo-001",
    eval_key="helpfulness",
    scorer="llm-judge",
    score=0.94,
    verdict="pass",
    dataset="prod-shadow",
)
client.ingest_eval(eval_)
print("Eval ingested:", eval_.eval_key)

# 5. Tool-call validity scorer
tools = [
    {
        "name": "search_web",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "required": False},
        },
    },
    {
        "name": "read_file",
        "parameters": {
            "path": {"type": "string", "required": True},
        },
    },
]

scorer = ToolCallScorer(tools=tools, dataset="prod-shadow")

# Valid call
eval_valid = scorer.score(
    run_id="run-demo-001",
    tool_call={"name": "search_web", "arguments": {"query": "open source Zig"}},
)
print(f"\nValid tool call → verdict={eval_valid.verdict}, score={eval_valid.score}")
print("Notes:", eval_valid.notes)

# Hallucinated / invalid call
eval_invalid = scorer.score(
    run_id="run-demo-001",
    tool_call={"name": "search_web", "arguments": {"querY": "open source Zig"}},
)
print(f"\nBad tool call → verdict={eval_invalid.verdict}, score={eval_invalid.score}")
print("Notes:", eval_invalid.notes)

# Send the evals
client.ingest_eval(eval_valid)
client.ingest_eval(eval_invalid)

# 6. Query runs
summary = client.get_run("run-demo-001")
if summary:
    print(
        f"\nRun summary: spans={summary.span_count}, evals={summary.eval_count}, verdict={summary.verdict}"
    )
else:
    print("\n(nullwatch server not running — skipping run summary query)")
