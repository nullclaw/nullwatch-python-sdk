import json
import time
import urllib.request
from urllib.error import URLError

from nullwatch import Eval, NullwatchClient, Span
from nullwatch.scorers import RAGHallucinationScorer, ToolCallScorer

# config
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"
NULLWATCH_URL = "http://127.0.0.1:7710"
RUN_ID = f"live-demo-{int(time.time())}"

# helpers

def check_ollama() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call Ollama chat API, return full response dict."""
    payload: dict = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)

# RAG documents
CONTEXT_DOCS = [
    "Python was created by Guido van Rossum and first released in 1991. "
    "It is known for its clear syntax and readability. "
    "Python 3.0 was released in 2008 and broke backward compatibility with Python 2.",

    "The Zig programming language was created by Andrew Kelley. "
    "Zig 0.14.0 was released in March 2025. "
    "Zig emphasizes simplicity, performance, and explicit memory management.",
]

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the documentation for a given query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_version",
            "description": "Get the current version of a programming language",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Programming language name",
                        "enum": ["python", "zig", "rust", "go"],
                    },
                },
                "required": ["language"],
            },
        },
    },
]

# nullwatch-py scorer schemas (internal format)
TOOL_SCORER_TOOLS = [
    {
        "name": "search_docs",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "required": False, "minimum": 1, "maximum": 20},
        },
    },
    {
        "name": "get_version",
        "parameters": {
            "language": {
                "type": "string",
                "required": True,
                "enum": ["python", "zig", "rust", "go"],
            },
        },
    },
]


def main():
    # preflight checks
    print("🔍 Checking services...")

    ollama_ok = check_ollama()
    print(f"  Ollama:    {'✅ running' if ollama_ok else '❌ not running (start with: ollama serve)'}")

    client = NullwatchClient(base_url=NULLWATCH_URL, raise_on_error=False)
    nullwatch_ok = client.is_alive()
    print(f"  nullwatch: {'✅ running' if nullwatch_ok else '⚠️  not running (spans/evals will be skipped)'}")

    if not ollama_ok:
        print("\n❌ Ollama must be running. Start it with: ollama serve")
        print(f"   Then pull the model: ollama pull {MODEL}")
        return

    rag_scorer = RAGHallucinationScorer()
    tool_scorer = ToolCallScorer(tools=TOOL_SCORER_TOOLS)

    # PART 1: RAG hallucination detection
    section("PART 1: RAG Hallucination Detection")

    question = "When was Python first released and who created it?"
    context_str = "\n\n".join(CONTEXT_DOCS)

    rag_prompt = f"""Answer the following question based ONLY on the provided context.
Do not use any outside knowledge.

Context:
{context_str}

Question: {question}

Answer:"""

    print(f"\nQuestion: {question}")
    print(f"Context: {len(CONTEXT_DOCS)} documents")
    print("\n🤖 Calling model...")

    t0 = time.time()
    response = ollama_chat([{"role": "user", "content": rag_prompt}])
    elapsed = time.time() - t0

    answer = response["message"]["content"].strip()
    usage = response.get("prompt_eval_count", 0), response.get("eval_count", 0)

    print(f"\nModel answer ({elapsed:.1f}s):\n  {answer}")

    # Send span to nullwatch
    if nullwatch_ok:
        span = Span(
            run_id=RUN_ID,
            operation="llm.call",
            model=MODEL,
            source="live-demo",
            input_tokens=usage[0],
            output_tokens=usage[1],
        )
        span.finish()
        client.ingest_span(span)

    # Score hallucination
    print("\n🔬 Running hallucination detection (loading model on first run)...")
    try:
        eval_result = rag_scorer.score(
            run_id=RUN_ID,
            contexts=CONTEXT_DOCS,
            question=question,
            answer=answer,
        )

        print(f"\n  Verdict: {'✅ PASS' if eval_result.verdict == 'pass' else '❌ FAIL'}")
        print(f"  Score:   {eval_result.score:.3f} (1.0 = fully grounded)")
        print(f"  Notes:   {eval_result.notes}")

        if nullwatch_ok:
            client.ingest_eval(eval_result)
            print("  → Sent to nullwatch ✓")

    except ImportError:
        print("  ⚠️  lettucedetect not installed. Run: pip install 'nullwatch-py[rag]'")

    # PART 2: Tool call hallucination detection
    section("PART 2: Tool Call Hallucination Detection")

    tool_prompt = """You are a helpful assistant with access to tools.
The user wants to search for documentation about Zig.
Call the appropriate tool. Return ONLY the tool call, no explanation."""

    print("\n🤖 Asking model to make a tool call...")

    t0 = time.time()
    tool_response = ollama_chat(
        messages=[{"role": "user", "content": tool_prompt}],
        tools=TOOLS_SCHEMA,
    )
    elapsed = time.time() - t0

    msg = tool_response["message"]
    tool_calls_raw = msg.get("tool_calls", [])

    print(f"\nModel response ({elapsed:.1f}s):")

    if tool_calls_raw:
        print(f"  Tool calls: {len(tool_calls_raw)}")
        for tc in tool_calls_raw:
            fn = tc.get("function", tc)
            print(f"    → {fn.get('name')}({fn.get('arguments', {})})")

        # Score tool calls using nullwatch-py ToolCallScorer
        # ToolCallScorer accepts OpenAI format directly via normalize_tool_call()
        eval_tool = tool_scorer.score(
            run_id=RUN_ID,
            tool_calls=tool_calls_raw,
        )

        print(f"\n  Verdict: {'✅ PASS' if eval_tool.verdict == 'pass' else '❌ FAIL'}")
        print(f"  Score:   {eval_tool.score:.3f} ({eval_tool.meta['valid_calls']}/{eval_tool.meta['total_calls']} valid)")
        if eval_tool.meta["issues"]:
            print(f"  Issues:")
            for issue in eval_tool.meta["issues"]:
                print(f"    ⚠️  {issue}")
        else:
            print(f"  Notes:   {eval_tool.notes}")

        if nullwatch_ok:
            client.ingest_eval(eval_tool)
            span2 = Span(run_id=RUN_ID, operation="tool.call", source="live-demo")
            span2.finish()
            client.ingest_span(span2)
            print("  → Sent to nullwatch ✓")

    else:
        # Model didn't use tool calling — validate from text response
        print(f"  Content: {msg.get('content', '')[:200]}")
        print("\n  ⚠️  Model didn't return structured tool calls.")
        print("  This is itself a hallucination/failure — model should have called a tool.")

        eval_tool = Eval(
            run_id=RUN_ID,
            eval_key="tool_call_validity",
            scorer="schema-validator",
            score=0.0,
            verdict="fail",
            notes="Model returned text instead of a tool call when a tool call was expected.",
        )
        if nullwatch_ok:
            client.ingest_eval(eval_tool)
            print("  → Failure eval sent to nullwatch ✓")

    # PART 3: Run summary
    if nullwatch_ok:
        section("PART 3: Run Summary from nullwatch")
        time.sleep(0.2)
        summary = client.get_run(RUN_ID)
        if summary:
            print(f"\n  Run ID:  {RUN_ID}")
            print(f"  Spans:   {summary.span_count}")
            print(f"  Evals:   {summary.eval_count}")
            print(f"  Passed:  {summary.pass_count}")
            print(f"  Failed:  {summary.fail_count}")
            print(f"  Verdict: {'✅ ' if summary.verdict == 'pass' else '❌ '}{summary.verdict}")
        else:
            print("\n  ⚠️  Could not fetch run summary.")

    print(f"\n{'═' * 60}")
    print(f"  Done! Run ID: {RUN_ID}")
    print('═' * 60)


if __name__ == "__main__":
    main()
