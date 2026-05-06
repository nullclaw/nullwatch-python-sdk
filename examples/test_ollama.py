import json
import time
import urllib.request
from urllib.error import URLError

from nullwatch import Eval, NullwatchClient, Span
from nullwatch.scorers import RAGHallucinationScorer, ToolCallGroundingScorer, ToolCallScorer

# Config
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:0.6b"
NULLWATCH_URL = "http://127.0.0.1:7710"
RUN_ID = f"ollama-test-{int(time.time())}"

CONTEXT_DOCS = [
    "Python was created by Guido van Rossum and first released in 1991. "
    "It is known for its clear syntax and readability.",
    "The Zig programming language was created by Andrew Kelley. "
    "Zig 0.14.0 was released in March 2025.",
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
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

# Helpers

def sep(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check_ollama() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            model_ok = any(MODEL in m for m in models)
            if not model_ok:
                print(f"  ⚠️  Model '{MODEL}' not found. Available: {models}")
                print(f"  Run: ollama pull {MODEL}")
            return model_ok
    except Exception as e:
        print(f"  ❌ Ollama not reachable: {e}")
        return False


def ollama_chat(messages: list, tools: list | None = None, think: bool = False) -> dict:
    payload: dict = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    if not think:
        # Disable chain-of-thought for faster responses with qwen3
        payload.setdefault("options", {})["think"] = False
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


# Main
def main():
    print("=" * 60)
    print("  nullwatch-py × qwen3:0.6b — smoke test")
    print("=" * 60)

    # 1. Preflight
    sep("1. Checking services")
    ollama_ok = check_ollama()
    if not ollama_ok:
        print("\n❌ Ollama must be running with qwen3:0.6b. Aborting.")
        return

    print(f"  ✅ Ollama running, model '{MODEL}' available")

    client = NullwatchClient(base_url=NULLWATCH_URL, raise_on_error=False)
    nullwatch_ok = client.is_alive()
    print(f"  {'✅' if nullwatch_ok else '⚠️ '} nullwatch: {'running' if nullwatch_ok else 'not running (optional)'}")

    # 2. Real RAG hallucination scoring
    sep("2. RAG hallucination detection")

    user_query = "Tell me about the Zig programming language and its creator."
    context_str = "\n\n".join(CONTEXT_DOCS)

    rag_prompt = (
        f"Answer the following question based ONLY on the provided context.\n\n"
        f"Context:\n{context_str}\n\nQuestion: {user_query}\n\nAnswer:"
    )

    print(f"\n  Question: {user_query}")
    print(f"  Calling {MODEL}...")
    t0 = time.time()
    resp = ollama_chat([{"role": "user", "content": rag_prompt}])
    answer = resp["message"]["content"].strip()
    # Strip <think> blocks if model has chain-of-thought
    if "<think>" in answer:
        answer = answer.split("</think>")[-1].strip()
    print(f"  Answer ({time.time()-t0:.1f}s): {answer[:200]}...")

    rag_scorer = RAGHallucinationScorer()
    eval_rag = rag_scorer.score(
        run_id=RUN_ID,
        contexts=CONTEXT_DOCS,
        question=user_query,
        answer=answer,
    )
    print(f"\n  RAG hallucination check:")
    print(f"    Verdict: {'✅ PASS' if eval_rag.verdict == 'pass' else '❌ FAIL'}")
    print(f"    Score:   {eval_rag.score:.3f}")
    print(f"    Notes:   {eval_rag.notes}")

    synthetic_hallucinated_answer = (
        "Zig was created by Brendan Eich and its first stable release was in 2023."
    )
    eval_rag_fail = rag_scorer.score(
        run_id=RUN_ID,
        contexts=CONTEXT_DOCS,
        question=user_query,
        answer=synthetic_hallucinated_answer,
    )
    print(f"\n  RAG hallucination check (synthetic bad answer):")
    print(f"    Verdict: {'✅ PASS' if eval_rag_fail.verdict == 'pass' else '❌ FAIL'}")
    print(f"    Score:   {eval_rag_fail.score:.3f}")
    print(f"    Notes:   {eval_rag_fail.notes}")

    if nullwatch_ok:
        client.ingest_eval(eval_rag)
        client.ingest_eval(eval_rag_fail)

    # 3. Tool-call grounding check (keyword backend, zero-deps)
    sep("3. Tool call grounding (keyword backend)")

    grounding_scorer = ToolCallGroundingScorer(context=CONTEXT_DOCS)

    # Simulate: model decided to call search_docs based on the answer
    simulated_tool_call = {
        "name": "search_docs",
        "arguments": {"query": "Zig programming language Andrew Kelley"},
    }
    eval_grounding = grounding_scorer.score(run_id=RUN_ID, tool_call=simulated_tool_call)
    print(f"\n  Grounding check (keyword):")
    print(f"    Verdict: {'✅ PASS' if eval_grounding.verdict == 'pass' else '❌ FAIL'}")
    print(f"    Score:   {eval_grounding.score:.3f}")
    print(f"    Notes:   {eval_grounding.notes}")

    # Simulate hallucinated tool call for contrast
    hallucinated_tool_call = {
        "name": "search_docs",
        "arguments": {"query": "Kubernetes Docker AWS Terraform"},
    }
    eval_hallucinated = grounding_scorer.score(run_id=RUN_ID, tool_call=hallucinated_tool_call)
    print(f"\n  Grounding check (hallucinated query):")
    print(f"    Verdict: {'✅ PASS' if eval_hallucinated.verdict == 'pass' else '❌ FAIL'}")
    print(f"    Score:   {eval_hallucinated.score:.3f}")
    print(f"    Notes:   {eval_hallucinated.notes}")

    if nullwatch_ok:
        client.ingest_eval(eval_grounding)
        client.ingest_eval(eval_hallucinated)

    # 4. Actual tool calling by the model
    sep("4. Actual tool call from model + schema validation")

    tool_prompt = (
        "You are a helpful assistant. The user wants to find documentation about "
        "the Zig programming language and Andrew Kelley. Use the search_docs tool."
    )
    print(f"\n  Asking model to make a tool call...")
    t0 = time.time()
    tool_resp = ollama_chat(
        messages=[{"role": "user", "content": tool_prompt}],
        tools=TOOLS_SCHEMA,
    )
    elapsed = time.time() - t0
    msg = tool_resp["message"]
    tool_calls_raw = msg.get("tool_calls", [])

    schema_scorer = ToolCallScorer(tools=TOOLS_SCHEMA)

    if tool_calls_raw:
        print(f"  Model returned {len(tool_calls_raw)} tool call(s) in {elapsed:.1f}s:")
        for tc in tool_calls_raw:
            fn = tc.get("function", tc)
            print(f"    → {fn.get('name')}({fn.get('arguments', {})})")

        # Schema validation
        eval_schema = schema_scorer.score(run_id=RUN_ID, tool_calls=tool_calls_raw)
        print(f"\n  Schema validation (ToolCallScorer):")
        print(f"    Verdict: {'✅ PASS' if eval_schema.verdict == 'pass' else '❌ FAIL'}")
        print(f"    Score:   {eval_schema.score:.3f}")
        print(f"    Notes:   {eval_schema.notes}")

        # Semantic grounding with LLM backend
        print(f"\n  Semantic grounding (ToolCallGroundingScorer, backend=llm, model={MODEL}):")
        llm_grounding_scorer = ToolCallGroundingScorer(
            context=CONTEXT_DOCS,
            backend="llm",
            llm_url=f"{OLLAMA_URL}/v1",
            llm_model=MODEL,
            fail_on_llm_error=False,
        )
        eval_llm_grounding = llm_grounding_scorer.score(run_id=RUN_ID, tool_calls=tool_calls_raw)
        print(f"    Verdict: {'✅ PASS' if eval_llm_grounding.verdict == 'pass' else '❌ FAIL'}")
        print(f"    Score:   {eval_llm_grounding.score:.3f}")
        print(f"    Notes:   {eval_llm_grounding.notes}")

        synthetic_bad_call = {
            "name": "search_docs",
            "arguments": {"query": "Kubernetes Docker AWS Terraform", "max_results": 99},
        }
        eval_llm_synthetic_bad = llm_grounding_scorer.score(
            run_id=RUN_ID,
            tool_call=synthetic_bad_call,
        )
        print(f"\n  LLM grounding sanity check (synthetic bad call):")
        print(f"    Verdict: {'✅ PASS' if eval_llm_synthetic_bad.verdict == 'pass' else '❌ FAIL'}")
        print(f"    Score:   {eval_llm_synthetic_bad.score:.3f}")
        print(f"    Notes:   {eval_llm_synthetic_bad.notes}")
        if eval_llm_synthetic_bad.verdict == "pass":
            print("    Warning: tiny local judge models may miss obvious tool-call hallucinations.")

        if nullwatch_ok:
            client.ingest_eval(eval_schema)
            client.ingest_eval(eval_llm_grounding)
            client.ingest_eval(eval_llm_synthetic_bad)
            span = Span(run_id=RUN_ID, operation="tool.call", source="ollama-test", model=MODEL)
            span.finish()
            client.ingest_span(span)

    else:
        content = msg.get("content", "")
        print(f"  ⚠️  Model returned text instead of tool call ({elapsed:.1f}s):")
        print(f"    {content[:200]}")
        print(f"\n  This is itself a failure — model should have called search_docs.")

        eval_no_call = Eval(
            run_id=RUN_ID,
            eval_key="tool_call_validity",
            scorer="schema-validator",
            score=0.0,
            verdict="fail",
            notes=f"Model returned text instead of a tool call. Content: {content[:100]}",
        )
        if nullwatch_ok:
            client.ingest_eval(eval_no_call)

    # 5. Summary
    if nullwatch_ok:
        sep("5. Run summary from nullwatch")
        time.sleep(0.2)
        summary = client.get_run(RUN_ID)
        if summary:
            print(f"  Run ID:  {RUN_ID}")
            print(f"  Spans:   {summary.span_count}")
            print(f"  Evals:   {summary.eval_count}")
            print(f"  Passed:  {summary.pass_count}")
            print(f"  Failed:  {summary.fail_count}")
            print(f"  Verdict: {summary.verdict}")

    print(f"\n{'=' * 60}")
    print(f"  Done! Run ID: {RUN_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
