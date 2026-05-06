import time

import pytest

from nullwatch import Eval, NullwatchClient, Span

BASE_URL = "http://127.0.0.1:7710"


@pytest.fixture(scope="module")
def client():
    c = NullwatchClient(base_url=BASE_URL, raise_on_error=True)
    if not c.is_alive():
        pytest.skip("nullwatch is not running at 127.0.0.1:7710 — start it with: zig build run -- serve")
    return c


@pytest.fixture
def run_id():
    """Unique run_id per test to avoid cross-test contamination."""
    return f"integ-{int(time.time() * 1000)}"


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        h = client.health()
        assert h.get("status") == "ok"

    def test_health_has_version(self, client):
        h = client.health()
        assert "version" in h

    def test_health_has_counts(self, client):
        h = client.health()
        counts = h.get("counts", {})
        assert "runs" in counts
        assert "spans" in counts
        assert "evals" in counts


class TestSpanIngestion:
    def test_ingest_single_span(self, client, run_id):
        s = Span(run_id=run_id, operation="llm.call", model="gpt-4o")
        s.finish()
        result = client.ingest_span(s)
        assert result is not None

    def test_ingest_span_context_manager(self, client, run_id):
        with client.span(run_id, "tool.call", tool_name="bash") as s:
            time.sleep(0.01)  # simulate work
        assert s.duration_ms is not None
        assert s.duration_ms >= 0
        assert s.status == "ok"

    def test_ingest_span_error_status(self, client, run_id):
        with pytest.raises(RuntimeError):
            with client.span(run_id, "tool.call") as s:
                raise RuntimeError("tool failed")
        assert s.status == "error"

    def test_ingest_span_bulk(self, client, run_id):
        spans = [
            Span(run_id=run_id, operation="llm.call", model="gpt-4o"),
            Span(run_id=run_id, operation="tool.call", tool_name="read_file"),
        ]
        result = client.ingest_spans(spans)
        assert result is not None


class TestSpanListing:
    def test_list_spans_returns_list(self, client, run_id):
        # Ingest first
        client.ingest_span(Span(run_id=run_id, operation="llm.call").finish())
        time.sleep(0.05)

        spans = client.list_spans(run_id=run_id)
        # BUG CHECK: nullwatch returns {"items": [...]}, not [...]
        # If this fails with an empty list, the client isn't unwrapping correctly
        assert isinstance(spans, list), f"Expected list, got {type(spans)}: {spans}"
        assert len(spans) >= 1

    def test_list_spans_filter_by_status(self, client, run_id):
        client.ingest_span(Span(run_id=run_id, operation="ok.call", status="ok").finish())
        time.sleep(0.05)

        spans = client.list_spans(run_id=run_id, status="ok")
        assert isinstance(spans, list)
        for s in spans:
            assert s.get("status") == "ok"

    def test_list_spans_limit(self, client, run_id):
        for i in range(5):
            client.ingest_span(Span(run_id=run_id, operation=f"call.{i}").finish())
        time.sleep(0.05)

        spans = client.list_spans(run_id=run_id, limit=2)
        assert isinstance(spans, list)
        assert len(spans) <= 2


class TestEvalIngestion:
    def test_ingest_eval(self, client, run_id):
        e = Eval(
            run_id=run_id,
            eval_key="rag_hallucination",
            score=0.95,
            verdict="pass",
            notes="No hallucinations detected",
        )
        result = client.ingest_eval(e)
        assert result is not None

    def test_ingest_eval_fail(self, client, run_id):
        e = Eval(
            run_id=run_id,
            eval_key="tool_call_validity",
            score=0.0,
            verdict="fail",
            notes="Unknown tool 'fake_tool'",
        )
        result = client.ingest_eval(e)
        assert result is not None


class TestEvalListing:
    def test_list_evals_returns_list(self, client, run_id):
        client.ingest_eval(Eval(run_id=run_id, eval_key="test", score=1.0, verdict="pass"))
        time.sleep(0.05)

        evals = client.list_evals(run_id=run_id)
        # BUG CHECK: nullwatch returns {"items": [...]}, not [...]
        assert isinstance(evals, list), f"Expected list, got {type(evals)}: {evals}"
        assert len(evals) >= 1

    def test_list_evals_filter_by_verdict(self, client, run_id):
        client.ingest_eval(Eval(run_id=run_id, eval_key="test", score=1.0, verdict="pass"))
        client.ingest_eval(Eval(run_id=run_id, eval_key="test2", score=0.0, verdict="fail"))
        time.sleep(0.05)

        fails = client.list_evals(run_id=run_id, verdict="fail")
        assert isinstance(fails, list)
        for e in fails:
            assert e.get("verdict") == "fail"

    def test_list_evals_filter_by_eval_key(self, client, run_id):
        client.ingest_eval(Eval(run_id=run_id, eval_key="rag_hallucination", score=1.0, verdict="pass"))
        time.sleep(0.05)

        evals = client.list_evals(run_id=run_id, eval_key="rag_hallucination")
        assert isinstance(evals, list)
        for e in evals:
            assert e.get("eval_key") == "rag_hallucination"


class TestRunSummary:
    def test_get_run_after_span_and_eval(self, client, run_id):
        # Ingest a span and eval
        client.ingest_span(Span(run_id=run_id, operation="llm.call").finish())
        client.ingest_eval(Eval(run_id=run_id, eval_key="test", score=1.0, verdict="pass"))
        time.sleep(0.05)

        summary = client.get_run(run_id)
        assert summary is not None
        assert summary.run_id == run_id
        assert summary.span_count >= 1
        assert summary.eval_count >= 1

    def test_get_nonexistent_run_returns_none(self, client):
        summary = client.get_run("nonexistent-run-xyz-12345")
        # Should return None gracefully, not raise
        assert summary is None

    def test_list_runs_returns_list(self, client, run_id):
        client.ingest_span(Span(run_id=run_id, operation="llm.call").finish())
        time.sleep(0.05)

        runs = client.list_runs()
        # BUG CHECK: nullwatch returns {"items": [...]}, not [...]
        assert isinstance(runs, list), f"Expected list, got {type(runs)}: {runs}"


class TestRoundTrip:
    def test_full_agent_run_roundtrip(self, client, run_id):
        """
        Simulates a full agent turn:
        span(llm.call) → span(tool.call) → eval(rag_hallucination) → get_run summary
        """
        # Step 1: LLM call span
        with client.span(run_id, "llm.call", model="gpt-4o") as s:
            s.input_tokens = 100
            s.output_tokens = 50
            s.cost_usd = 0.002

        # Step 2: Tool call span
        with client.span(run_id, "tool.call", tool_name="search_web") as s:
            pass

        # Step 3: Eval
        client.ingest_eval(Eval(
            run_id=run_id,
            eval_key="rag_hallucination",
            scorer="lettucedect-large-modernbert-en-v1",
            score=0.92,
            verdict="pass",
            notes="No hallucinations detected",
        ))

        time.sleep(0.05)

        # Step 4: Verify via summary
        summary = client.get_run(run_id)
        assert summary is not None
        assert summary.span_count == 2
        assert summary.eval_count == 1
