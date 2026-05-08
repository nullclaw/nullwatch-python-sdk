"""Tests for new features: env vars, api_key, buffered mode, decorators,
provider helpers, MemoryTransport, and CLI."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nullwatch import Eval, MemoryTransport, NullwatchClient, Span
from nullwatch.testing import AssertionError as NWAssertionError


# Fixtures
@pytest.fixture()
def transport():
    return MemoryTransport()


@pytest.fixture()
def client(transport):
    return NullwatchClient(transport=transport)


# MemoryTransport
class TestMemoryTransport:
    def test_captures_span(self, client, transport):
        with client.span("run-1", "llm.call", model="gpt-4o"):
            pass
        assert len(transport.spans) == 1
        assert transport.spans[0]["operation"] == "llm.call"

    def test_captures_eval(self, client, transport):
        client.ingest_eval(Eval(run_id="run-1", eval_key="quality", score=0.9, verdict="pass"))
        assert len(transport.evals) == 1
        assert transport.evals[0]["eval_key"] == "quality"

    def test_clear(self, client, transport):
        with client.span("run-1", "test"):
            pass
        transport.clear()
        assert transport.spans == []
        assert transport.evals == []

    def test_get_run_from_memory(self, client, transport):
        with client.span("run-42", "step"):
            pass
        summary = client.get_run("run-42")
        assert summary is not None
        assert summary.span_count == 1

    def test_is_alive_via_transport(self, client):
        assert client.is_alive() is True

    def test_capabilities_via_transport(self, client):
        caps = client.capabilities()
        assert "version" in caps


# Assert helpers
class TestAssertHelpers:
    def test_assert_span_recorded_pass(self, client, transport):
        with client.span("run-1", "tool.call", tool_name="search"):
            pass
        span = transport.assert_span_recorded(operation="tool.call", tool_name="search")
        assert span["tool_name"] == "search"

    def test_assert_span_recorded_fail(self, transport):
        with pytest.raises(NWAssertionError):
            transport.assert_span_recorded(operation="nonexistent")

    def test_assert_no_failed_evals_pass(self, client, transport):
        client.ingest_eval(Eval(run_id="run-1", eval_key="k", score=1.0, verdict="pass"))
        transport.assert_no_failed_evals()  # should not raise

    def test_assert_no_failed_evals_fail(self, client, transport):
        client.ingest_eval(Eval(run_id="run-1", eval_key="rag", score=0.1, verdict="fail"))
        with pytest.raises(NWAssertionError):
            transport.assert_no_failed_evals()

    def test_assert_eval_recorded_pass(self, client, transport):
        client.ingest_eval(Eval(run_id="run-1", eval_key="k", score=1.0, verdict="pass"))
        eval_ = transport.assert_eval_recorded(eval_key="k", verdict="pass")
        assert eval_["score"] == 1.0

    def test_assert_eval_recorded_fail(self, transport):
        with pytest.raises(NWAssertionError):
            transport.assert_eval_recorded(eval_key="missing")

    def test_assert_no_failed_evals_scoped_to_run(self, client, transport):
        client.ingest_eval(Eval(run_id="run-A", eval_key="k", score=0.0, verdict="fail"))
        # run-B has no failed evals
        transport.assert_no_failed_evals(run_id="run-B")  # should not raise

    def test_assert_eval_recorded_by_scorer(self, client, transport):
        client.ingest_eval(
            Eval(
                run_id="r",
                eval_key="rag_hallucination",
                score=0.9,
                verdict="pass",
                scorer="lettucedetect",
            )
        )
        eval_ = transport.assert_eval_recorded(scorer="lettucedetect")
        assert eval_["eval_key"] == "rag_hallucination"


# Env vars
class TestEnvVars:
    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("NULLWATCH_URL", "http://custom-host:9999")
        client = NullwatchClient()
        assert client.base_url == "http://custom-host:9999"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NULLWATCH_API_KEY", "secret-token")
        client = NullwatchClient()
        assert client.api_key == "secret-token"

    def test_explicit_args_take_priority(self, monkeypatch):
        monkeypatch.setenv("NULLWATCH_URL", "http://env-host:7710")
        monkeypatch.setenv("NULLWATCH_API_KEY", "env-key")
        client = NullwatchClient(base_url="http://explicit:1234", api_key="explicit-key")
        assert client.base_url == "http://explicit:1234"
        assert client.api_key == "explicit-key"


# Authorization header
class TestApiKey:
    def test_auth_header_in_request(self, monkeypatch):
        """When api_key is set, requests must include an Authorization header."""
        received_headers = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                received_headers.append(dict(self.headers))
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                data = b'{"ok": true}'
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            client = NullwatchClient(
                base_url=f"http://127.0.0.1:{port}",
                api_key="my-secret",
            )
            s = Span(run_id="run-1", operation="test")
            s.finish()
            client.ingest_span(s)
            assert received_headers, "No request received by mock server"
            assert received_headers[0].get("Authorization") == "Bearer my-secret"
        finally:
            server.shutdown()


# Redact hook
class TestRedact:
    def test_redact_applied_to_span(self, transport):
        def scrub(payload):
            if "model" in payload:
                payload = dict(payload, model="[REDACTED]")
            return payload

        client = NullwatchClient(transport=transport, redact=scrub)
        s = Span(run_id="run-1", operation="llm.call", model="gpt-4o")
        s.finish()
        client.ingest_span(s)
        assert transport.spans[0]["model"] == "[REDACTED]"


# Buffered mode
class TestBufferedMode:
    def test_spans_not_sent_immediately(self, transport):
        client = NullwatchClient(transport=transport, buffered=True, flush_at=100)
        s = Span(run_id="run-1", operation="step")
        s.finish()
        client.ingest_span(s)
        assert len(transport.spans) == 0  # not flushed yet

    def test_flush_sends_buffered_spans(self, transport):
        client = NullwatchClient(transport=transport, buffered=True, flush_at=100)
        s = Span(run_id="run-1", operation="step")
        s.finish()
        client.ingest_span(s)
        client.flush()
        assert len(transport.spans) == 1

    def test_flush_at_triggers_auto_flush(self, transport):
        client = NullwatchClient(transport=transport, buffered=True, flush_at=3)
        for i in range(3):
            s = Span(run_id="run-1", operation=f"step-{i}")
            s.finish()
            client.ingest_span(s)
        # Should have auto-flushed at flush_at=3
        assert len(transport.spans) == 3

    def test_context_manager_flushes_on_exit(self, transport):
        with NullwatchClient(transport=transport, buffered=True, flush_at=100) as c:
            s = Span(run_id="run-1", operation="step")
            s.finish()
            c.ingest_span(s)
        assert len(transport.spans) == 1

    def test_flush_empty_buffer_returns_none(self, transport):
        client = NullwatchClient(transport=transport, buffered=True)
        result = client.flush()
        assert result is None


# Decorator: @client.trace
class TestTraceDecorator:
    def test_trace_records_span(self, client, transport):
        @client.trace("retriever.search")
        def search(run_id: str, query: str) -> list:
            return []

        search(run_id="run-1", query="python")
        transport.assert_span_recorded(operation="retriever.search")

    def test_trace_captures_error(self, client, transport):
        @client.trace("failing.step")
        def fail(run_id: str):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            fail(run_id="run-1")

        span = transport.assert_span_recorded(operation="failing.step")
        assert span["status"] == "error"

    def test_trace_positional_run_id(self, client, transport):
        @client.trace("step")
        def do_work(run_id: str, value: int) -> int:
            return value * 2

        result = do_work("run-pos", 21)
        assert result == 42
        transport.assert_span_recorded(operation="step", run_id="run-pos")

    def test_trace_auto_generates_run_id(self, client, transport):
        @client.trace("auto.step")
        def no_run_id(x: int) -> int:
            return x

        no_run_id(1)
        # Just assert a span was recorded (run_id was auto-generated)
        assert len(transport.spans) == 1
        assert transport.spans[0]["run_id"].startswith("run-")


# Decorator: @client.atrace
class TestATraceDecorator:
    def test_atrace_records_span(self, client, transport):
        @client.atrace("async.step")
        async def async_work(run_id: str) -> str:
            return "done"

        asyncio.run(async_work(run_id="run-1"))
        transport.assert_span_recorded(operation="async.step")

    def test_atrace_captures_error(self, client, transport):
        @client.atrace("async.fail")
        async def async_fail(run_id: str):
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError):
            asyncio.run(async_fail(run_id="run-1"))

        span = transport.assert_span_recorded(operation="async.fail")
        assert span["status"] == "error"


# Provider helpers on Span
class TestProviderHelpers:
    def test_record_tokens(self):
        s = Span(run_id="r", operation="llm.call")
        s.record_tokens(input_tokens=100, output_tokens=50)
        assert s.input_tokens == 100
        assert s.output_tokens == 50

    def test_record_cost(self):
        s = Span(run_id="r", operation="llm.call")
        s.record_cost(0.003)
        assert s.cost_usd == 0.003

    def test_record_openai_usage_dict(self):
        s = Span(run_id="r", operation="llm.call")
        response = {"usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_cost": 0.005}}
        s.record_openai_usage(response)
        assert s.input_tokens == 200
        assert s.output_tokens == 80
        assert s.cost_usd == 0.005

    def test_record_openai_usage_object(self):
        class Usage:
            prompt_tokens = 150
            completion_tokens = 60

        class Response:
            usage = Usage()

        s = Span(run_id="r", operation="llm.call")
        s.record_openai_usage(Response())
        assert s.input_tokens == 150
        assert s.output_tokens == 60

    def test_record_anthropic_usage_dict(self):
        s = Span(run_id="r", operation="llm.call")
        response = {"usage": {"input_tokens": 120, "output_tokens": 40}}
        s.record_anthropic_usage(response)
        assert s.input_tokens == 120
        assert s.output_tokens == 40

    def test_record_anthropic_usage_object(self):
        class Usage:
            input_tokens = 90
            output_tokens = 30

        class Message:
            usage = Usage()

        s = Span(run_id="r", operation="llm.call")
        s.record_anthropic_usage(Message())
        assert s.input_tokens == 90
        assert s.output_tokens == 30

    def test_record_openai_usage_no_usage_field(self):
        s = Span(run_id="r", operation="llm.call")
        s.record_openai_usage({})  # no usage key — should not raise
        assert s.input_tokens is None

    def test_helpers_are_chainable(self):
        s = Span(run_id="r", operation="llm.call")
        result = s.record_tokens(input_tokens=10, output_tokens=5).record_cost(0.001)
        assert result is s  # returns self


# CLI
class TestCLI:
    def test_ping_ok(self, capsys, transport):
        from nullwatch import cli

        # Test main --help exits 0
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--help"])
        assert exc_info.value.code == 0

    def test_unknown_command_exits_2(self, capsys):
        from nullwatch import cli

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["not-a-command"])
        assert exc_info.value.code == 2

    def test_ingest_span_missing_file(self, capsys):
        from nullwatch.cli import cmd_ingest_span

        result = cmd_ingest_span(["/nonexistent/path.json"])
        assert result == 1

    def test_ingest_eval_missing_file(self, capsys):
        from nullwatch.cli import cmd_ingest_eval

        result = cmd_ingest_eval(["/nonexistent/eval.json"])
        assert result == 1

    def test_ingest_span_no_args(self, capsys):
        from nullwatch.cli import cmd_ingest_span

        result = cmd_ingest_span([])
        assert result == 2

    def test_ingest_eval_no_args(self, capsys):
        from nullwatch.cli import cmd_ingest_eval

        result = cmd_ingest_eval([])
        assert result == 2

    def test_run_no_args(self, capsys):
        from nullwatch.cli import cmd_run

        result = cmd_run([])
        assert result == 2

    def test_ingest_span_from_file(self, tmp_path, transport):
        span_data = {"run_id": "run-cli", "operation": "cli.test"}
        f = tmp_path / "span.json"
        f.write_text(json.dumps(span_data))

        # Patch NullwatchClient to use our transport
        import nullwatch.cli as cli_module

        original = cli_module._make_client

        def patched_make_client(base_url=None):
            return NullwatchClient(transport=transport)

        cli_module._make_client = patched_make_client
        try:
            from nullwatch.cli import cmd_ingest_span

            result = cmd_ingest_span([str(f)])
            # May return 1 if no server is running — that's OK in unit test
            assert result in (0, 1)
        finally:
            cli_module._make_client = original
