"""Tests for NullwatchClient (uses mock HTTP server)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nullwatch import Eval, NullwatchClient, Span

# Minimal mock nullwatch server
_received: list = []


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence output

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path.startswith("/v1/runs/"):
            run_id = self.path.split("/")[-1]
            self._respond(
                200,
                {
                    "run_id": run_id,
                    "span_count": 2,
                    "eval_count": 1,
                    "pass_count": 1,
                    "fail_count": 0,
                    "verdict": "pass",
                },
            )
        elif self.path.startswith("/v1/runs"):
            self._respond(200, [])
        elif self.path.startswith("/v1/spans"):
            self._respond(200, [])
        elif self.path.startswith("/v1/evals"):
            self._respond(200, [])
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _received.append((self.path, body))
        self._respond(201, {"ok": True})

    def _respond(self, status: int, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def mock_server():
    server = HTTPServer(("127.0.0.1", 17710), _MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:17710"
    server.shutdown()


@pytest.fixture(autouse=True)
def clear_received():
    _received.clear()


# Tests
class TestNullwatchClient:
    def test_is_alive(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        assert client.is_alive() is True

    def test_ingest_span(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        s = Span(run_id="run-1", operation="llm.call", model="gpt-4o")
        s.finish()
        client.ingest_span(s)
        assert len(_received) == 1
        path, body = _received[0]
        assert path == "/v1/spans"
        assert body["run_id"] == "run-1"
        assert body["operation"] == "llm.call"
        assert body["model"] == "gpt-4o"

    def test_ingest_span_auto_finish(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        s = Span(run_id="run-1", operation="tool.call")
        # Don't call finish() — client should do it
        client.ingest_span(s)
        _, body = _received[0]
        assert "ended_at_ms" in body

    def test_span_context_manager(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        with client.span("run-2", "tool.call", tool_name="bash") as s:
            s.status = "ok"
        assert len(_received) == 1
        _, body = _received[0]
        assert body["tool_name"] == "bash"
        assert body["status"] == "ok"
        assert "duration_ms" in body

    def test_span_context_manager_error(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        with pytest.raises(ValueError):
            with client.span("run-2", "tool.call"):
                raise ValueError("boom")
        _, body = _received[0]
        assert body["status"] == "error"

    def test_ingest_eval(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        e = Eval(run_id="run-1", eval_key="rag_hallucination", score=0.95, verdict="pass")
        client.ingest_eval(e)
        path, body = _received[0]
        assert path == "/v1/evals"
        assert body["eval_key"] == "rag_hallucination"
        assert body["score"] == 0.95

    def test_ingest_spans_bulk(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        spans = [
            Span(run_id="run-1", operation="llm.call"),
            Span(run_id="run-1", operation="tool.call"),
        ]
        client.ingest_spans(spans)
        path, body = _received[0]
        assert path == "/v1/spans/bulk"
        assert len(body["items"]) == 2

    def test_get_run(self, mock_server):
        client = NullwatchClient(base_url=mock_server)
        summary = client.get_run("run-42")
        assert summary is not None
        assert summary.run_id == "run-42"
        assert summary.span_count == 2
        assert summary.verdict == "pass"

    def test_default_source_applied(self, mock_server):
        client = NullwatchClient(base_url=mock_server, default_source="my-app")
        s = Span(run_id="run-1", operation="llm.call")
        client.ingest_span(s)
        _, body = _received[0]
        assert body["source"] == "my-app"

    def test_raise_on_error_false(self, mock_server):
        # Use mock server with a bad path to trigger a 404 instead of connection error
        client = NullwatchClient(base_url=mock_server, raise_on_error=False)
        # Direct non-existent endpoint
        result = client._get("/v1/nonexistent")
        assert result is None  # 404 with raise_on_error=False returns None
