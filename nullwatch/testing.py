"""Testing utilities for nullwatch-py.

These helpers let you assert telemetry behaviour without running a real
``nullwatch`` server.

Example::

    from nullwatch.testing import MemoryTransport

    transport = MemoryTransport()
    client = NullwatchClient(transport=transport)

    with client.span("run-123", "tool.execute", tool_name="search"):
        pass

    assert len(transport.spans) == 1
    transport.assert_span_recorded(operation="tool.execute", tool_name="search")
    transport.assert_no_failed_evals()
"""

from __future__ import annotations

from typing import Any, List, Optional

from .models import Eval, RunSummary, Span


class AssertionError(Exception):  # noqa: A001 — intentionally shadows builtins for clarity
    """Raised when a transport assertion fails."""


class MemoryTransport:
    """In-memory replacement for a real nullwatch server.

    Pass an instance to :class:`~nullwatch.NullwatchClient` via the
    ``transport`` keyword argument.  All spans and evals are captured in
    ``transport.spans`` and ``transport.evals`` respectively.

    The transport is intentionally *not* thread-safe; for concurrent tests use
    one transport per thread or protect access with a lock.
    """

    def __init__(self) -> None:
        self.spans: List[dict] = []
        self.evals: List[dict] = []
        self._runs: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Mimic the HTTP methods called by NullwatchClient._request
    # ------------------------------------------------------------------

    def post(self, path: str, body: dict) -> dict:
        if path == "/v1/spans":
            self.spans.append(body)
            return {"ok": True}
        if path == "/v1/spans/bulk":
            for item in body.get("items", []):
                self.spans.append(item)
            return {"ok": True}
        if path == "/v1/evals":
            self.evals.append(body)
            return {"ok": True}
        return {}

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/capabilities":
            return {"version": "memory-transport"}
        if path.startswith("/v1/runs/"):
            run_id = path.split("/")[-1]
            if run_id in self._runs:
                return self._runs[run_id]
            span_count = sum(1 for s in self.spans if s.get("run_id") == run_id)
            eval_count = sum(1 for e in self.evals if e.get("run_id") == run_id)
            if span_count == 0 and eval_count == 0:
                return None
            return {
                "run_id": run_id,
                "span_count": span_count,
                "eval_count": eval_count,
                "verdict": "pass",
            }
        if path.startswith("/v1/runs"):
            return {"items": list(self._runs.values())}
        if path.startswith("/v1/spans"):
            run_id = (params or {}).get("run_id")
            items = [s for s in self.spans if run_id is None or s.get("run_id") == run_id]
            return {"items": items}
        if path.startswith("/v1/evals"):
            run_id = (params or {}).get("run_id")
            items = [e for e in self.evals if run_id is None or e.get("run_id") == run_id]
            return {"items": items}
        return {}

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all captured spans, evals, and run state."""
        self.spans.clear()
        self.evals.clear()
        self._runs.clear()

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

    def assert_no_failed_evals(self, *, run_id: Optional[str] = None) -> None:
        """Assert that no captured evals have ``verdict == "fail"``.

        Args:
            run_id: Scope the assertion to a specific run.  When *None* all
                    captured evals are checked.

        Raises:
            AssertionError: If any matching eval has a failing verdict.
        """
        evals = self.evals
        if run_id is not None:
            evals = [e for e in evals if e.get("run_id") == run_id]
        failed = [e for e in evals if e.get("verdict") == "fail"]
        if failed:
            notes = "; ".join(
                f"{e.get('eval_key', '?')} ({e.get('notes', '')})" for e in failed
            )
            raise AssertionError(f"{len(failed)} failed eval(s): {notes}")

    def assert_span_recorded(
        self,
        *,
        operation: Optional[str] = None,
        run_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """Assert that at least one span matching the given filters was recorded.

        Returns the first matching span dict.

        Raises:
            AssertionError: If no matching span is found.
        """
        filters = {
            k: v
            for k, v in {
                "operation": operation,
                "run_id": run_id,
                "tool_name": tool_name,
                "model": model,
                "status": status,
            }.items()
            if v is not None
        }
        for span in self.spans:
            if all(span.get(k) == v for k, v in filters.items()):
                return span
        raise AssertionError(
            f"No span matching {filters} found.  "
            f"Recorded spans: {[s.get('operation') for s in self.spans]}"
        )

    def assert_eval_recorded(
        self,
        *,
        eval_key: Optional[str] = None,
        run_id: Optional[str] = None,
        verdict: Optional[str] = None,
        scorer: Optional[str] = None,
    ) -> dict:
        """Assert that at least one eval matching the given filters was recorded.

        Returns the first matching eval dict.

        Raises:
            AssertionError: If no matching eval is found.
        """
        filters = {
            k: v
            for k, v in {
                "eval_key": eval_key,
                "run_id": run_id,
                "verdict": verdict,
                "scorer": scorer,
            }.items()
            if v is not None
        }
        for eval_ in self.evals:
            if all(eval_.get(k) == v for k, v in filters.items()):
                return eval_
        raise AssertionError(
            f"No eval matching {filters} found.  "
            f"Recorded evals: {[e.get('eval_key') for e in self.evals]}"
        )
