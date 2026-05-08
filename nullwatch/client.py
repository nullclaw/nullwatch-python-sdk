import asyncio
import contextlib
import functools
import inspect
import json
import os
import threading
from typing import Any, Callable, Generator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Eval, RunSummary, Span


class NullwatchError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"nullwatch API error {status}: {body}")


class NullwatchClient:
    """Python client for the nullwatch observability service.

    Args:
        base_url:       Service URL. Defaults to NULLWATCH_URL env var or
                        ``http://127.0.0.1:7710``.
        api_key:        Optional bearer token. Defaults to NULLWATCH_API_KEY
                        env var.
        timeout:        HTTP request timeout in seconds.
        raise_on_error: Raise :class:`NullwatchError` on non-2xx responses.
        default_source: ``source`` field written to every span that still has
                        the placeholder ``"python-sdk"`` value.
        buffered:       When *True*, spans are queued in memory and flushed in
                        bulk via ``/v1/spans/bulk``.  Evals are always sent
                        immediately.
        flush_at:       Flush the buffer automatically after this many spans.
        redact:         Optional callable ``(payload: dict) -> dict`` that runs
                        before every HTTP request body is serialised.  Use it to
                        scrub secrets or sensitive fields.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout: int = 10,
        raise_on_error: bool = True,
        default_source: str = "python-sdk",
        buffered: bool = False,
        flush_at: int = 100,
        redact: Optional[Callable[[dict], dict]] = None,
        transport: Any = None,
    ):
        self.base_url = (
            base_url or os.environ.get("NULLWATCH_URL", "http://127.0.0.1:7710")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("NULLWATCH_API_KEY")
        self.timeout = timeout
        self.raise_on_error = raise_on_error
        self.default_source = default_source
        self.buffered = buffered
        self.flush_at = flush_at
        self.redact = redact
        self._transport = transport  # e.g. MemoryTransport for testing

        self._buffer: List[Span] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context-manager support (for buffered mode)
    # ------------------------------------------------------------------

    def __enter__(self) -> "NullwatchClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict:
        headers: dict = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _apply_redact(self, payload: dict) -> dict:
        if self.redact is not None:
            return self.redact(payload)
        return payload

    def _request(
        self, method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None
    ) -> Any:
        # Use in-memory transport when provided (for testing)
        if self._transport is not None:
            if method == "POST":
                if body is not None:
                    body = self._apply_redact(body)
                return self._transport.post(path, body or {})
            else:
                return self._transport.get(path, params)

        url = self.base_url + path
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})

        if body is not None:
            body = self._apply_redact(body)

        data = json.dumps(body).encode() if body is not None else None
        req = Request(url, data=data, headers=self._build_headers(), method=method)

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else None
        except HTTPError as e:
            body_text = e.read().decode()
            if self.raise_on_error:
                raise NullwatchError(e.code, body_text) from e
            return None
        except URLError as e:
            if self.raise_on_error:
                raise ConnectionError(f"Cannot reach nullwatch at {self.base_url}: {e.reason}") from e
            return None

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict) -> Any:
        return self._request("POST", path, body=body)

    # ------------------------------------------------------------------
    # Health / capabilities
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._get("/health") or {}

    def capabilities(self) -> dict:
        """Query server capabilities (``GET /v1/capabilities``)."""
        return self._get("/v1/capabilities") or {}

    def is_alive(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Span ingestion
    # ------------------------------------------------------------------

    def _prepare_span(self, span: Span) -> None:
        if span.ended_at_ms is None:
            span.finish()
        if span.source == "python-sdk":
            span.source = self.default_source

    def ingest_span(self, span: Span) -> Optional[dict]:
        self._prepare_span(span)
        if self.buffered:
            with self._lock:
                self._buffer.append(span)
                if len(self._buffer) >= self.flush_at:
                    return self._flush_locked()
            return None
        return self._post("/v1/spans", span.to_dict())

    def ingest_spans(self, spans: List[Span]) -> Optional[dict]:
        items = []
        for s in spans:
            self._prepare_span(s)
            items.append(s.to_dict())
        return self._post("/v1/spans/bulk", {"items": items})

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _flush_locked(self) -> Optional[dict]:
        """Flush the internal buffer (must be called with _lock held)."""
        if not self._buffer:
            return None
        spans = self._buffer[:]
        self._buffer.clear()
        items = [s.to_dict() for s in spans]
        return self._post("/v1/spans/bulk", {"items": items})

    def flush(self) -> Optional[dict]:
        """Flush all buffered spans immediately.

        Returns the API response dict, or *None* when the buffer was empty.
        """
        with self._lock:
            return self._flush_locked()

    def close(self) -> None:
        """Flush any remaining buffered spans and release resources."""
        self.flush()

    # ------------------------------------------------------------------
    # Span query
    # ------------------------------------------------------------------

    def list_spans(
        self,
        *,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        source: Optional[str] = None,
        operation: Optional[str] = None,
        status: Optional[str] = None,
        model: Optional[str] = None,
        tool_name: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        params = {
            "run_id": run_id,
            "trace_id": trace_id,
            "source": source,
            "operation": operation,
            "status": status,
            "model": model,
            "tool_name": tool_name,
            "task_id": task_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "limit": limit,
        }
        result = self._get("/v1/spans", params=params)
        if isinstance(result, dict) and "items" in result:
            return result["items"]
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Eval ingestion / query
    # ------------------------------------------------------------------

    def ingest_eval(self, eval_: Eval) -> Optional[dict]:
        return self._post("/v1/evals", eval_.to_dict())

    def list_evals(
        self,
        *,
        run_id: Optional[str] = None,
        eval_key: Optional[str] = None,
        verdict: Optional[str] = None,
        dataset: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        params = {
            "run_id": run_id,
            "eval_key": eval_key,
            "verdict": verdict,
            "dataset": dataset,
            "limit": limit,
        }
        result = self._get("/v1/evals", params=params)
        if isinstance(result, dict) and "items" in result:
            return result["items"]
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Run query
    # ------------------------------------------------------------------

    def list_runs(self, *, verdict: Optional[str] = None, limit: int = 20) -> List[dict]:
        params = {"verdict": verdict, "limit": limit}
        result = self._get("/v1/runs", params=params)
        if isinstance(result, dict) and "items" in result:
            return result["items"]
        return result if isinstance(result, list) else []

    def get_run(self, run_id: str) -> Optional[RunSummary]:
        try:
            data = self._get(f"/v1/runs/{run_id}")
        except NullwatchError as e:
            if e.status == 404:
                return None
            raise
        if not data:
            return None
        summary_data = data.get("summary", data)
        return RunSummary.from_dict(summary_data, run_id=run_id)

    # ------------------------------------------------------------------
    # Span context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def span(
        self,
        run_id: str,
        operation: str,
        *,
        source: Optional[str] = None,
        model: Optional[str] = None,
        tool_name: Optional[str] = None,
        **kwargs,
    ) -> Generator[Span, None, None]:
        s = Span(
            run_id=run_id,
            operation=operation,
            source=source or self.default_source,
            model=model,
            tool_name=tool_name,
            **kwargs,
        )
        error_occurred = False
        try:
            yield s
        except Exception:
            error_occurred = True
            raise
        finally:
            s.finish(status="error" if error_occurred else "ok")
            try:
                self.ingest_span(s)
            except Exception:
                # Preserve the original user exception from inside the span body.
                if not error_occurred:
                    raise

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    def trace(
        self,
        operation: str,
        *,
        run_id_kwarg: str = "run_id",
        source: Optional[str] = None,
        model: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> Callable:
        """Decorator that wraps a *synchronous* function in a span.

        The decorated function must accept ``run_id`` as a keyword argument
        (or the name configured via *run_id_kwarg*).  If no ``run_id`` is
        found a fresh one is generated automatically.

        Example::

            @client.trace("retriever.search")
            def search_docs(run_id: str, query: str) -> list[str]:
                return retriever.search(query)
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                rid = kwargs.get(run_id_kwarg)
                if rid is None:
                    # Try to find run_id positionally from the function signature
                    sig = inspect.signature(fn)
                    param_names = list(sig.parameters.keys())
                    if run_id_kwarg in param_names:
                        idx = param_names.index(run_id_kwarg)
                        if idx < len(args):
                            rid = args[idx]
                if rid is None:
                    from .models import _new_id
                    rid = _new_id("run-")

                with self.span(
                    rid,
                    operation,
                    source=source,
                    model=model,
                    tool_name=tool_name,
                ):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def atrace(
        self,
        operation: str,
        *,
        run_id_kwarg: str = "run_id",
        source: Optional[str] = None,
        model: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> Callable:
        """Decorator that wraps an *async* function in a span.

        Example::

            @client.atrace("llm.call")
            async def call_model(run_id: str, prompt: str) -> str:
                return await model.generate(prompt)
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                rid = kwargs.get(run_id_kwarg)
                if rid is None:
                    sig = inspect.signature(fn)
                    param_names = list(sig.parameters.keys())
                    if run_id_kwarg in param_names:
                        idx = param_names.index(run_id_kwarg)
                        if idx < len(args):
                            rid = args[idx]
                if rid is None:
                    from .models import _new_id
                    rid = _new_id("run-")

                s = Span(
                    run_id=rid,
                    operation=operation,
                    source=source or self.default_source,
                    model=model,
                    tool_name=tool_name,
                )
                error_occurred = False
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception:
                    error_occurred = True
                    raise
                finally:
                    s.finish(status="error" if error_occurred else "ok")
                    try:
                        self.ingest_span(s)
                    except Exception:
                        if not error_occurred:
                            raise

            return wrapper

        return decorator
