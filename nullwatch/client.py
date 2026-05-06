import contextlib
import json
from typing import Any, Generator, List, Optional
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
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7710",
        timeout: int = 10,
        raise_on_error: bool = True,
        default_source: str = "python-sdk",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.raise_on_error = raise_on_error
        self.default_source = default_source

    def _request(
        self, method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None
    ) -> Any:
        url = self.base_url + path
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})

        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        req = Request(url, data=data, headers=headers, method=method)

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

    def health(self) -> dict:
        return self._get("/health") or {}

    def is_alive(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    def ingest_span(self, span: Span) -> Optional[dict]:
        if span.ended_at_ms is None:
            span.finish()
        if span.source == "python-sdk":
            span.source = self.default_source
        return self._post("/v1/spans", span.to_dict())

    def ingest_spans(self, spans: List[Span]) -> Optional[dict]:
        items = []
        for s in spans:
            if s.ended_at_ms is None:
                s.finish()
            if s.source == "python-sdk":
                s.source = self.default_source
            items.append(s.to_dict())
        return self._post("/v1/spans/bulk", {"items": items})

    def list_spans(
        self,
        *,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
        status: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        params = {
            "run_id": run_id,
            "source": source,
            "status": status,
            "tool_name": tool_name,
            "limit": limit,
        }
        result = self._get("/v1/spans", params=params)
        # nullwatch returns {"items": [...]} for list endpoints
        if isinstance(result, dict) and "items" in result:
            return result["items"]
        return result if isinstance(result, list) else []

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
        # nullwatch returns {"items": [...]} for list endpoints
        if isinstance(result, dict) and "items" in result:
            return result["items"]
        return result if isinstance(result, list) else []

    def list_runs(self, *, verdict: Optional[str] = None, limit: int = 20) -> List[dict]:
        params = {"verdict": verdict, "limit": limit}
        result = self._get("/v1/runs", params=params)
        # nullwatch returns {"items": [...]} for list endpoints
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
            self.ingest_span(s)
