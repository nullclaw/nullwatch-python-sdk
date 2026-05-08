import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@dataclass
class Span:
    run_id: str
    operation: str
    source: str = "python-sdk"

    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    started_at_ms: Optional[int] = None
    ended_at_ms: Optional[int] = None
    duration_ms: Optional[int] = None

    status: str = "ok"  # "ok" | "error"

    model: Optional[str] = None
    prompt_version: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    tool_name: Optional[str] = None
    meta: Optional[dict] = None

    def __post_init__(self):
        if self.span_id is None:
            self.span_id = _new_id("span-")
        if self.trace_id is None:
            self.trace_id = _new_id("trace-")
        if self.started_at_ms is None:
            self.started_at_ms = _now_ms()

    def finish(self, status: str = "ok") -> "Span":
        self.ended_at_ms = _now_ms()
        self.status = status
        if self.started_at_ms:
            self.duration_ms = self.ended_at_ms - self.started_at_ms
        return self

    # ------------------------------------------------------------------
    # Provider helpers — best-effort adapters, no provider SDK required
    # ------------------------------------------------------------------

    def record_tokens(self, *, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None) -> "Span":
        """Set token counts directly."""
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        return self

    def record_cost(self, cost_usd: float) -> "Span":
        """Set the cost in USD."""
        self.cost_usd = cost_usd
        return self

    def record_openai_usage(self, response: Any) -> "Span":
        """Extract token counts and cost from an OpenAI ChatCompletion response object or dict.

        Works with ``openai.types.chat.ChatCompletion`` objects and plain dicts
        returned by OpenAI-compatible APIs.  Missing fields are silently skipped.
        """
        usage = None
        if isinstance(response, dict):
            usage = response.get("usage", {})
        else:
            usage = getattr(response, "usage", None)

        if usage is None:
            return self

        if isinstance(usage, dict):
            self.input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            self.output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            cost = usage.get("total_cost") or usage.get("cost_usd")
        else:
            self.input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
            self.output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
            cost = getattr(usage, "total_cost", None) or getattr(usage, "cost_usd", None)

        if cost is not None:
            self.cost_usd = float(cost)
        return self

    def record_anthropic_usage(self, response: Any) -> "Span":
        """Extract token counts from an Anthropic ``Message`` response object or dict.

        Works with ``anthropic.types.Message`` objects and plain dicts returned
        by Anthropic-compatible APIs.  Missing fields are silently skipped.
        """
        usage = None
        if isinstance(response, dict):
            usage = response.get("usage", {})
        else:
            usage = getattr(response, "usage", None)

        if usage is None:
            return self

        if isinstance(usage, dict):
            self.input_tokens = usage.get("input_tokens")
            self.output_tokens = usage.get("output_tokens")
        else:
            self.input_tokens = getattr(usage, "input_tokens", None)
            self.output_tokens = getattr(usage, "output_tokens", None)
        return self

    def to_dict(self) -> dict:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        meta = payload.pop("meta", None)
        if meta is not None:
            payload["attributes_json"] = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        return payload


@dataclass
class Eval:
    run_id: str
    eval_key: str
    score: float
    verdict: str  # "pass" | "fail"

    scorer: str = "heuristic"
    dataset: Optional[str] = None
    notes: Optional[str] = None
    meta: Optional[dict] = None

    def to_dict(self) -> dict:
        payload = {k: v for k, v in asdict(self).items() if v is not None}
        meta = payload.pop("meta", None)
        if meta is not None:
            payload["metadata_json"] = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        return payload


@dataclass
class RunSummary:
    run_id: str
    span_count: int = 0
    eval_count: int = 0
    error_count: int = 0
    total_duration_ms: Optional[int] = None
    total_cost_usd: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    pass_count: int = 0
    fail_count: int = 0
    verdict: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, run_id: Optional[str] = None) -> "RunSummary":
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "verdict" not in filtered and "overall_verdict" in data:
            filtered["verdict"] = data["overall_verdict"]
        if "run_id" not in filtered:
            filtered["run_id"] = run_id or data.get("id", "unknown")
        return cls(**filtered)


@dataclass
class HallucinationSpan:
    text: str
    start: int
    end: int
    confidence: float


@dataclass
class HallucinationResult:
    is_hallucinated: bool
    score: float  # 0.0 = clean, 1.0 = fully hallucinated
    spans: List[HallucinationSpan] = field(default_factory=list)
    raw: Optional[Any] = None

    def to_eval(self, run_id: str, dataset: Optional[str] = None, notes: Optional[str] = None) -> Eval:
        hallucinated_texts = [s.text for s in self.spans]
        eval_notes = notes or (
            f"Hallucinated spans: {hallucinated_texts}"
            if hallucinated_texts
            else "No hallucinations detected"
        )
        return Eval(
            run_id=run_id,
            eval_key="rag_hallucination",
            scorer="lettucedetect-large-modernbert-en-v1",
            score=1.0 - self.score,
            verdict="fail" if self.is_hallucinated else "pass",
            dataset=dataset,
            notes=eval_notes,
        )
