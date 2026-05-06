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

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


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
        return {k: v for k, v in asdict(self).items() if v is not None}


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
