from typing import List, Optional, Union

from ..models import Eval, HallucinationResult, HallucinationSpan
from .base import BaseScorer

DEFAULT_THRESHOLD = 0.5
DEFAULT_FAIL_THRESHOLD = 0.3
DEFAULT_MODEL = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


class RAGHallucinationScorer(BaseScorer):
    """
    Detects hallucinations in RAG answers using LettuceDetect.

    Requires: pip install lettucedetect
    Model: https://huggingface.co/KRLabsOrg/lettucedect-large-modernbert-en-v1
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
        dataset: Optional[str] = None,
        fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    ):
        self.model_name = model
        self.threshold = threshold
        self.device = device
        self.dataset = dataset
        self.fail_threshold = fail_threshold
        self._detector = None

    @property
    def eval_key(self) -> str:
        return "rag_hallucination"

    @property
    def scorer_name(self) -> str:
        return self.model_name

    def _load_detector(self):
        if self._detector is not None:
            return self._detector
        try:
            from lettucedetect.models.inference import HallucinationDetector
        except ImportError as e:
            raise ImportError("lettucedetect is required: pip install lettucedetect") from e

        kwargs: dict = {"method": "transformer", "model_path": self.model_name, "lang": "en"}
        if self.device:
            kwargs["device"] = self.device

        self._detector = HallucinationDetector(**kwargs)
        return self._detector

    def detect(self, contexts: Union[str, List[str]], question: str, answer: str) -> HallucinationResult:
        if isinstance(contexts, str):
            contexts = [contexts]

        detector = self._load_detector()
        raw = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")

        hallucinated_spans = []
        for item in raw:
            if isinstance(item, dict):
                conf = item.get("confidence", item.get("hallucination_score", 1.0))
                text, start, end = item.get("text", ""), item.get("start", 0), item.get("end", 0)
            else:
                conf = getattr(item, "confidence", getattr(item, "hallucination_score", 1.0))
                text, start, end = (
                    getattr(item, "text", ""),
                    getattr(item, "start", 0),
                    getattr(item, "end", 0),
                )
            if conf >= self.threshold:
                hallucinated_spans.append(
                    HallucinationSpan(text=text, start=start, end=end, confidence=conf)
                )

        total_chars = len(answer)
        hallucinated_chars = sum(s.end - s.start for s in hallucinated_spans)
        aggregate_score = hallucinated_chars / total_chars if total_chars > 0 else 0.0

        return HallucinationResult(
            is_hallucinated=bool(hallucinated_spans),
            score=aggregate_score,
            spans=hallucinated_spans,
            raw=raw,
        )

    def score(
        self,
        run_id: str,
        contexts: Union[str, List[str]] = "",
        question: str = "",
        answer: str = "",
        **kwargs,
    ) -> Eval:
        result = self.detect(contexts=contexts, question=question, answer=answer)
        should_fail = bool(result.spans) and result.score >= self.fail_threshold

        if result.spans:
            parts = [f'"{s.text.strip()}" (conf={s.confidence:.2f})' for s in result.spans]
            if should_fail:
                notes = "Hallucinated spans detected: " + "; ".join(parts)
            else:
                notes = "Hallucinated spans detected but below fail threshold: " + "; ".join(parts)
        else:
            notes = "No hallucinations detected — answer is grounded in context."

        return Eval(
            run_id=run_id,
            eval_key=self.eval_key,
            scorer=self.scorer_name,
            score=round(1.0 - result.score, 4),
            verdict="fail" if should_fail else "pass",
            dataset=self.dataset,
            notes=notes,
            meta={
                "hallucinated_span_count": len(result.spans),
                "hallucinated_char_ratio": round(result.score, 4),
                "threshold": self.threshold,
                "fail_threshold": self.fail_threshold,
                "passed_below_fail_threshold": bool(result.spans) and not should_fail,
            },
        )
