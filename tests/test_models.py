"""Tests for nullwatch data models."""

import time

from nullwatch.models import Eval, HallucinationResult, HallucinationSpan, Span


class TestSpan:
    def test_auto_ids(self):
        s = Span(run_id="run-1", operation="llm.call")
        assert s.span_id is not None
        assert s.trace_id is not None
        assert s.started_at_ms is not None

    def test_finish(self):
        s = Span(run_id="run-1", operation="llm.call")
        time.sleep(0.01)
        s.finish()
        assert s.ended_at_ms is not None
        assert s.duration_ms is not None
        assert s.duration_ms >= 0
        assert s.status == "ok"

    def test_finish_error(self):
        s = Span(run_id="run-1", operation="llm.call")
        s.finish(status="error")
        assert s.status == "error"

    def test_to_dict_excludes_none(self):
        s = Span(run_id="run-1", operation="llm.call")
        s.finish()
        d = s.to_dict()
        assert "run_id" in d
        assert "operation" in d
        # Optional fields that weren't set should not appear
        assert "model" not in d
        assert "tool_name" not in d

    def test_to_dict_includes_model(self):
        s = Span(run_id="run-1", operation="llm.call", model="gpt-4o")
        d = s.to_dict()
        assert d["model"] == "gpt-4o"


class TestEval:
    def test_basic(self):
        e = Eval(run_id="run-1", eval_key="helpfulness", score=0.9, verdict="pass")
        assert e.scorer == "heuristic"
        d = e.to_dict()
        assert d["score"] == 0.9
        assert d["verdict"] == "pass"

    def test_to_dict_excludes_none(self):
        e = Eval(run_id="run-1", eval_key="test", score=1.0, verdict="pass")
        d = e.to_dict()
        assert "dataset" not in d
        assert "notes" not in d


class TestHallucinationResult:
    def test_to_eval_pass(self):
        result = HallucinationResult(is_hallucinated=False, score=0.0, spans=[])
        eval_ = result.to_eval(run_id="run-1")
        assert eval_.verdict == "pass"
        assert eval_.eval_key == "rag_hallucination"
        assert eval_.score == 1.0

    def test_to_eval_fail(self):
        spans = [HallucinationSpan(text="wrong fact", start=0, end=10, confidence=0.95)]
        result = HallucinationResult(is_hallucinated=True, score=0.5, spans=spans)
        eval_ = result.to_eval(run_id="run-1")
        assert eval_.verdict == "fail"
        assert eval_.score == 0.5
        assert "wrong fact" in eval_.notes
