from nullwatch.scorers import RAGHallucinationScorer


class _FakeDetector:
    def __init__(self, raw):
        self._raw = raw

    def predict(self, **kwargs):
        return self._raw


class TestRAGHallucinationScorer:
    def test_short_hallucinated_span_still_fails(self):
        scorer = RAGHallucinationScorer(threshold=0.5, fail_threshold=0.99)
        scorer._detector = _FakeDetector(
            [
                {
                    "text": "New",
                    "start": 72,
                    "end": 75,
                    "confidence": 0.52,
                }
            ]
        )

        eval_ = scorer.score(
            run_id="run-1",
            contexts=["The Zig programming language was created by Andrew Kelley."],
            question="Complete this sentence with the most likely facts: Zig was created by Andrew Kelley in the city of",
            answer="The Zig programming language was created by Andrew Kelley in the city of New York.",
        )

        assert eval_.verdict == "fail"
        assert eval_.meta["hallucinated_span_count"] == 1
        assert eval_.meta["hallucinated_char_ratio"] < 0.3
        assert '"New"' in eval_.notes

    def test_no_hallucinated_spans_passes(self):
        scorer = RAGHallucinationScorer()
        scorer._detector = _FakeDetector([])

        eval_ = scorer.score(
            run_id="run-1",
            contexts=["Python was created by Guido van Rossum."],
            question="Who created Python?",
            answer="Python was created by Guido van Rossum.",
        )

        assert eval_.verdict == "pass"
        assert eval_.score == 1.0
        assert eval_.meta["hallucinated_span_count"] == 0
