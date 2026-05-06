"""Tests for ToolCallGroundingScorer (keyword backend, no LLM required)."""

import pytest

from nullwatch.scorers import ToolCallGroundingScorer
from nullwatch.scorers.tool_call_grounding import (
    _flatten_args,
    _keyword_is_grounded,
    _number_is_grounded,
)

CONTEXT = (
    "The user wants to search for Python documentation. "
    "They are working on a project called nullwatch-py. "
    "The repository is at github.com/nullclaw/nullwatch-python-sdk."
)


class TestKeywordIsGrounded:
    def test_grounded_word(self):
        grounded, _ = _keyword_is_grounded("Python", CONTEXT)
        assert grounded is True

    def test_grounded_phrase(self):
        grounded, _ = _keyword_is_grounded("Python documentation", CONTEXT)
        assert grounded is True

    def test_hallucinated_word(self):
        grounded, reason = _keyword_is_grounded("Kubernetes cluster deployment", CONTEXT)
        assert grounded is False
        assert "not in context" in reason

    def test_short_value_always_grounded(self):
        # Values shorter than min_word_len are not checked
        grounded, _ = _keyword_is_grounded("en", CONTEXT)
        assert grounded is True

    def test_numeric_string(self):
        # Numbers too short to be meaningful
        grounded, _ = _keyword_is_grounded("5", CONTEXT)
        assert grounded is True

    def test_case_insensitive(self):
        grounded, _ = _keyword_is_grounded("PYTHON", CONTEXT)
        assert grounded is True

    def test_partial_match_passes(self):
        # "Python docs" — "Python" is in context, "docs" is not,
        # but 1/2 = 50% which meets the threshold
        grounded, _ = _keyword_is_grounded("Python docs", CONTEXT)
        assert grounded is True  # 1/2 words matched = 50% >= threshold

    def test_mostly_ungrounded_fails(self):
        grounded, _ = _keyword_is_grounded("Kubernetes Docker AWS Redis", CONTEXT)
        assert grounded is False


class TestNumberIsGrounded:
    def test_grounded_number(self):
        grounded, _ = _number_is_grounded(3, "Limit the results to 3 items.")
        assert grounded is True

    def test_hallucinated_number(self):
        grounded, reason = _number_is_grounded(50, "Limit the results to 3 items.")
        assert grounded is False
        assert "not found in context numbers" in reason

    def test_no_numeric_anchor_is_soft_pass(self):
        grounded, _ = _number_is_grounded(50, "Search the documentation for Zig.")
        assert grounded is True


class TestFlattenArgs:
    def test_simple_string_arg(self):
        result = _flatten_args({"query": "Python docs"})
        assert result == [("query", "Python docs")]

    def test_nested_object(self):
        result = _flatten_args({"filters": {"language": "en", "limit": 5}})
        # Only string values are returned
        assert ("filters.language", "en") in result
        # numeric values are included for numeric grounding checks
        assert ("filters.limit", 5) in result

    def test_array_of_strings(self):
        result = _flatten_args({"paths": ["docs/readme.md", "src/main.py"]})
        assert ("paths[0]", "docs/readme.md") in result
        assert ("paths[1]", "src/main.py") in result

    def test_array_of_numbers(self):
        result = _flatten_args({"limits": [3, 5]})
        assert ("limits[0]", 3) in result
        assert ("limits[1]", 5) in result

    def test_deeply_nested(self):
        result = _flatten_args({"a": {"b": {"c": "deep_value"}}})
        assert ("a.b.c", "deep_value") in result

    def test_empty_args(self):
        assert _flatten_args({}) == []


class TestToolCallGroundingScorer:
    def setup_method(self):
        self.scorer = ToolCallGroundingScorer(context=CONTEXT)

    def test_grounded_call_passes(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_docs", "arguments": {"query": "Python documentation"}},
        )
        assert eval_.verdict == "pass"
        assert eval_.score == 1.0

    def test_hallucinated_call_fails(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "name": "search_docs",
                "arguments": {"query": "Kubernetes Docker AWS cluster"},
            },
        )
        assert eval_.verdict == "fail"
        assert eval_.score == 0.0
        assert "query" in eval_.notes

    def test_no_call_provided(self):
        eval_ = self.scorer.score(run_id="run-1")
        assert eval_.verdict == "fail"
        assert "No tool call provided" in eval_.notes

    def test_empty_context_always_passes(self):
        scorer = ToolCallGroundingScorer(context="")
        eval_ = scorer.score(
            run_id="run-1",
            tool_call={"name": "foo", "arguments": {"query": "anything at all xyz"}},
        )
        assert eval_.verdict == "pass"

    def test_context_as_list(self):
        scorer = ToolCallGroundingScorer(context=["Python docs", "nullwatch project"])
        eval_ = scorer.score(
            run_id="run-1",
            tool_call={"name": "search", "arguments": {"query": "Python"}},
        )
        assert eval_.verdict == "pass"

    def test_context_override_in_score(self):
        # Scorer was created with CONTEXT about Python, but we override with different context
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search", "arguments": {"query": "Rust programming"}},
            context="The user wants Rust documentation. The project uses Rust.",
        )
        assert eval_.verdict == "pass"

    def test_context_not_mutated_after_override(self):
        # After override, the instance should use its original context again
        self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search", "arguments": {"query": "anything"}},
            context="totally different context",
        )
        # Original context should be restored
        assert "Python" in self.scorer.context

    def test_batch_all_grounded(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_calls=[
                {"name": "search_docs", "arguments": {"query": "Python"}},
                {"name": "search_docs", "arguments": {"query": "nullwatch documentation"}},
            ],
        )
        assert eval_.verdict == "pass"
        assert eval_.score == 1.0

    def test_batch_partial_grounded(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_calls=[
                {"name": "search_docs", "arguments": {"query": "Python"}},
                {"name": "search_docs", "arguments": {"query": "Kubernetes Docker AWS"}},
            ],
        )
        assert eval_.verdict == "fail"
        assert eval_.score == 0.5

    def test_meta_structure(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_docs", "arguments": {"query": "Python"}},
        )
        assert eval_.meta is not None
        assert eval_.meta["backend"] == "keyword"
        assert eval_.meta["total_calls"] == 1
        assert eval_.meta["grounded_calls"] == 1
        assert eval_.meta["issues"] == []

    def test_eval_key(self):
        assert self.scorer.eval_key == "tool_call_grounding"

    def test_scorer_name_keyword(self):
        assert self.scorer.scorer_name == "grounding-keyword"

    def test_scorer_name_llm(self):
        scorer = ToolCallGroundingScorer(backend="llm")
        assert scorer.scorer_name == "grounding-llm"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be"):
            ToolCallGroundingScorer(backend="magic")

    def test_non_string_args_ignored(self):
        # Boolean args are ignored; grounded numeric args are checked
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "paginate", "arguments": {"limit": 10, "active": True}},
            context="Pagination limit is 10 for this request.",
        )
        assert eval_.verdict == "pass"

    def test_numeric_arg_checked_against_context(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "paginate", "arguments": {"limit": 50}},
            context="Pagination limit is 10 for this request.",
        )
        assert eval_.verdict == "fail"
        assert "numeric value 50" in eval_.notes

    def test_openai_format_supported(self):
        import json

        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "arguments": json.dumps({"query": "Python documentation"}),
                },
            },
        )
        assert eval_.verdict == "pass"

    def test_anthropic_format_supported(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "type": "tool_use",
                "name": "search_docs",
                "input": {"query": "Python docs"},
            },
        )
        assert eval_.verdict == "pass"

    def test_combined_with_tool_call_scorer(self):
        """Demonstrate the two scorers working together for full coverage."""
        from nullwatch.scorers import ToolCallScorer

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

        tool_call = {"name": "search_docs", "arguments": {"query": "Python documentation"}}

        schema_eval = ToolCallScorer(tools=tools).score(run_id="run-1", tool_call=tool_call)
        grounding_eval = self.scorer.score(run_id="run-1", tool_call=tool_call)

        # Both should pass for a well-formed, grounded call
        assert schema_eval.verdict == "pass"
        assert grounding_eval.verdict == "pass"
