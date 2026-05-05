"""Tests for nullwatch scorers (no ML model required for tool_call tests)."""

from nullwatch.scorers import ToolCallScorer
from nullwatch.scorers.tool_call import _levenshtein

TOOLS = [
    {
        "name": "search_web",
        "parameters": {
            "query": {"type": "string", "required": True},
            "max_results": {"type": "integer", "required": False},
        },
    },
    {
        "name": "read_file",
        "parameters": {
            "path": {"type": "string", "required": True},
            "encoding": {"type": "string", "required": False},
        },
    },
]


class TestToolCallScorer:
    def setup_method(self):
        self.scorer = ToolCallScorer(tools=TOOLS, dataset="test")

    def test_valid_call(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {"query": "zig lang"}},
        )
        assert eval_.verdict == "pass"
        assert eval_.score == 1.0

    def test_valid_call_all_params(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {"query": "zig", "max_results": 5}},
        )
        assert eval_.verdict == "pass"

    def test_unknown_tool(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "nonexistent_tool", "arguments": {}},
        )
        assert eval_.verdict == "fail"
        assert "Unknown tool" in eval_.notes

    def test_missing_required_arg(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {}},
        )
        assert eval_.verdict == "fail"
        assert "Missing required argument 'query'" in eval_.notes

    def test_misspelled_arg(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {"querY": "zig"}},
        )
        assert eval_.verdict == "fail"
        assert "Unknown argument 'querY'" in eval_.notes
        # Should suggest the correct spelling
        assert "query" in eval_.notes

    def test_wrong_type(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {"query": "zig", "max_results": "five"}},
        )
        assert eval_.verdict == "fail"
        assert "max_results" in eval_.notes

    def test_multiple_calls_partial_valid(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_calls=[
                {"name": "search_web", "arguments": {"query": "zig"}},
                {"name": "fake_tool", "arguments": {}},
            ],
        )
        assert eval_.verdict == "fail"
        assert eval_.score == 0.5  # 1 of 2 valid

    def test_multiple_calls_all_valid(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_calls=[
                {"name": "search_web", "arguments": {"query": "zig"}},
                {"name": "read_file", "arguments": {"path": "/tmp/file.txt"}},
            ],
        )
        assert eval_.verdict == "pass"
        assert eval_.score == 1.0

    def test_no_call_provided(self):
        eval_ = self.scorer.score(run_id="run-1")
        assert eval_.verdict == "fail"

    def test_eval_key(self):
        assert self.scorer.eval_key == "tool_call_validity"

    def test_register_tool(self):
        self.scorer.register_tool(
            {
                "name": "new_tool",
                "parameters": {"x": {"type": "integer", "required": True}},
            }
        )
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "new_tool", "arguments": {"x": 42}},
        )
        assert eval_.verdict == "pass"


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_one_insert(self):
        assert _levenshtein("query", "querY") == 1

    def test_empty(self):
        assert _levenshtein("", "abc") == 3
