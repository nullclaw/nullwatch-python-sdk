"""Tests for nullwatch scorers (no ML model required for tool_call tests)."""

from nullwatch.scorers import ToolCallScorer
from nullwatch.scorers.tool_call import _levenshtein, normalize_tool_call

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
    {
        "name": "set_status",
        "parameters": {
            "status": {
                "type": "string",
                "required": True,
                "enum": ["active", "inactive", "pending"],
            },
        },
    },
    {
        "name": "paginate",
        "parameters": {
            "limit": {"type": "integer", "required": True, "minimum": 1, "maximum": 100},
            "offset": {"type": "integer", "required": False, "minimum": 0},
        },
    },
]


class TestToolCallScorer:
    def setup_method(self):
        self.scorer = ToolCallScorer(tools=TOOLS, dataset="test")

    # --- basic happy path ---

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

    # --- tool name errors ---

    def test_unknown_tool(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "nonexistent_tool", "arguments": {}},
        )
        assert eval_.verdict == "fail"
        assert "Unknown tool" in eval_.notes

    def test_misspelled_tool_name_suggests_correction(self):
        # "search_web" vs "search_wab" — distance 1
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_wab", "arguments": {"query": "zig"}},
        )
        assert eval_.verdict == "fail"
        assert "search_web" in eval_.notes  # typo hint present

    # --- argument name errors ---

    def test_missing_required_arg(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {}},
        )
        assert eval_.verdict == "fail"
        assert "Missing required argument 'query'" in eval_.notes

    def test_misspelled_arg_suggests_correction(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {"querY": "zig"}},
        )
        assert eval_.verdict == "fail"
        assert "Unknown argument 'querY'" in eval_.notes
        assert "query" in eval_.notes  # correct spelling suggested

    # --- type errors ---

    def test_wrong_type(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {"query": "zig", "max_results": "five"}},
        )
        assert eval_.verdict == "fail"
        assert "max_results" in eval_.notes

    # --- enum validation ---

    def test_valid_enum_value(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "set_status", "arguments": {"status": "active"}},
        )
        assert eval_.verdict == "pass"

    def test_invalid_enum_value(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "set_status", "arguments": {"status": "maybe"}},
        )
        assert eval_.verdict == "fail"
        assert "not in allowed values" in eval_.notes
        assert "maybe" in eval_.notes

    # --- numeric range validation ---

    def test_valid_range(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "paginate", "arguments": {"limit": 50}},
        )
        assert eval_.verdict == "pass"

    def test_below_minimum(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "paginate", "arguments": {"limit": 0}},
        )
        assert eval_.verdict == "fail"
        assert "below minimum" in eval_.notes

    def test_above_maximum(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "paginate", "arguments": {"limit": 200}},
        )
        assert eval_.verdict == "fail"
        assert "exceeds maximum" in eval_.notes

    def test_negative_offset_fails(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "paginate", "arguments": {"limit": 10, "offset": -1}},
        )
        assert eval_.verdict == "fail"
        assert "offset" in eval_.notes

    # --- OpenAI / Anthropic format normalization ---

    def test_openai_format_string_args(self):
        """OpenAI returns arguments as a JSON string."""
        import json

        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": json.dumps({"query": "zig lang"}),
                },
            },
        )
        assert eval_.verdict == "pass"

    def test_openai_format_dict_args(self):
        """Some wrappers already decode the arguments dict."""
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "type": "function",
                "function": {"name": "search_web", "arguments": {"query": "zig lang"}},
            },
        )
        assert eval_.verdict == "pass"

    def test_openai_format_invalid_call(self):
        """OpenAI format with a schema violation should still fail."""
        import json

        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": json.dumps({"query": "zig", "max_results": "many"}),
                },
            },
        )
        assert eval_.verdict == "fail"
        assert "max_results" in eval_.notes

    def test_anthropic_tool_use_format(self):
        """Anthropic uses type='tool_use' with 'input' instead of 'arguments'."""
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "type": "tool_use",
                "id": "toolu_01abc",
                "name": "search_web",
                "input": {"query": "zig lang"},
            },
        )
        assert eval_.verdict == "pass"

    def test_anthropic_format_missing_required(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={
                "type": "tool_use",
                "name": "search_web",
                "input": {},  # missing required 'query'
            },
        )
        assert eval_.verdict == "fail"
        assert "query" in eval_.notes

    # --- batch scoring ---

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

    def test_mixed_formats_in_batch(self):
        """Batch with OpenAI + internal format together."""
        import json

        eval_ = self.scorer.score(
            run_id="run-1",
            tool_calls=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "arguments": json.dumps({"query": "zig"}),
                    },
                },
                {"name": "read_file", "arguments": {"path": "/etc/hosts"}},
            ],
        )
        assert eval_.verdict == "pass"
        assert eval_.score == 1.0

    # --- edge cases ---

    def test_no_call_provided(self):
        eval_ = self.scorer.score(run_id="run-1")
        assert eval_.verdict == "fail"
        assert "No tool call provided" in eval_.notes

    def test_empty_dict_tool_call_is_not_ignored(self):
        """Bug fix: tool_call={} should NOT be silently dropped (it was with `if tool_call:`)."""
        eval_ = self.scorer.score(run_id="run-1", tool_call={})
        # {} has no "name" key → treated as unknown tool ""
        assert eval_.verdict == "fail"
        assert eval_.score == 0.0

    def test_eval_key(self):
        assert self.scorer.eval_key == "tool_call_validity"

    def test_scorer_name(self):
        assert self.scorer.scorer_name == "schema-validator"

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

    def test_meta_contains_structured_issues(self):
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "search_web", "arguments": {}},
        )
        assert eval_.meta is not None
        assert eval_.meta["total_calls"] == 1
        assert eval_.meta["valid_calls"] == 0
        assert len(eval_.meta["issues"]) > 0

    def test_boolean_not_treated_as_integer(self):
        """bool is a subclass of int in Python — make sure True/False don't pass integer checks."""
        eval_ = self.scorer.score(
            run_id="run-1",
            tool_call={"name": "paginate", "arguments": {"limit": True}},
        )
        # True == 1 as int, but type is bool not int
        assert eval_.verdict == "fail"
        assert "expected type 'integer'" in eval_.notes


class TestNormalizeToolCall:
    def test_internal_format_passthrough(self):
        call = {"name": "foo", "arguments": {"x": 1}}
        assert normalize_tool_call(call) == call

    def test_openai_string_args(self):
        result = normalize_tool_call(
            {
                "type": "function",
                "function": {"name": "foo", "arguments": '{"x": 1}'},
            }
        )
        assert result == {"name": "foo", "arguments": {"x": 1}}

    def test_openai_dict_args(self):
        result = normalize_tool_call(
            {
                "type": "function",
                "function": {"name": "foo", "arguments": {"x": 1}},
            }
        )
        assert result == {"name": "foo", "arguments": {"x": 1}}

    def test_openai_malformed_json_args(self):
        result = normalize_tool_call(
            {
                "type": "function",
                "function": {"name": "foo", "arguments": "{broken json"},
            }
        )
        assert result["name"] == "foo"
        assert result["arguments"] == {}

    def test_anthropic_tool_use(self):
        result = normalize_tool_call(
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "foo",
                "input": {"x": 1},
            }
        )
        assert result == {"name": "foo", "arguments": {"x": 1}}


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("abc", "abc") == 0

    def test_one_substitution(self):
        assert _levenshtein("query", "querY") == 1

    def test_empty(self):
        assert _levenshtein("", "abc") == 3

    def test_symmetric(self):
        assert _levenshtein("abc", "xyz") == _levenshtein("xyz", "abc")

    def test_insert(self):
        assert _levenshtein("search_web", "search_wab") == 1
