import json
from typing import Dict, List, Optional, Union

from ..models import Eval
from .base import BaseScorer

_PYTHON_TYPE_MAP = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "number": (int, float),
    "float": float,
    "boolean": bool,
    "bool": bool,
    "array": list,
    "list": list,
    "object": dict,
    "dict": dict,
    "null": type(None),
}


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalize_tool_call(call: dict) -> dict:
    """
    Normalize various LLM tool call formats into internal format.

    Internal format: {"name": str, "arguments": dict}

    Handles:
    - OpenAI:    {"type": "function", "function": {"name": ..., "arguments": "<json str>"}}
    - Anthropic: {"type": "tool_use", "name": ..., "input": {...}}
    - Internal:  {"name": ..., "arguments": {...}}  (pass-through)
    """
    # OpenAI function call format
    if "function" in call:
        fn = call["function"]
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                raw_args = {}
        return {"name": fn.get("name", ""), "arguments": raw_args}

    # Anthropic tool_use format
    if call.get("type") == "tool_use":
        return {"name": call.get("name", ""), "arguments": call.get("input", {})}

    # Internal / already-normalized format
    return call


class ToolCallScorer(BaseScorer):
    """
    Validates LLM-generated tool calls against a JSON-schema-like spec.

    Checks performed:
    - Tool name exists in registered tools (with Levenshtein typo hints)
    - All required arguments are present
    - No unknown argument names (with Levenshtein-based typo hints)
    - Argument types match the schema ("string", "integer", "boolean", etc.)
    - Enum values are valid when "enum" is specified
    - Numeric values satisfy "minimum" / "maximum" constraints when specified

    Accepts tool calls in OpenAI, Anthropic, or internal format automatically.
    """

    def __init__(self, tools: Optional[List[dict]] = None, dataset: Optional[str] = None):
        self._tools: Dict[str, dict] = {}
        for t in tools or []:
            self._tools[t["name"]] = t
        self.dataset = dataset

    @property
    def eval_key(self) -> str:
        return "tool_call_validity"

    @property
    def scorer_name(self) -> str:
        return "schema-validator"

    def register_tool(self, tool_schema: dict) -> None:
        """Register a tool schema. Can be called after construction."""
        self._tools[tool_schema["name"]] = tool_schema

    def validate(self, tool_call: dict) -> tuple[bool, List[str]]:
        """
        Validate a single tool call (any supported format).

        Returns (is_valid, list_of_issue_strings).
        """
        call = normalize_tool_call(tool_call)
        issues: List[str] = []
        name = call.get("name", "")
        args = call.get("arguments", {}) or {}

        # --- 1. Tool name must be registered ---
        if name not in self._tools:
            close = [t for t in self._tools if _levenshtein(name, t) <= 2]
            hint = f" (did you mean: {close})?" if close else ""
            issues.append(f"Unknown tool '{name}'{hint}. Known tools: {list(self._tools.keys())}")
            return False, issues

        params = self._tools[name].get("parameters", {})

        # --- 2. Required arguments must be present ---
        for param_name, param_spec in params.items():
            if isinstance(param_spec, dict) and param_spec.get("required", False):
                if param_name not in args:
                    issues.append(f"Missing required argument '{param_name}'")

        # --- 3. Unknown argument names (with typo hints) ---
        for arg_name in args:
            if arg_name not in params:
                close = [p for p in params if _levenshtein(arg_name, p) <= 2]
                hint = f" (did you mean: {close})?" if close else ""
                issues.append(f"Unknown argument '{arg_name}'{hint}")

        # --- 4. Type, enum, and range validation ---
        for arg_name, arg_value in args.items():
            if arg_name not in params:
                continue
            param_spec = params[arg_name]
            if not isinstance(param_spec, dict):
                continue

            # Type check
            # Note: bool is a subclass of int in Python, so we must check it explicitly
            # before checking for int/number to avoid False/True passing as integer.
            expected_type_str = param_spec.get("type")
            if expected_type_str:
                expected_type = _PYTHON_TYPE_MAP.get(expected_type_str.lower())
                is_bool_value = isinstance(arg_value, bool)
                is_bool_schema = expected_type_str.lower() in ("boolean", "bool")
                type_mismatch = expected_type and not isinstance(arg_value, expected_type)
                bool_as_int = is_bool_value and not is_bool_schema  # True/False passed as integer
                if type_mismatch or bool_as_int:
                    actual = type(arg_value).__name__
                    issues.append(
                        f"Argument '{arg_name}' expected type '{expected_type_str}', got '{actual}'"
                    )
                    continue  # skip further checks if type is already wrong

            # Enum check
            allowed_values = param_spec.get("enum")
            if allowed_values is not None and arg_value not in allowed_values:
                issues.append(
                    f"Argument '{arg_name}' value {arg_value!r} not in allowed values: {allowed_values}"
                )

            # Numeric range checks (guard against bool, which is a subclass of int)
            if isinstance(arg_value, (int, float)) and not isinstance(arg_value, bool):
                minimum = param_spec.get("minimum")
                maximum = param_spec.get("maximum")
                if minimum is not None and arg_value < minimum:
                    issues.append(f"Argument '{arg_name}' value {arg_value} is below minimum {minimum}")
                if maximum is not None and arg_value > maximum:
                    issues.append(f"Argument '{arg_name}' value {arg_value} exceeds maximum {maximum}")

        return len(issues) == 0, issues

    def score(
        self,
        run_id: str,
        tool_call: Optional[dict] = None,
        tool_calls: Optional[Union[List[dict], None]] = None,
        **kwargs,
    ) -> Eval:
        """
        Score one or more tool calls.

        Args:
            run_id:     The run identifier to attach the eval to.
            tool_call:  A single tool call dict (any supported format).
            tool_calls: A list of tool call dicts (any supported format).

        Returns an Eval with:
            score   = fraction of valid calls (1.0 = all valid)
            verdict = "pass" if all valid, "fail" otherwise
            notes   = human-readable summary of issues
            meta    = structured breakdown for downstream analysis
        """
        calls: List[dict] = []
        if tool_call is not None:  # explicit None check: {} is a valid (empty args) call
            calls.append(tool_call)
        if tool_calls:
            calls.extend(tool_calls)

        if not calls:
            return Eval(
                run_id=run_id,
                eval_key=self.eval_key,
                scorer=self.scorer_name,
                score=0.0,
                verdict="fail",
                dataset=self.dataset,
                notes="No tool call provided to validate.",
            )

        all_issues: List[str] = []
        valid_count = 0

        for call in calls:
            is_valid, issues = self.validate(call)
            if is_valid:
                valid_count += 1
            else:
                normalized_name = normalize_tool_call(call).get("name", "<unknown>")
                all_issues.extend(f"[{normalized_name}] {issue}" for issue in issues)

        total = len(calls)
        pass_rate = valid_count / total

        if not all_issues:
            notes = f"All {total} tool call(s) passed schema validation."
        else:
            notes = f"{valid_count}/{total} valid. Issues: " + "; ".join(all_issues)

        return Eval(
            run_id=run_id,
            eval_key=self.eval_key,
            scorer=self.scorer_name,
            score=round(pass_rate, 4),
            verdict="pass" if not all_issues else "fail",
            dataset=self.dataset,
            notes=notes,
            meta={"total_calls": total, "valid_calls": valid_count, "issues": all_issues},
        )
