from typing import Dict, List, Optional

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


class ToolCallScorer(BaseScorer):
    """
    Validates LLM-generated tool calls against a schema.

    Catches fabricated tool names, misspelled argument names, and wrong types.
    No ML model needed.
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
        self._tools[tool_schema["name"]] = tool_schema

    def validate(self, tool_call: dict) -> tuple[bool, List[str]]:
        issues: List[str] = []
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {}) or {}

        if name not in self._tools:
            issues.append(f"Unknown tool '{name}'. Known tools: {list(self._tools.keys())}")
            return False, issues

        params = self._tools[name].get("parameters", {})

        for param_name, param_spec in params.items():
            if isinstance(param_spec, dict) and param_spec.get("required", False):
                if param_name not in args:
                    issues.append(f"Missing required argument '{param_name}'")

        for arg_name in args:
            if arg_name not in params:
                close = [p for p in params if _levenshtein(arg_name, p) <= 2]
                hint = f" (did you mean: {close})?" if close else ""
                issues.append(f"Unknown argument '{arg_name}'{hint}")

        for arg_name, arg_value in args.items():
            if arg_name not in params:
                continue
            param_spec = params[arg_name]
            if not isinstance(param_spec, dict):
                continue
            expected_type_str = param_spec.get("type")
            if not expected_type_str:
                continue
            expected_type = _PYTHON_TYPE_MAP.get(expected_type_str.lower())
            if expected_type and not isinstance(arg_value, expected_type):
                actual = type(arg_value).__name__
                issues.append(f"Argument '{arg_name}' expected '{expected_type_str}', got '{actual}'")

        return len(issues) == 0, issues

    def score(
        self,
        run_id: str,
        tool_call: Optional[dict] = None,
        tool_calls: Optional[List[dict]] = None,
        **kwargs,
    ) -> Eval:
        calls = []
        if tool_call:
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
                call_name = call.get("name", "<unknown>")
                all_issues.extend(f"[{call_name}] {issue}" for issue in issues)

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
