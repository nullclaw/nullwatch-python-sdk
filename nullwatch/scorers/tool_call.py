import json
import re
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


def _extract_argument_parse_error(call: dict) -> Optional[str]:
    """Return a validation error if function.arguments contains malformed JSON."""
    if "function" not in call:
        return None
    raw_args = call["function"].get("arguments", {})
    if not isinstance(raw_args, str):
        return None
    try:
        json.loads(raw_args)
    except (json.JSONDecodeError, ValueError) as exc:
        return f"Malformed JSON in tool arguments: {exc}"
    return None


def _normalize_tool_schema(tool_schema: dict) -> dict:
    """Normalize internal or OpenAI-style tool schemas into a JSON Schema object."""
    if "function" in tool_schema:
        tool_schema = tool_schema["function"]

    name = tool_schema["name"]
    parameters = tool_schema.get("parameters", {})

    if isinstance(parameters, dict) and parameters.get("type") == "object":
        normalized = dict(parameters)
        normalized.setdefault("properties", {})
        return {"name": name, "schema": normalized}

    properties: Dict[str, dict] = {}
    required: List[str] = []
    for param_name, param_spec in parameters.items():
        if isinstance(param_spec, dict):
            spec_copy = dict(param_spec)
        else:
            spec_copy = {}
        if spec_copy.pop("required", False):
            required.append(param_name)
        properties[param_name] = spec_copy

    return {
        "name": name,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _format_unknown_key_issue(path: str, key: str, known_keys: List[str]) -> str:
    close = [candidate for candidate in known_keys if _levenshtein(key, candidate) <= 2]
    hint = f" (did you mean: {close})?" if close else ""
    if path:
        return f"Unknown field '{path}.{key}'{hint}"
    return f"Unknown argument '{key}'{hint}"


def _format_missing_key_issue(path: str, key: str) -> str:
    if path:
        return f"Missing required field '{path}.{key}'"
    return f"Missing required argument '{key}'"


def _format_value_label(path: str) -> str:
    if "." in path or "[" in path:
        return f"Field '{path}'"
    return f"Argument '{path}'"


def _validate_schema_value(value, schema: dict, path: str, issues: List[str]) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema_type = schema_type.lower()

    if schema_type in ("object", "dict") or "properties" in schema or "required" in schema:
        if not isinstance(value, dict):
            actual = type(value).__name__
            issues.append(f"{_format_value_label(path)} expected type 'object', got '{actual}'")
            return

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_properties = schema.get("additionalProperties", False)

        for key in required:
            if key not in value:
                issues.append(_format_missing_key_issue(path, key))

        for key, child_value in value.items():
            if key not in properties:
                if additional_properties is False:
                    issues.append(_format_unknown_key_issue(path, key, list(properties.keys())))
                continue
            child_path = f"{path}.{key}" if path else key
            _validate_schema_value(child_value, properties[key], child_path, issues)
        return

    if schema_type in ("array", "list") or "items" in schema:
        if not isinstance(value, list):
            actual = type(value).__name__
            issues.append(f"{_format_value_label(path)} expected type 'array', got '{actual}'")
            return

        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            issues.append(
                f"{_format_value_label(path)} has {len(value)} item(s), below minimum {min_items}"
            )
        if max_items is not None and len(value) > max_items:
            issues.append(
                f"{_format_value_label(path)} has {len(value)} item(s), exceeds maximum {max_items}"
            )

        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{idx}]", issues)
        return

    if schema_type:
        expected_type = _PYTHON_TYPE_MAP.get(schema_type)
        is_bool_value = isinstance(value, bool)
        is_bool_schema = schema_type in ("boolean", "bool")
        type_mismatch = expected_type and not isinstance(value, expected_type)
        bool_as_int = is_bool_value and not is_bool_schema
        if type_mismatch or bool_as_int:
            actual = type(value).__name__
            issues.append(f"{_format_value_label(path)} expected type '{schema_type}', got '{actual}'")
            return

    allowed_values = schema.get("enum")
    if allowed_values is not None and value not in allowed_values:
        issues.append(
            f"{_format_value_label(path)} value {value!r} not in allowed values: {allowed_values}"
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            issues.append(f"{_format_value_label(path)} value {value} is below minimum {minimum}")
        if maximum is not None and value > maximum:
            issues.append(f"{_format_value_label(path)} value {value} exceeds maximum {maximum}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        if min_length is not None and len(value) < min_length:
            issues.append(
                f"{_format_value_label(path)} length {len(value)} is below minimum {min_length}"
            )
        if max_length is not None and len(value) > max_length:
            issues.append(
                f"{_format_value_label(path)} length {len(value)} exceeds maximum {max_length}"
            )
        if pattern is not None and re.search(pattern, value) is None:
            issues.append(
                f"{_format_value_label(path)} value {value!r} does not match pattern {pattern!r}"
            )


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
            normalized = _normalize_tool_schema(t)
            self._tools[normalized["name"]] = normalized
        self.dataset = dataset

    @property
    def eval_key(self) -> str:
        return "tool_call_validity"

    @property
    def scorer_name(self) -> str:
        return "schema-validator"

    def register_tool(self, tool_schema: dict) -> None:
        """Register a tool schema. Can be called after construction."""
        normalized = _normalize_tool_schema(tool_schema)
        self._tools[normalized["name"]] = normalized

    def validate(self, tool_call: dict) -> tuple[bool, List[str]]:
        """
        Validate a single tool call (any supported format).

        Returns (is_valid, list_of_issue_strings).
        """
        call = normalize_tool_call(tool_call)
        issues: List[str] = []
        name = call.get("name", "")
        args = call.get("arguments", {}) or {}
        parse_error = _extract_argument_parse_error(tool_call)
        if parse_error:
            issues.append(parse_error)

        # --- 1. Tool name must be registered ---
        if name not in self._tools:
            close = [t for t in self._tools if _levenshtein(name, t) <= 2]
            hint = f" (did you mean: {close})?" if close else ""
            issues.append(f"Unknown tool '{name}'{hint}. Known tools: {list(self._tools.keys())}")
            return False, issues

        schema = self._tools[name]["schema"]
        _validate_schema_value(args, schema, "", issues)

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
