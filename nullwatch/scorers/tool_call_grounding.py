import json
import math
import re
import urllib.request
from typing import List, Optional, Union
from urllib.error import URLError

from ..models import Eval
from .base import BaseScorer
from .tool_call import normalize_tool_call

_OPERATIONAL_STRING_ARG_NAMES = {
    "path",
    "paths",
    "cwd",
    "directory",
    "dir",
    "root",
    "workspace",
    "workspace_dir",
    "file",
    "filename",
    "url",
    "uri",
    "endpoint",
    "base_url",
    "command",
    "cmd",
    "program",
    "executable",
    "model",
    "provider",
}

_OPERATIONAL_NUMERIC_ARG_NAMES = {
    "max_results",
    "offset",
    "page",
    "page_size",
    "timeout",
    "timeout_ms",
    "retries",
    "temperature",
    "top_k",
    "top_p",
    "port",
}


def _flatten_args(args: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Recursively extract scalar argument values with their dotted paths."""
    result = []
    for key, value in args.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, str):
            result.append((path, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result.append((path, value))
        elif isinstance(value, dict):
            result.extend(_flatten_args(value, path))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    result.append((f"{path}[{i}]", item))
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    result.append((f"{path}[{i}]", item))
                elif isinstance(item, dict):
                    result.extend(_flatten_args(item, f"{path}[{i}]"))
    return result


def _extract_context_numbers(context: str) -> list[float]:
    """Extract numeric anchors from free-text context."""
    matches = re.findall(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?", context)
    return [float(m) for m in matches]


def _leaf_arg_name(path: str) -> str:
    normalized = re.sub(r"\[\d+\]", "", path)
    return normalized.rsplit(".", 1)[-1].lower()


def _looks_like_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    return (
        stripped.startswith(("/", "~/", "./", "../"))
        or ("\\" in stripped)
        or ("/" in stripped and " " not in stripped and not stripped.startswith(("http://", "https://")))
    )


def _looks_like_url(value: str) -> bool:
    return value.strip().startswith(("http://", "https://"))


def _looks_like_shell_command(value: str) -> bool:
    stripped = value.strip()
    if not stripped or "\n" in stripped:
        return False
    first = stripped.split()[0]
    return first in {
        "pwd",
        "ls",
        "cat",
        "find",
        "grep",
        "rg",
        "git",
        "python",
        "python3",
        "pytest",
        "zig",
        "ollama",
        "npm",
        "pnpm",
        "bun",
        "cargo",
        "make",
        "echo",
    }


def _is_operational_string_arg(path: str, value: str) -> bool:
    name = _leaf_arg_name(path)
    if name in _OPERATIONAL_STRING_ARG_NAMES:
        return True
    return _looks_like_path(value) or _looks_like_url(value) or _looks_like_shell_command(value)


def _is_operational_numeric_arg(path: str) -> bool:
    return _leaf_arg_name(path) in _OPERATIONAL_NUMERIC_ARG_NAMES


def _number_is_grounded(value: Union[int, float], context: str) -> tuple[bool, str]:
    """
    Heuristic numeric grounding check.

    If the context provides explicit numeric anchors, require the exact value
    to appear there. If the context has no numbers at all, treat the value as
    uncheckable rather than hallucinated.
    """
    context_numbers = _extract_context_numbers(context)
    if not context_numbers:
        return True, "context contains no explicit numeric anchors"

    value_num = float(value)
    if any(
        math.isclose(value_num, candidate, rel_tol=0.0, abs_tol=1e-9)
        for candidate in context_numbers
    ):
        return True, f"numeric value {value} found in context"

    rendered = []
    for candidate in context_numbers[:8]:
        rendered.append(str(int(candidate)) if candidate.is_integer() else str(candidate))
    suffix = "..." if len(context_numbers) > 8 else ""
    return False, f"numeric value {value} not found in context numbers: {rendered}{suffix}"


def _keyword_is_grounded(value: str, context: str, min_word_len: int = 3) -> tuple[bool, str]:
    """
    Heuristic check: is this argument value grounded in the context?

    Strategy:
    1. Extract content words (len >= min_word_len) from the argument value.
    2. Check if at least half of them appear in the context (case-insensitive).
    3. Short values (< min_word_len chars) are always considered grounded —
       they're likely structural (e.g. "en", "5", "true").

    Returns (is_grounded, reason_string).
    """
    value_stripped = value.strip()
    if len(value_stripped) < min_word_len:
        return True, "value too short to meaningfully check"

    # Tokenize: keep alphanumeric words
    words = re.findall(r"\b[a-zA-Z0-9_\-]{%d,}\b" % min_word_len, value_stripped)
    if not words:
        return True, "no meaningful words to check"

    context_lower = context.lower()
    matched = [w for w in words if w.lower() in context_lower]
    ratio = len(matched) / len(words)

    if ratio >= 0.5:
        return True, f"{len(matched)}/{len(words)} words found in context"
    else:
        missing = [w for w in words if w.lower() not in context_lower]
        return False, f"words not in context: {missing} ({len(matched)}/{len(words)} matched)"


def _llm_check_grounding(
    context: str,
    tool_name: str,
    arguments: dict,
    llm_url: str,
    llm_model: str,
    timeout: int = 30,
) -> tuple[bool, str]:
    """
    Ask an LLM judge (OpenAI-compatible API) whether the tool call arguments
    are grounded in the provided context.

    Returns (is_grounded, explanation).
    """
    args_str = json.dumps(arguments, ensure_ascii=False, indent=2)
    prompt = f"""You are a tool call grounding checker. Your job is to determine whether
the ARGUMENT VALUES in a tool call are supported by and consistent with the given context.

Evaluate ONLY the argument values.
Ignore whether the context mentions the tool name, repository name, API surface,
or other surrounding runtime details unless an argument value directly depends on them.
If a value is a trivial reordering or paraphrase of context content, treat it as grounded.
Mark HALLUCINATED only when a concrete value is contradicted by the context or invents
a specific detail (such as a name, repo, identifier, date, count, or limit) not supported there.

Context (what the user/system actually said or provided):
---
{context}
---

Tool call being evaluated:
  Tool name: {tool_name}
  Arguments: {args_str}

Answer with exactly one of:
GROUNDED - if all argument values are supported by or clearly derivable from the context
HALLUCINATED - if any argument value contradicts the context or invents unsupported specifics

Then on the next line, briefly explain why (one sentence).

Your response:"""

    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode()
    # Support both /v1/chat/completions (OpenAI) and /api/chat (Ollama native)
    url = llm_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            content = result["choices"][0]["message"]["content"].strip()
            first_line = content.split("\n")[0].upper()
            explanation = content.split("\n")[1].strip() if "\n" in content else content
            is_grounded = "HALLUCINATED" not in first_line
            return is_grounded, explanation
    except URLError as e:
        raise ConnectionError(f"Cannot reach LLM at {url}: {e.reason}") from e
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise ValueError(f"Unexpected LLM response format: {e}") from e


class ToolCallGroundingScorer(BaseScorer):
    """
    Checks whether tool call argument *values* are grounded in the provided context.

    This is the semantic complement to ToolCallScorer (which checks schema/types).
    Together they cover both structural and semantic hallucination in tool calls.

    Args:
        context:    The conversation context / retrieved documents that the agent
                    should be drawing from. Can be a string or list of strings.
        backend:    "keyword" (default, zero-deps heuristic) or "llm" (LLM judge).
        llm_url:    Base URL for OpenAI-compatible API (used when backend="llm").
                    Examples: "http://localhost:11434/v1" (ollama),
                              "https://api.openai.com/v1" (OpenAI).
        llm_model:  Model name for LLM judge (e.g. "qwen3:0.6b", "gpt-4o-mini").
        llm_timeout: Request timeout in seconds for LLM calls.
        dataset:    Optional dataset tag for the resulting Eval.
        fail_on_llm_error: If True (default), treat LLM connectivity errors as fail.
                    If False, return a "pass" with a warning note instead.
    """

    def __init__(
        self,
        context: Union[str, List[str]] = "",
        backend: str = "keyword",
        llm_url: str = "http://localhost:11434/v1",
        llm_model: str = "qwen3:0.6b",
        llm_timeout: int = 30,
        dataset: Optional[str] = None,
        fail_on_llm_error: bool = True,
    ):
        if isinstance(context, list):
            self.context = "\n\n".join(context)
        else:
            self.context = context
        if backend not in ("keyword", "llm"):
            raise ValueError(f"backend must be 'keyword' or 'llm', got {backend!r}")
        self.backend = backend
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout
        self.dataset = dataset
        self.fail_on_llm_error = fail_on_llm_error

    @property
    def eval_key(self) -> str:
        return "tool_call_grounding"

    @property
    def scorer_name(self) -> str:
        return f"grounding-{self.backend}"

    def check(self, tool_call: dict) -> tuple[bool, List[str]]:
        """
        Check a single tool call for grounding.

        Returns (is_grounded, list_of_issue_strings).
        """
        call = normalize_tool_call(tool_call)
        name = call.get("name", "<unknown>")
        args = call.get("arguments", {}) or {}

        if not self.context.strip():
            return True, []  # No context provided — nothing to check against

        if self.backend == "keyword":
            issues = []
            flat = _flatten_args(args)
            if not flat:
                return True, []  # No string args to check

            for path, value in flat:
                if isinstance(value, str):
                    if _is_operational_string_arg(path, value):
                        continue
                    grounded, reason = _keyword_is_grounded(value, self.context)
                    issue_prefix = f"Argument '{path}' value {value!r}"
                else:
                    if _is_operational_numeric_arg(path):
                        continue
                    grounded, reason = _number_is_grounded(value, self.context)
                    issue_prefix = f"Argument '{path}' numeric value {value!r}"
                if not grounded:
                    issues.append(f"{issue_prefix} may be hallucinated — {reason}")
            return len(issues) == 0, issues

        elif self.backend == "llm":
            try:
                is_grounded, explanation = _llm_check_grounding(
                    context=self.context,
                    tool_name=name,
                    arguments=args,
                    llm_url=self.llm_url,
                    llm_model=self.llm_model,
                    timeout=self.llm_timeout,
                )
                if is_grounded:
                    return True, []
                else:
                    return False, [f"LLM judge: {explanation}"]
            except (ConnectionError, ValueError) as e:
                if self.fail_on_llm_error:
                    return False, [f"LLM grounding check failed: {e}"]
                else:
                    return True, []  # Soft fail

        return True, []

    def score(
        self,
        run_id: str,
        tool_call: Optional[dict] = None,
        tool_calls: Optional[List[dict]] = None,
        context: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Eval:
        """
        Score one or more tool calls for semantic grounding.

        Args:
            run_id:     The run identifier.
            tool_call:  A single tool call dict (any supported format).
            tool_calls: A list of tool call dicts (any supported format).
            context:    Override the instance context for this call only.

        Returns an Eval with:
            score   = fraction of grounded calls (1.0 = all grounded)
            verdict = "pass" if all grounded, "fail" otherwise
            notes   = human-readable summary
            meta    = structured breakdown
        """
        # Allow per-call context override
        if context is not None:
            original_context = self.context
            if isinstance(context, list):
                self.context = "\n\n".join(context)
            else:
                self.context = context
        else:
            original_context = None

        try:
            calls: List[dict] = []
            if tool_call is not None:
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
                    notes="No tool call provided to check.",
                )

            all_issues: List[str] = []
            grounded_count = 0

            for call in calls:
                is_grounded, issues = self.check(call)
                if is_grounded:
                    grounded_count += 1
                else:
                    normalized_name = normalize_tool_call(call).get("name", "<unknown>")
                    all_issues.extend(f"[{normalized_name}] {issue}" for issue in issues)

            total = len(calls)
            pass_rate = grounded_count / total

            if not all_issues:
                notes = f"All {total} tool call(s) appear grounded in context."
            else:
                notes = f"{grounded_count}/{total} grounded. Issues: " + "; ".join(all_issues)

            return Eval(
                run_id=run_id,
                eval_key=self.eval_key,
                scorer=self.scorer_name,
                score=round(pass_rate, 4),
                verdict="pass" if not all_issues else "fail",
                dataset=self.dataset,
                notes=notes,
                meta={
                    "total_calls": total,
                    "grounded_calls": grounded_count,
                    "issues": all_issues,
                    "backend": self.backend,
                },
            )
        finally:
            if original_context is not None:
                self.context = original_context
