"""nullwatch-py CLI — convenience wrapper over NullwatchClient.

Available commands:

    nullwatch-py ping
    nullwatch-py ingest-span span.json
    nullwatch-py ingest-eval eval.json
    nullwatch-py run <run-id>

All commands respect the ``NULLWATCH_URL`` and ``NULLWATCH_API_KEY``
environment variables.
"""

from __future__ import annotations

import json
import sys
from typing import Optional


def _make_client(base_url: Optional[str] = None) -> "NullwatchClient":
    from .client import NullwatchClient
    return NullwatchClient(base_url=base_url)


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_ping(args: list[str]) -> int:
    """Check if the nullwatch service is reachable."""
    base_url = args[0] if args else None
    client = _make_client(base_url)
    try:
        result = client.health()
        print(f"OK  {client.base_url}")
        if result:
            _print_json(result)
        return 0
    except Exception as exc:
        print(f"FAIL  {client.base_url}: {exc}", file=sys.stderr)
        return 1


def cmd_ingest_span(args: list[str]) -> int:
    """Ingest a span from a JSON file.

    Usage: nullwatch-py ingest-span <span.json>
    """
    if not args:
        print("Usage: nullwatch-py ingest-span <span.json>", file=sys.stderr)
        return 2

    path = args[0]
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error reading {path}: {exc}", file=sys.stderr)
        return 1

    from .client import NullwatchClient
    from .models import Span

    client = NullwatchClient()
    span = Span(
        run_id=data.get("run_id", "cli-run"),
        operation=data.get("operation", "cli.span"),
        **{k: v for k, v in data.items() if k not in ("run_id", "operation")},
    )
    try:
        result = client.ingest_span(span)
        print("Span ingested.")
        if result:
            _print_json(result)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_ingest_eval(args: list[str]) -> int:
    """Ingest an eval from a JSON file.

    Usage: nullwatch-py ingest-eval <eval.json>
    """
    if not args:
        print("Usage: nullwatch-py ingest-eval <eval.json>", file=sys.stderr)
        return 2

    path = args[0]
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error reading {path}: {exc}", file=sys.stderr)
        return 1

    from .client import NullwatchClient
    from .models import Eval

    client = NullwatchClient()
    eval_ = Eval(
        run_id=data.get("run_id", "cli-run"),
        eval_key=data.get("eval_key", "cli.eval"),
        score=float(data.get("score", 0.0)),
        verdict=data.get("verdict", "pass"),
        **{
            k: v
            for k, v in data.items()
            if k not in ("run_id", "eval_key", "score", "verdict")
        },
    )
    try:
        result = client.ingest_eval(eval_)
        print("Eval ingested.")
        if result:
            _print_json(result)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_run(args: list[str]) -> int:
    """Print a run summary.

    Usage: nullwatch-py run <run-id>
    """
    if not args:
        print("Usage: nullwatch-py run <run-id>", file=sys.stderr)
        return 2

    run_id = args[0]
    client = _make_client()
    try:
        summary = client.get_run(run_id)
        if summary is None:
            print(f"Run '{run_id}' not found.", file=sys.stderr)
            return 1
        _print_json(
            {
                "run_id": summary.run_id,
                "span_count": summary.span_count,
                "eval_count": summary.eval_count,
                "error_count": summary.error_count,
                "verdict": summary.verdict,
                "total_cost_usd": summary.total_cost_usd,
                "total_duration_ms": summary.total_duration_ms,
            }
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


_COMMANDS = {
    "ping": cmd_ping,
    "ingest-span": cmd_ingest_span,
    "ingest-eval": cmd_ingest_eval,
    "run": cmd_run,
}


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the ``nullwatch-py`` CLI."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(
            "Usage: nullwatch-py <command> [args]\n\n"
            "Commands:\n"
            "  ping              Check service connectivity\n"
            "  ingest-span FILE  Ingest a span from a JSON file\n"
            "  ingest-eval FILE  Ingest an eval from a JSON file\n"
            "  run RUN_ID        Print a run summary\n"
        )
        sys.exit(0)

    command = argv[0]
    rest = argv[1:]

    handler = _COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command: {command!r}.  Run 'nullwatch-py --help' for usage.", file=sys.stderr)
        sys.exit(2)

    sys.exit(handler(rest))


if __name__ == "__main__":
    main()
