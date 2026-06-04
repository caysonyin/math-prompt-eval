"""CLI entrypoint for the bare-LLM math solver.

Mirrors `math_prove/main.py` argparse surface, adapted for the no-scaffolding
single-shot solver.

Default output layout (when `--output` is not given) is

    outputs/{prompt_name}/{model}/{timestamp}/
        results.jsonl
        results.json
        run_summary.json
        logs/

This isolates every prompt-and-model combination on disk so prompt A/B runs
don't clobber each other and `compare_prompts` can glob across the tree.
Passing `--output` explicitly bypasses the nesting — the other three paths
default to `<output>.parent / ...` for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from runtime.solver import BareLLMSolver
from shared.io import (
    SAMPLE_PROBLEMS,
    load_problems,
    resolve_api_config,
)
from runtime.prompts import list_prompts
from runtime.runner import run_parallel_batch
from shared.vendor import (
    fallback_solution,
    solution_to_json,
)


DEFAULT_PROMPT = "naive_v1"
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")
DEFAULT_INPUT = str(
    Path(__file__).resolve().parent.parent.parent
    / "mathsolve-agent"
    / "math_prove"
    / "validation"
    / "core_18_sample.jsonl"
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bare-LLM math solver (single-shot, no scaffolding).")
    p.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT,
                   help="Input JSON/JSONL/CSV/XLSX")
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Incremental JSONL output path. If omitted, the solver writes to "
            "outputs/{prompt_name}/{model}/{timestamp}/results.jsonl. Pass "
            "explicitly to bypass the nested layout."
        ),
    )
    p.add_argument("--results-json", type=str, default=None, help="Merged JSON array output")
    p.add_argument("--log-dir", type=str, default=None, help="Per-problem log directory")
    p.add_argument("--summary", type=str, default=None, help="Run summary JSON path")
    p.add_argument("--base-output-dir", type=str, default="outputs",
                   help="Root directory for the nested {prompt}/{model}/{ts}/ layout.")
    p.add_argument("--run-ts", type=str, default=None,
                   help="Override the timestamp used in the nested output path (e.g. "
                        "for replaying a known run). Default: now() formatted %%Y%%m%%d_%%H%%M%%S.")
    p.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Model name")
    p.add_argument("--api-key", type=str, default=None, help="API key")
    p.add_argument("--api-base", type=str, default=None, help="OpenAI-compatible chat endpoint")
    p.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        choices=[t.name for t in list_prompts()],
        help="Prompt template name. Default: naive_v1. (LLM-extra flag, ignored by upstream.)",
    )
    p.add_argument("--limit", "-n", type=int, default=None, help="Only process first N rows")
    p.add_argument("--resume", action="store_true", help="Skip IDs already in output JSONL")
    p.add_argument(
        "--workers", type=int, default=3,
        help="Thread-pool worker count. (LLM-extra flag.)",
    )
    p.add_argument(
        "--rpm-limit", type=int, default=80,
        help="Per-process RPM cap. (LLM-extra flag.)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=4096,
        help="LLM max_tokens. (LLM-extra flag.)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM temperature. (LLM-extra flag.)",
    )
    p.add_argument(
        "--timeout", type=int, default=120,
        help="HTTP request timeout (s). (LLM-extra flag.)",
    )
    p.add_argument("--demo", action="store_true", help="Run a single demo problem and exit")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip the LLM call entirely; emit a fallback_solution for every row. (LLM-extra flag.)",
    )
    return p


def _safe_dirname(value: str, fallback: str) -> str:
    """Sanitize a value (model name, prompt name) for use as a directory segment."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in value).strip("._")
    return cleaned or fallback


def _default_nested_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Return the four default output paths nested under `outputs/{prompt}/{model}/{ts}/`.

    Used when the caller does not pass `--output`. The `ts` segment is
    `args.run_ts` if provided, otherwise `datetime.now().strftime(...)`.
    """
    prompt_segment = _safe_dirname(args.prompt, "prompt")
    model_segment = _safe_dirname(args.model, "model")
    ts = args.run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.base_output_dir) / prompt_segment / model_segment / ts
    return {
        "output": base / "results.jsonl",
        "results_json": base / "results.json",
        "log_dir": base / "logs",
        "summary": base / "run_summary.json",
    }


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    """Pick the four canonical paths for this run.

    If the user passed `--output`, that path is honored verbatim and the
    other three default to `<output.parent> / <sibling>` (legacy behaviour).
    Otherwise we nest everything under `outputs/{prompt}/{model}/{ts}/`.
    """
    if args.output:
        output_path = Path(args.output)
        log_dir = Path(args.log_dir) if args.log_dir else output_path.parent / "logs"
        results_json_path = (
            Path(args.results_json) if args.results_json else output_path.with_suffix(".json")
        )
        summary_path = (
            Path(args.summary) if args.summary else output_path.parent / "run_summary.json"
        )
    else:
        nested = _default_nested_paths(args)
        output_path = nested["output"]
        log_dir = nested["log_dir"]
        results_json_path = nested["results_json"]
        summary_path = nested["summary"]
    return output_path, log_dir, results_json_path, summary_path


def _make_solver_factory(
    *,
    model: str,
    api_key: Optional[str],
    api_base: Optional[str],
    prompt_name: str,
    rpm_limit: int,
    max_tokens: int,
    temperature: float,
    timeout: int,
    dry_run: bool,
):
    if dry_run:
        def factory():
            return _DryRunSolver()
        return factory

    def factory():
        return BareLLMSolver(
            model=model,
            api_key=api_key,
            api_base=api_base,
            prompt_name=prompt_name,
            rpm_limit=rpm_limit,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    return factory


class _DryRunSolver:
    """Stub used by --dry-run: emits fallback_solution without calling any API."""

    last_run_log: Dict[str, Any] = {}

    def solve(self, problem_text: str, problem_id: str, raw_metadata: Optional[Dict[str, Any]] = None):
        from shared.vendor import MathSolution
        solution = fallback_solution(problem_id, "dry-run")
        self.last_run_log = {
            "problem_id": problem_id,
            "raw_problem": problem_text,
            "raw_metadata": raw_metadata or {},
            "dry_run": True,
            "final_json": solution.model_dump(mode="json"),
        }
        return solution


def run_demo(args: argparse.Namespace) -> None:
    api_key, api_base = resolve_api_config(args)
    sample = SAMPLE_PROBLEMS[0]
    if args.dry_run:
        solver = _DryRunSolver()
    else:
        if not api_key or not api_base:
            raise SystemExit("--demo requires --api-key and --api-base (or env vars).")
        solver = BareLLMSolver(
            model=args.model,
            api_key=api_key,
            api_base=api_base,
            prompt_name=args.prompt,
            rpm_limit=args.rpm_limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
    started = time.time()
    solution = solver.solve(sample["problem_text"], sample["problem_id"])
    elapsed = time.time() - started
    print("=" * 72)
    print(f"Problem [{sample['problem_id']}]: {sample['problem_text']}")
    print(f"Prompt: {args.prompt}, Model: {args.model}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("-" * 72)
    print(solution_to_json(solution))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.demo:
        run_demo(args)
        return 0

    api_key, api_base = resolve_api_config(args)
    if not args.dry_run and (not api_key or not api_base):
        raise SystemExit(
            "API key and base required. Set --api-key/--api-base or env "
            "OPENAI_API_KEY/LLM_API_BASE (or use --dry-run to skip the LLM call)."
        )

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path, log_dir, results_json_path, summary_path = _resolve_paths(args)
    for p in (output_path, log_dir, results_json_path, summary_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    problems = load_problems(str(input_path))
    if args.limit is not None:
        problems = problems[: args.limit]

    factory = _make_solver_factory(
        model=args.model,
        api_key=api_key,
        api_base=api_base,
        prompt_name=args.prompt,
        rpm_limit=args.rpm_limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )

    summary = run_parallel_batch(
        solver_factory=factory,
        problems=problems,
        output_path=output_path,
        log_dir=log_dir,
        workers=args.workers,
        resume=args.resume,
    )

    # Also write a merged JSON array (the convention promised by the
    # `results_json` path; downstream tools like `compare.py` consume it).
    merged: List[Dict[str, Any]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    merged.append(json.loads(line))
                except Exception:
                    continue
    results_json_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary.update(
        {
            "input_path": str(input_path),
            "output_jsonl": str(output_path),
            "output_json": str(results_json_path),
            "log_dir": str(log_dir),
            "model": args.model,
            "prompt": args.prompt,
            "dry_run": args.dry_run,
            "resume": args.resume,
            "ablation": "bare_llm",
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
