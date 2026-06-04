"""Batch-run a set of prompt templates for A/B experiments.

This is the orchestrator for prompt engineering experiments. The workflow is:

    1. `prompt_sweep.py` runs each selected prompt over the same input
       file, sharing a single timestamp so all results land under
       `outputs/{prompt}/{model}/{ts}/` and can be cross-compared by
       `compare_prompts.py`.
    2. `evaluate_sweep.py` walks the same tree and writes a
       `validation_report.json` next to each `results.jsonl`.
    3. `compare_prompts.py --markdown` emits a single side-by-side table.

Per-prompt runs are serial by default (to keep RPM usage predictable with
a single API key); pass `--parallel-prompts` to run multiple prompts in
parallel. Inside each prompt, the workers in `run.py` still parallelize
problem-level calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_math.io import resolve_api_config
from llm_math.prompts import list_prompts
from llm_math.run import _resolve_paths, main as run_main, build_arg_parser as run_build_arg_parser


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run several prompt templates over the same input and stage results for compare_prompts."
    )
    # Pass-through selection ----------------------------------------------------------
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated subset of prompt names. Default: every registered prompt.")
    p.add_argument("--input", type=str, required=True, help="Input JSON/JSONL/CSV/XLSX path.")
    p.add_argument("--base-output-dir", type=str, default="outputs",
                   help="Root directory for the nested {prompt}/{model}/{ts}/ layout (default: outputs).")

    # Run knobs (mirror of run.py for ergonomics) ---------------------------------------
    p.add_argument("--model", type=str, default=os.environ.get("MODEL_NAME", "gpt-4o-mini"))
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--api-base", type=str, default=None)
    p.add_argument("--workers", type=int, default=3, help="Workers per prompt (intra-prompt).")
    p.add_argument("--rpm-limit", type=int, default=80)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Run every selected prompt in dry-run mode; emit fallback_solution rows without API calls.")

    # Sweep orchestration --------------------------------------------------------------
    p.add_argument("--parallel-prompts", action="store_true",
                   help="Run multiple prompts in parallel (default: serial across prompts).")
    p.add_argument("--run-ts", type=str, default=None,
                   help="Override the shared timestamp; useful for replaying a known sweep.")
    p.add_argument("--summary", type=str, default="sweep_summary.json",
                   help="Path to the sweep-level summary (lives next to per-prompt dirs).")
    return p


def _select_prompts(only: Optional[str]) -> List[str]:
    available = [t.name for t in list_prompts()]
    if not only:
        return available
    wanted = [name.strip() for name in only.split(",") if name.strip()]
    missing = [name for name in wanted if name not in available]
    if missing:
        raise SystemExit(
            f"--only references unknown prompt(s) {missing}; available: {available}"
        )
    return wanted


def _run_single_prompt(
    prompt_name: str,
    args: argparse.Namespace,
    run_ts: str,
) -> Dict[str, Any]:
    """Run a single prompt via `llm_math.run.main` and return its summary dict.

    We invoke `run.main(...)` directly (rather than subprocess) so the sweep
    shares the same Python process, catches exceptions in-band, and
    surfaces them as `error` in the per-prompt summary without aborting
    the whole sweep.
    """
    argv = [
        "--input", str(args.input),
        "--prompt", prompt_name,
        "--model", args.model,
        "--workers", str(args.workers),
        "--rpm-limit", str(args.rpm_limit),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--timeout", str(args.timeout),
        "--base-output-dir", str(args.base_output_dir),
        "--run-ts", run_ts,
    ]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    if args.api_base:
        argv += ["--api-base", args.api_base]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.resume:
        argv += ["--resume"]
    if args.dry_run:
        argv += ["--dry-run"]

    try:
        rc = run_main(argv)
        if rc != 0:
            return {
                "prompt": prompt_name,
                "error": f"run.main returned {rc}",
                "rc": rc,
            }
        # Re-derive the canonical output paths so the sweep summary can point
        # at them without re-running.
        from types import SimpleNamespace
        fake_args = SimpleNamespace(
            prompt=prompt_name,
            model=args.model,
            base_output_dir=args.base_output_dir,
            run_ts=run_ts,
            output=None,
            results_json=None,
            log_dir=None,
            summary=None,
        )
        out, log_dir, results_json, summary = _resolve_paths(fake_args)
        summary_dict: Dict[str, Any] = {
            "prompt": prompt_name,
            "output_jsonl": str(out),
            "results_json": str(results_json),
            "log_dir": str(log_dir),
            "summary_path": str(summary),
            "rc": 0,
        }
        if summary.exists():
            try:
                summary_dict["run_summary"] = json.loads(summary.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover — defensive
                summary_dict["run_summary"] = {"_read_error": repr(exc)}
        return summary_dict
    except SystemExit as exc:
        return {
            "prompt": prompt_name,
            "error": f"SystemExit({exc.code})",
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:
        return {
            "prompt": prompt_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    prompts = _select_prompts(args.only)
    if not prompts:
        raise SystemExit("no prompts to run (registry is empty or --only filter is too tight)")

    if not Path(args.input).exists():
        raise SystemExit(f"Input file not found: {args.input}")

    run_ts = args.run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[sweep] ts={run_ts} prompts={prompts} parallel={args.parallel_prompts} dry_run={args.dry_run}")

    results: List[Dict[str, Any]] = []
    if args.parallel_prompts:
        with ThreadPoolExecutor(max_workers=min(8, len(prompts))) as pool:
            futures = {
                pool.submit(_run_single_prompt, name, args, run_ts): name
                for name in prompts
            }
            for fut in as_completed(futures):
                name = futures[fut]
                outcome = fut.result()
                if "error" in outcome:
                    print(f"[sweep] {name}: ERROR — {outcome['error']}", file=sys.stderr)
                else:
                    print(f"[sweep] {name}: done → {outcome.get('output_jsonl')}")
                results.append(outcome)
    else:
        for name in prompts:
            outcome = _run_single_prompt(name, args, run_ts)
            if "error" in outcome:
                print(f"[sweep] {name}: ERROR — {outcome['error']}", file=sys.stderr)
            else:
                print(f"[sweep] {name}: done → {outcome.get('output_jsonl')}")
            results.append(outcome)

    # Build the sweep-level summary; sort prompts to keep output stable.
    results.sort(key=lambda r: r.get("prompt", ""))
    succeeded = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    summary = {
        "ts": run_ts,
        "input": str(args.input),
        "base_output_dir": str(args.base_output_dir),
        "model": args.model,
        "prompts_requested": prompts,
        "prompts_succeeded": [r["prompt"] for r in succeeded],
        "prompts_failed": [r["prompt"] for r in failed],
        "dry_run": args.dry_run,
        "parallel_prompts": args.parallel_prompts,
        "per_prompt": results,
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sweep] {len(succeeded)}/{len(prompts)} prompts completed; summary → {summary_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
