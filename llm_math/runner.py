"""Parallel batch runner for the bare-LLM framework.

Adapted from `math_prove/run_parallel_batch.py`. The original wraps a
`MathSolverAgent`; this version wraps any object with a duck-typed
`solve(problem_text, problem_id, raw_metadata) -> MathSolution` and
`last_run_log` attribute — `BareLLMSolver` qualifies, and so do test
fakes.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_math.io import (
    fallback_solution,
    load_problems,
    load_and_validate_results,
    read_existing_results,
    solution_to_json,
    write_problem_log,
)
from llm_math.vendor import MathSolution


def _safe_print(*args: Any, **kwargs: Any) -> None:
    try:
        print(*args, **kwargs)
    except ValueError:
        try:
            sys.stderr.write(" ".join(str(arg) for arg in args) + "\n")
        except Exception:
            pass


def run_parallel_batch(
    solver_factory: Callable[[], Any],
    problems: List[Dict[str, Any]],
    output_path: Path,
    log_dir: Path,
    workers: int = 3,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run `problems` through `solver_factory()` concurrently.

    Each worker thread builds its own solver instance via `solver_factory()`
    (so per-thread state — e.g. rate limiters, IPython sessions — is isolated).
    The output JSONL is appended incrementally; per-problem logs land in
    `log_dir`. When `resume=True`, problems whose IDs already appear in
    `output_path` are skipped.

    Returns a summary dict: total / processed / skipped / fallback counts
    and timing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Dict[str, Any]] = {}
    if resume and output_path.exists():
        existing = read_existing_results(output_path)

    mode = "a" if resume and output_path.exists() else "w"
    output_handle = output_path.open(mode, encoding="utf-8")

    total = len(problems)
    processed = 0
    skipped = 0
    fallback_count = 0
    started_at = time.time()
    pending: List[Tuple[Any, Dict[str, Any]]] = []

    _safe_print(f"Loaded {total} problems. Writing JSONL to {output_path}")
    _safe_print(f"Writing per-problem logs to {log_dir}")
    _safe_print(f"Workers: {workers}, resume: {resume}")

    for index, record in enumerate(problems, start=1):
        pid = record["problem_id"]
        if resume and pid in existing:
            skipped += 1
            _safe_print(f"[{index}/{total}] skip {pid} (resume)")
            continue
        pending.append((index, record))

    def solve_one(payload: Tuple[int, Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any], bool]:
        index, record = payload
        pid = record["problem_id"]
        text = record["problem_text"]
        metadata = record.get("raw_metadata", {})
        solver = solver_factory()
        item_start = time.time()
        used_fallback = False
        try:
            solution = solver.solve(text, pid, raw_metadata=metadata)
            run_log = dict(getattr(solver, "last_run_log", {}) or {})
            run_log.setdefault("latency_seconds", round(time.time() - item_start, 3))
        except Exception as exc:
            used_fallback = True
            solution = fallback_solution(pid, f"{type(exc).__name__}: {exc}")
            run_log = {
                "problem_id": pid,
                "raw_problem": text,
                "raw_metadata": metadata,
                "api_status": "parallel_batch_fallback_exception",
                "exception": repr(exc),
                "latency_seconds": round(time.time() - item_start, 3),
                "final_json": solution.model_dump(mode="json"),
            }
        line = solution_to_json(solution, indent=None)
        return pid, line, run_log, used_fallback

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(solve_one, p) for p in pending]
            for fut in as_completed(futures):
                pid, line, run_log, used_fallback = fut.result()
                output_handle.write(line + "\n")
                output_handle.flush()
                write_problem_log(log_dir, pid, run_log)
                processed += 1
                if used_fallback or '"unable_to_determine"' in line:
                    fallback_count += 1
                _safe_print(
                    f"  done {pid} | fallback={used_fallback} | "
                    f"latency={run_log.get('latency_seconds', '?')}s"
                )

    output_handle.close()
    all_results, schema_errors = load_and_validate_results(output_path)
    summary = {
        "input_total": total,
        "processed_this_run": processed,
        "skipped_by_resume": skipped,
        "results_in_jsonl": len(all_results),
        "schema_error_count": len(schema_errors),
        "schema_errors": schema_errors[:20],
        "fallback_count": fallback_count,
        "output_jsonl": str(output_path),
        "log_dir": str(log_dir),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    _safe_print(
        f"Batch complete. processed={processed} skipped={skipped} "
        f"results={len(all_results)} schema_errors={len(schema_errors)} "
        f"fallback={fallback_count} elapsed={summary['elapsed_seconds']}s"
    )
    return summary


__all__ = ["run_parallel_batch"]
