"""Validate every results.jsonl under a sweep directory tree.

Walks `outputs/*/<model>/<ts>/results.jsonl` and calls `validate_results` on
each one, writing `validation_report.json` next to the JSONL. Designed to
follow `prompt_sweep.py` so a single sweep produces a uniform set of
reports ready for `compare_prompts.py`.

Dirs already containing a `validation_report.json` are skipped unless
`--force` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_math.evaluate import build_llm_judge_config, print_score_summary
from llm_math.vendor import validate_results, write_validation_report


def _iter_result_files(sweep_dir: Path) -> List[Path]:
    """Return every `*/<model>/<ts>/results.jsonl` under `sweep_dir`.

    Three path components between the root and the filename are required
    (prompt / model / ts), matching the layout `llm_math.run` writes.
    """
    if not sweep_dir.exists():
        return []
    return sorted(sweep_dir.glob("*/*/*/results.jsonl"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate every results.jsonl under a sweep directory tree."
    )
    p.add_argument("--sweep-dir", type=str, default="outputs",
                   help="Root directory of the sweep (default: outputs).")
    p.add_argument("--expected", type=str, required=True,
                   help="Expected answers JSONL file.")
    p.add_argument("--force", action="store_true",
                   help="Re-validate even if a validation_report.json already exists.")
    p.add_argument("--summary", type=str, default="sweep_eval_summary.json",
                   help="Sweep-level summary path (lives next to the sweep root).")

    # LLM-judge surface — same flags as `evaluate.py`.
    p.add_argument("--llm-judge", action="store_true")
    p.add_argument("--llm-judge-all", action="store_true")
    p.add_argument("--judge-timeout", type=int, default=60)
    p.add_argument("--judge-model", type=str, default=os.environ.get("JUDGE_MODEL", "deepseek-chat"))
    p.add_argument("--judge-api-key", type=str, default=None)
    p.add_argument("--judge-api-base", type=str, default=None)
    p.add_argument("--judge-model2", type=str, default=None)
    p.add_argument("--judge-api-key2", type=str, default=None)
    p.add_argument("--judge-api-base2", type=str, default=None)
    p.add_argument("--judge-model3", type=str, default=None)
    p.add_argument("--judge-api-key3", type=str, default=None)
    p.add_argument("--judge-api-base3", type=str, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sweep_dir = Path(args.sweep_dir)
    expected_path = Path(args.expected)
    if not expected_path.exists():
        raise SystemExit(f"Expected file not found: {expected_path}")

    result_files = _iter_result_files(sweep_dir)
    if not result_files:
        print(f"No results.jsonl files under {sweep_dir}", file=sys.stderr)
        return 1

    judge = build_llm_judge_config(args)

    per_dir: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for results_path in result_files:
        report_path = results_path.parent / "validation_report.json"
        if report_path.exists() and not args.force:
            skipped.append({"results_jsonl": str(results_path), "reason": "already validated"})
            print(f"[skip] {results_path} (validation_report.json exists; pass --force to override)")
            continue
        try:
            report = validate_results(
                result_path=str(results_path),
                expected_path=str(expected_path),
                log_dir=str(results_path.parent / "logs"),
                strict_expected_ids=False,
                llm_judge=judge,
            )
            write_validation_report(report, str(report_path))
            rdict = report.to_dict()
            per_dir.append({
                "results_jsonl": str(results_path),
                "validation_report": str(report_path),
                "label_hint": str(results_path.parent.parent.parent.name),
                "model": str(results_path.parent.parent.name),
                "ts": str(results_path.parent.name),
                "local_acc": rdict.get("answer_accuracy"),
                "judge_acc": rdict.get("llm_judge_accuracy"),
                "schema_valid_rate": rdict.get("schema_valid_rate"),
            })
            print_score_summary(rdict, prefix=f"[validate] {results_path.parent.parent.parent.name}/")
        except Exception as exc:
            failed.append({
                "results_jsonl": str(results_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[fail] {results_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "sweep_dir": str(sweep_dir),
        "expected_path": str(expected_path),
        "judge_enabled": bool(judge and getattr(judge, "enabled", False)),
        "validated": per_dir,
        "skipped": skipped,
        "failed": failed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sweep-eval] {len(per_dir)} validated, {len(skipped)} skipped, "
          f"{len(failed)} failed; summary → {summary_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
