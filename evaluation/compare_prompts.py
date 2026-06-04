"""N-prompt comparison report.

Compare two or more results JSONL files (typically produced by running the
same model over different prompt templates) and emit a single comparison
report. Built specifically for prompt A/B experiments — for bare-vs-agent
comparisons use `evaluation.compare`.

The report has four top-level sections:

  per_prompt:    one row per (results, label) pair with local / judge accuracy,
                 fallback count, median latency, schema-validity rate.
  deltas:        an N×N accuracy matrix `accuracy_matrix[i][j]` = local_acc_i
                 − local_acc_j (signed pp), for quick at-a-glance comparison.
  per_problem:   for every problem_id, the winner label(s) — empty list means
                 no prompt got it right (locally).
  winner:        {problem_id: [label, ...]} lookup table; consumers can
                 diff this against a single baseline to find regressions.

When the LLM-judge tier is enabled, `judge_acc` is the multi-judge majority
accuracy and `winner` is determined by `llm_judge_correct`. When the judge is
disabled, both fall back to local equivalence.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.evaluate import build_llm_judge_config
from shared.vendor import validate_results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare N prompt-template results JSONL files and emit a comparison report."
    )
    p.add_argument(
        "--results",
        action="append",
        type=str,
        default=[],
        required=True,
        help="Path to a results JSONL. Pass once per prompt you want to compare "
        "(e.g. --results a.jsonl --results b.jsonl).",
    )
    p.add_argument(
        "--label",
        action="append",
        type=str,
        default=[],
        help="Display label for the corresponding --results entry. Must be the "
        "same length as --results (e.g. --label A --label B). Defaults to "
        "the file stem.",
    )
    p.add_argument("--expected", type=str, required=True,
                   help="Expected answers JSONL file.")
    p.add_argument("--report", type=str, default="prompt_comparison_report.json",
                   help="Output JSON report path.")
    p.add_argument("--markdown", action="store_true",
                   help="Also emit a markdown table next to the JSON report.")
    p.add_argument("--log-dir", action="append", type=str, default=[],
                   help="Optional log dir per results (parallel to --results). "
                   "Use when the original run wrote per-problem logs.")

    # LLM-judge surface — mirrors evaluation.evaluate.build_llm_judge_config.
    p.add_argument("--llm-judge", action="store_true", help="Enable LLM-judge tier.")
    p.add_argument("--llm-judge-all", action="store_true",
                   help="Run judges even when local eq passes.")
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


def _resolve_labels(results: Sequence[str], labels: Sequence[str]) -> List[str]:
    if labels and len(labels) != len(results):
        raise SystemExit(
            f"--results ({len(results)}) and --label ({len(labels)}) counts disagree; "
            "pass one --label per --results, or omit --label entirely."
        )
    if labels:
        return list(labels)
    return [Path(r).stem for r in results]


def _median_latency_from_logs(log_dir: Optional[str]) -> Optional[float]:
    """Read per-problem logs and return median `latency_seconds` (None if absent)."""
    if not log_dir:
        return None
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return None
    latencies: List[float] = []
    for log_file in log_path.glob("*.json"):
        try:
            payload = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        latency = payload.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
    if not latencies:
        return None
    return round(statistics.median(latencies), 3)


def _load_results_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _per_prompt_row(
    label: str,
    results_path: Path,
    report_dict: Dict[str, Any],
    log_dir: Optional[str],
) -> Dict[str, Any]:
    """Build the per_prompt entry. `report_dict` is the output of `ValidationReport.to_dict()`."""
    rows = _load_results_jsonl(results_path)
    fallback = sum(1 for r in rows if str(r.get("answer", "")).strip() == "unable_to_determine")
    fallback_rate = fallback / len(rows) if rows else 0.0
    return {
        "label": label,
        "results_path": str(results_path),
        "log_dir": str(log_dir) if log_dir else None,
        "total": report_dict.get("total", 0),
        "schema_valid_rate": report_dict.get("schema_valid_rate", 0.0),
        "local_acc": report_dict.get("answer_accuracy"),
        "judge_acc": report_dict.get("llm_judge_accuracy"),
        "local_correct": report_dict.get("answer_correct", 0),
        "local_checked": report_dict.get("answer_checked", 0),
        "judge_correct": report_dict.get("llm_judge_correct", 0),
        "judge_checked": report_dict.get("llm_judge_checked", 0),
        "fallback": fallback,
        "fallback_rate": round(fallback_rate, 4),
        "median_latency_seconds": _median_latency_from_logs(log_dir),
    }


def _accuracy_matrix(per_prompt: List[Dict[str, Any]]) -> List[List[Optional[float]]]:
    """N×N matrix of (row local_acc − col local_acc) in percentage points.

    The diagonal is 0.0; off-diagonal entries are None when either side has
    no local accuracy to compare (e.g. zero answer_checked).
    """
    n = len(per_prompt)
    matrix: List[List[Optional[float]]] = []
    for i, row_i in enumerate(per_prompt):
        row: List[Optional[float]] = []
        acc_i = row_i.get("local_acc")
        for j, row_j in enumerate(per_prompt):
            if i == j:
                row.append(0.0)
                continue
            acc_j = row_j.get("local_acc")
            if acc_i is None or acc_j is None:
                row.append(None)
            else:
                row.append(round((acc_i - acc_j) * 100, 2))
        matrix.append(row)
    return matrix


def _per_problem_winners(
    per_prompt: List[Dict[str, Any]],
    report_dicts: List[Dict[str, Any]],
    judge_enabled: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """For every problem_id seen in any prompt's report, list the labels that got it right.

    Returns a list of per-problem rows (one per pid) plus a `winner` lookup
    table mapping pid -> [label, ...] filtered down to "any prompt got it
    right" so downstream diffs only see real disagreements.
    """
    per_problem: Dict[str, Dict[str, Any]] = {}
    for label, rdict in zip([p["label"] for p in per_prompt], report_dicts):
        for item in rdict.get("items", []):
            pid = item.get("problem_id")
            if not pid:
                continue
            slot = per_problem.setdefault(
                pid,
                {
                    "problem_id": pid,
                    "per_prompt_correct": {},
                },
            )
            if judge_enabled and item.get("llm_judge_correct") is not None:
                slot["per_prompt_correct"][label] = bool(item["llm_judge_correct"])
            else:
                slot["per_prompt_correct"][label] = bool(item.get("answer_equivalent"))

    rows: List[Dict[str, Any]] = []
    winner_map: Dict[str, List[str]] = {}
    for pid, slot in per_problem.items():
        winners = [
            label for label, ok in slot["per_prompt_correct"].items() if ok
        ]
        if winners:
            winner_map[pid] = winners
        rows.append({
            "problem_id": pid,
            "winner": winners,
            "per_prompt_correct": slot["per_prompt_correct"],
        })
    rows.sort(key=lambda r: r["problem_id"])
    return rows, winner_map


def build_report(
    results: Sequence[str],
    labels: Sequence[str],
    expected_path: str,
    log_dirs: Sequence[Optional[str]],
    judge,
) -> Dict[str, Any]:
    """Validate every results file, then assemble the comparison report."""
    if len(set(labels)) != len(labels):
        raise SystemExit(f"--label values must be unique; got {list(labels)}")

    per_prompt: List[Dict[str, Any]] = []
    report_dicts: List[Dict[str, Any]] = []
    for label, rpath, log_dir in zip(labels, results, log_dirs):
        report = validate_results(
            result_path=rpath,
            expected_path=expected_path,
            log_dir=log_dir,
            strict_expected_ids=False,
            llm_judge=judge,
        )
        rdict = report.to_dict()
        report_dicts.append(rdict)
        per_prompt.append(_per_prompt_row(label, Path(rpath), rdict, log_dir))

    judge_enabled = judge is not None and getattr(judge, "enabled", False)
    per_problem, winner_map = _per_problem_winners(per_prompt, report_dicts, judge_enabled)

    return {
        "labels": list(labels),
        "results_paths": [str(Path(r)) for r in results],
        "expected_path": str(Path(expected_path)),
        "judge_enabled": judge_enabled,
        "per_prompt": per_prompt,
        "deltas": {
            "accuracy_matrix": _accuracy_matrix(per_prompt),
            "matrix_labels": list(labels),
        },
        "per_problem": per_problem,
        "winner": winner_map,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    """Render a small markdown table summarizing the per-prompt rows."""
    lines: List[str] = []
    lines.append(f"# Prompt comparison ({len(report['per_prompt'])} runs)")
    lines.append("")
    lines.append(
        "| label | total | local_acc | judge_acc | fallback | median_latency (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report["per_prompt"]:
        local = row.get("local_acc")
        judge_acc = row.get("judge_acc")
        median = row.get("median_latency_seconds")
        lines.append(
            "| {label} | {total} | {local} | {judge} | {fb} | {med} |".format(
                label=row["label"],
                total=row["total"],
                local=("n/a" if local is None else f"{local:.2%}"),
                judge=("n/a" if judge_acc is None else f"{judge_acc:.2%}"),
                fb=f"{row['fallback']} ({row['fallback_rate']:.1%})",
                med=("n/a" if median is None else f"{median:.2f}"),
            )
        )
    lines.append("")
    matrix = report["deltas"]["accuracy_matrix"]
    if len(matrix) > 1:
        lines.append("## Local-accuracy deltas (rows − cols, percentage points)")
        labels = report["deltas"]["matrix_labels"]
        header = "| row \\ col | " + " | ".join(labels) + " |"
        sep = "|---" * (len(labels) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for i, row_label in enumerate(labels):
            cells = []
            for j, val in enumerate(matrix[i]):
                if val is None:
                    cells.append("n/a")
                else:
                    sign = "+" if val > 0 else ("" if val == 0 else "")
                    cells.append(f"{sign}{val:.2f}pp")
            lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.results:
        raise SystemExit("at least one --results is required")

    labels = _resolve_labels(args.results, args.label)
    if args.log_dir and len(args.log_dir) == len(args.results):
        log_dirs: List[Optional[str]] = list(args.log_dir)
    elif args.log_dir:
        raise SystemExit(
            f"--log-dir count ({len(args.log_dir)}) does not match --results count "
            f"({len(args.results)}); pass one per prompt or omit."
        )
    else:
        log_dirs = [None] * len(args.results)

    expected_path = Path(args.expected)
    if not expected_path.exists():
        raise SystemExit(f"Expected file not found: {expected_path}")
    for r in args.results:
        if not Path(r).exists():
            raise SystemExit(f"Results file not found: {r}")

    judge = build_llm_judge_config(args)

    report = build_report(
        results=args.results,
        labels=labels,
        expected_path=str(expected_path),
        log_dirs=log_dirs,
        judge=judge,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {report_path}")

    # Print the per-prompt summary table to stdout for quick eyeballing.
    sys.stdout.write(render_markdown(report))
    sys.stdout.write("\n")

    if args.markdown:
        md_path = report_path.with_suffix(".md")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Markdown written to {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
