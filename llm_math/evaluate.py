"""Validate a JSONL/JSON results file against an expected-answer file.

Mirrors `math_prove/evaluate.py` argparse surface. Wraps the vendored
`validate_results` and `write_validation_report` so the bare-LLM framework
shares the upstream 2-tier evaluator (local equivalence + multi-model LLM judge).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_math.vendor import (
    LLMJudgeConfig,
    validate_results,
    write_validation_report,
)


def print_score_summary(row: Dict[str, Any], prefix: str = "") -> None:
    accuracy = row.get("answer_accuracy")
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.2%}"
    judge_accuracy = row.get("llm_judge_accuracy")
    judge_text = "" if judge_accuracy is None else f" | llm_judge={judge_accuracy:.2%}"
    schema_rate = row.get("schema_valid_rate", 0.0)
    print(
        f"{prefix}Accuracy={accuracy_text} "
        f"({row.get('answer_correct', 0)}/{row.get('answer_checked', 0)} checked) | "
        f"schema_valid={schema_rate:.2%} | "
        f"preflight_issues={row.get('preflight_issue_count', 0)}"
        f"{judge_text}"
    )


def build_llm_judge_config(args: argparse.Namespace) -> Optional[LLMJudgeConfig]:
    if not getattr(args, "llm_judge", False):
        return None
    config = LLMJudgeConfig(
        enabled=True,
        timeout=getattr(args, "judge_timeout", 60),
        judge_all=getattr(args, "llm_judge_all", False),
    )

    def _add(model: Optional[str], key: Optional[str], base: Optional[str]) -> None:
        if model and key:
            config.add_judge(model, key, base or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1/chat/completions"))

    _add(
        getattr(args, "judge_model", None),
        getattr(args, "judge_api_key", None) or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MODEL_API_KEY"),
        getattr(args, "judge_api_base", None) or os.environ.get("LLM_API_BASE"),
    )
    _add(
        getattr(args, "judge_model2", None),
        getattr(args, "judge_api_key2", None) or os.environ.get("MODEL2_API_KEY"),
        getattr(args, "judge_api_base2", None) or os.environ.get("LLM_API_BASE"),
    )
    _add(
        getattr(args, "judge_model3", None),
        getattr(args, "judge_api_key3", None) or os.environ.get("MODEL3_API_KEY"),
        getattr(args, "judge_api_base3", None) or os.environ.get("LLM_API_BASE"),
    )

    if not config.judges:
        return None
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate a bare-LLM results file against expected answers.")
    p.add_argument("--results", type=str, required=True, help="Results JSON or JSONL file.")
    p.add_argument("--expected", type=str, required=True, help="Expected answers JSONL file.")
    p.add_argument("--report", type=str, default=None,
                   help="Output report path. Default: <results_dir>/validation_report.json.")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Optional directory of per-problem logs; missing logs are flagged.")
    p.add_argument("--strict-expected-ids", dest="strict_expected_ids", action="store_true",
                   help="Error if any expected problem_id is missing from results.")
    p.add_argument("--ignore-missing-expected", dest="strict_expected_ids", action="store_false",
                   help="Don't error on missing IDs (default).")
    p.set_defaults(strict_expected_ids=False)

    p.add_argument("--llm-judge", action="store_true", help="Enable LLM-judge tier.")
    p.add_argument("--llm-judge-all", action="store_true", help="Run judges even when local eq passes.")
    p.add_argument("--judge-timeout", type=int, default=60, help="Per-judge-call timeout (s).")
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

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    expected_path = Path(args.expected)
    if not expected_path.exists():
        raise SystemExit(f"Expected file not found: {expected_path}")

    if args.report:
        report_path = Path(args.report)
    else:
        report_path = results_path.parent / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    judge = build_llm_judge_config(args)
    log_dir = Path(args.log_dir) if args.log_dir else None

    report = validate_results(
        result_path=str(results_path),
        expected_path=str(expected_path),
        log_dir=str(log_dir) if log_dir else None,
        strict_expected_ids=args.strict_expected_ids,
        llm_judge=judge,
    )
    write_validation_report(report, str(report_path))
    print_score_summary(report.to_dict(), prefix="[validate] ")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
