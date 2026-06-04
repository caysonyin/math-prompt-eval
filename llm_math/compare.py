"""Side-by-side comparison of bare-LLM vs. upstream MathSolverAgent results.

Runs the same 2-tier evaluator on both JSONL files (with the same LLMJudgeConfig
when --llm-judge is on) and writes a per-problem delta report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_math.evaluate import build_llm_judge_config, print_score_summary
from llm_math.vendor import ValidationReport, validate_results, write_validation_report


def _to_dict(report: ValidationReport) -> Dict[str, Any]:
    return report.to_dict()


def _read_answers(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read a results JSONL file and return {problem_id: {answer, answer_type, ...}}."""
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    import json
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = str(obj.get("problem_id", "")).strip()
            if pid:
                out[pid] = obj
    return out


def _per_problem(
    bare: ValidationReport,
    agent: Optional[ValidationReport],
    bare_answers: Dict[str, Dict[str, Any]],
    agent_answers: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    bare_items = {it["problem_id"]: it for it in bare.to_dict()["items"]}
    agent_items: Dict[str, Dict[str, Any]] = {}
    if agent is not None:
        agent_items = {it["problem_id"]: it for it in agent.to_dict()["items"]}

    out: List[Dict[str, Any]] = []
    for pid in sorted(set(bare_items) | set(agent_items) | set(bare_answers) | set(agent_answers)):
        b_item = bare_items.get(pid, {})
        a_item = agent_items.get(pid, {})
        b_ans = (bare_answers.get(pid) or {}).get("answer")
        a_ans = (agent_answers.get(pid) or {}).get("answer")
        b_correct = b_item.get("answer_equivalent")
        a_correct = a_item.get("answer_equivalent")
        out.append(
            {
                "problem_id": pid,
                "expected_answer": b_item.get("expected_answer") or a_item.get("expected_answer"),
                "bare_answer": b_ans,
                "bare_local_correct": b_correct,
                "bare_judge_correct": b_item.get("llm_judge_correct"),
                "agent_answer": a_ans,
                "agent_local_correct": a_correct,
                "agent_judge_correct": a_item.get("llm_judge_correct"),
                "delta": (
                    (1 if a_correct else 0) - (1 if b_correct else 0)
                    if a_correct is not None and b_correct is not None
                    else None
                ),
            }
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare bare-LLM results with upstream MathSolverAgent results.")
    p.add_argument("--bare-results", type=str, required=True, help="Bare-LLM JSONL results.")
    p.add_argument("--agent-results", type=str, default=None,
                   help="Upstream MathSolverAgent JSONL results (optional).")
    p.add_argument("--expected", type=str, required=True, help="Expected answers JSONL file.")
    p.add_argument("--prompt", type=str, default="naive_v1", help="Label only — prompt name used for the bare side.")
    p.add_argument("--model", type=str, default="", help="Label only — model name used for the bare side.")
    p.add_argument("--report", type=str, default=None,
                   help="Output comparison report path. Default: <bare_dir>/comparison_report.json.")
    p.add_argument("--log-dir-bare", type=str, default=None)
    p.add_argument("--log-dir-agent", type=str, default=None)
    p.add_argument("--strict-expected-ids", dest="strict_expected_ids", action="store_true")
    p.add_argument("--ignore-missing-expected", dest="strict_expected_ids", action="store_false")
    p.set_defaults(strict_expected_ids=False)

    p.add_argument("--llm-judge", action="store_true", help="Enable LLM-judge tier (shared between both sides).")
    p.add_argument("--llm-judge-all", action="store_true")
    p.add_argument("--judge-timeout", type=int, default=60)
    p.add_argument("--judge-model", type=str, default="deepseek-chat")
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

    bare_path = Path(args.bare_results)
    expected_path = Path(args.expected)
    if not bare_path.exists():
        raise SystemExit(f"Bare results file not found: {bare_path}")
    if not expected_path.exists():
        raise SystemExit(f"Expected file not found: {expected_path}")

    if args.report:
        report_path = Path(args.report)
    else:
        report_path = bare_path.parent / "comparison_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    judge = build_llm_judge_config(args)
    log_dir_bare = Path(args.log_dir_bare) if args.log_dir_bare else None
    log_dir_agent = Path(args.log_dir_agent) if args.log_dir_agent else None

    bare_report = validate_results(
        result_path=str(bare_path),
        expected_path=str(expected_path),
        log_dir=str(log_dir_bare) if log_dir_bare else None,
        strict_expected_ids=args.strict_expected_ids,
        llm_judge=judge,
    )

    agent_report: Optional[ValidationReport] = None
    if args.agent_results:
        agent_path = Path(args.agent_results)
        if not agent_path.exists():
            raise SystemExit(f"Agent results file not found: {agent_path}")
        agent_report = validate_results(
            result_path=str(agent_path),
            expected_path=str(expected_path),
            log_dir=str(log_dir_agent) if log_dir_agent else None,
            strict_expected_ids=args.strict_expected_ids,
            llm_judge=judge,
        )

    bare_answers = _read_answers(bare_path)
    agent_answers = _read_answers(Path(args.agent_results)) if args.agent_results else {}

    bare_dict = _to_dict(bare_report)
    agent_dict = _to_dict(agent_report) if agent_report is not None else None

    accuracy_bare = bare_dict.get("answer_accuracy")
    accuracy_agent = agent_dict.get("answer_accuracy") if agent_dict else None
    accuracy_delta = (
        accuracy_agent - accuracy_bare
        if accuracy_agent is not None and accuracy_bare is not None
        else None
    )

    comparison = {
        "bare": {
            "results_path": str(bare_path),
            "summary": bare_dict,
        },
        "agent": (
            {
                "results_path": str(args.agent_results),
                "summary": agent_dict,
            }
            if agent_report is not None
            else None
        ),
        "labels": {
            "prompt": args.prompt,
            "model": args.model,
        },
        "deltas": {
            "accuracy": accuracy_delta,
            "fallback_rate_bare": (
                bare_dict.get("fallback_or_unpassed_count", 0) / bare_dict["total"]
                if bare_dict.get("total") else 0.0
            ),
            "fallback_rate_agent": (
                (agent_dict.get("fallback_or_unpassed_count", 0) / agent_dict["total"])
                if agent_dict and agent_dict.get("total") else None
            ),
        },
        "per_problem": _per_problem(bare_report, agent_report, bare_answers, agent_answers),
    }

    # Also write individual side reports for the convenience of callers.
    write_validation_report(bare_report, str(report_path.with_name("bare_validation_report.json")))
    if agent_report is not None:
        write_validation_report(agent_report, str(report_path.with_name("agent_validation_report.json")))

    report_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print_score_summary(bare_dict, prefix="[bare]  ")
    if agent_dict is not None:
        print_score_summary(agent_dict, prefix="[agent] ")
    if accuracy_delta is not None:
        sign = "+" if accuracy_delta >= 0 else ""
        print(f"[delta] accuracy: {sign}{accuracy_delta:.2%} (agent - bare)")
    print(f"Comparison report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
