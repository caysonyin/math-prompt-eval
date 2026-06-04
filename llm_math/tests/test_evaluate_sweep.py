"""Tests for evaluate_sweep.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from llm_math import evaluate_sweep as es


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _make_sweep_tree(root: Path) -> Path:
    """Create two per-prompt result files under root/outputs/{prompt}/m/ts/."""
    expected = root / "expected.jsonl"
    _write_jsonl(expected, [
        {"problem_id": "p1", "expected_answer": "42"},
        {"problem_id": "p2", "expected_answer": "20"},
    ])
    for prompt in ("naive_v1", "boxed_v1"):
        d = root / "outputs" / prompt / "m" / "20260604_120000"
        _write_jsonl(d / "results.jsonl", [
            {"problem_id": "p1", "answer": "42", "answer_type": "numeric"},
            {"problem_id": "p2", "answer": "20", "answer_type": "numeric"},
        ])
        (d / "logs").mkdir()
    return expected


def test_iter_result_files_finds_prompt_model_ts_layout(tmp_path: Path):
    _make_sweep_tree(tmp_path)
    found = es._iter_result_files(tmp_path / "outputs")
    assert len(found) == 2
    pids = sorted(p.parent.parent.parent.name for p in found)
    assert pids == ["boxed_v1", "naive_v1"]


def test_iter_result_files_empty_when_no_tree(tmp_path: Path):
    assert es._iter_result_files(tmp_path / "outputs") == []


def test_main_validates_all_dirs(tmp_path: Path):
    expected = _make_sweep_tree(tmp_path)
    rc = es.main([
        "--sweep-dir", str(tmp_path / "outputs"),
        "--expected", str(expected),
        "--summary", str(tmp_path / "sweep_eval_summary.json"),
    ])
    assert rc == 0
    # Each prompt dir got a validation_report.json
    for prompt in ("naive_v1", "boxed_v1"):
        report = tmp_path / "outputs" / prompt / "m" / "20260604_120000" / "validation_report.json"
        assert report.exists()
    sweep = json.loads((tmp_path / "sweep_eval_summary.json").read_text())
    assert len(sweep["validated"]) == 2
    assert sweep["skipped"] == []
    assert sweep["failed"] == []


def test_main_skips_already_validated(tmp_path: Path):
    expected = _make_sweep_tree(tmp_path)
    target = tmp_path / "outputs" / "naive_v1" / "m" / "20260604_120000"
    (target / "validation_report.json").write_text("{}", encoding="utf-8")
    rc = es.main([
        "--sweep-dir", str(tmp_path / "outputs"),
        "--expected", str(expected),
        "--summary", str(tmp_path / "sweep_eval_summary.json"),
    ])
    assert rc == 0
    sweep = json.loads((tmp_path / "sweep_eval_summary.json").read_text())
    assert len(sweep["validated"]) == 1
    assert len(sweep["skipped"]) == 1
    assert sweep["skipped"][0]["results_jsonl"].endswith("naive_v1/m/20260604_120000/results.jsonl")


def test_main_force_overrides_skip(tmp_path: Path):
    expected = _make_sweep_tree(tmp_path)
    target = tmp_path / "outputs" / "naive_v1" / "m" / "20260604_120000"
    (target / "validation_report.json").write_text("{}", encoding="utf-8")
    rc = es.main([
        "--sweep-dir", str(tmp_path / "outputs"),
        "--expected", str(expected),
        "--force",
        "--summary", str(tmp_path / "sweep_eval_summary.json"),
    ])
    assert rc == 0
    sweep = json.loads((tmp_path / "sweep_eval_summary.json").read_text())
    assert len(sweep["validated"]) == 2
    assert sweep["skipped"] == []


def test_main_no_results_returns_1(tmp_path: Path):
    expected = tmp_path / "expected.jsonl"
    expected.write_text("{}\n", encoding="utf-8")
    rc = es.main([
        "--sweep-dir", str(tmp_path / "empty"),
        "--expected", str(expected),
    ])
    assert rc == 1
