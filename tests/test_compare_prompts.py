"""Tests for compare_prompts.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from evaluation import compare_prompts as cp


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def expected_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "expected.jsonl"
    _write_jsonl(path, [
        {"problem_id": "p1", "expected_answer": "42", "domain": "other"},
        {"problem_id": "p2", "expected_answer": "20", "domain": "operations_research_optimization"},
        {"problem_id": "p3", "expected_answer": "{-2,-1,1,2}", "domain": "calculus_real_analysis"},
    ])
    return path


@pytest.fixture
def results_a(tmp_path: Path) -> Path:
    path = tmp_path / "a.jsonl"
    _write_jsonl(path, [
        {"problem_id": "p1", "answer": "42", "answer_type": "numeric"},
        {"problem_id": "p2", "answer": "20", "answer_type": "numeric"},
        {"problem_id": "p3", "answer": "unable_to_determine", "answer_type": "text"},
    ])
    return path


@pytest.fixture
def results_b(tmp_path: Path) -> Path:
    path = tmp_path / "b.jsonl"
    _write_jsonl(path, [
        {"problem_id": "p1", "answer": "42", "answer_type": "numeric"},
        {"problem_id": "p2", "answer": "unable_to_determine", "answer_type": "text"},
        {"problem_id": "p3", "answer": "{-2,-1,1,2}", "answer_type": "set"},
    ])
    return path


def test_resolve_labels_uses_file_stem_when_no_labels():
    labels = cp._resolve_labels(["a.jsonl", "b.jsonl"], [])
    assert labels == ["a", "b"]


def test_resolve_labels_honours_explicit_labels():
    labels = cp._resolve_labels(["a.jsonl", "b.jsonl"], ["A", "B"])
    assert labels == ["A", "B"]


def test_resolve_labels_rejects_mismatched_counts():
    with pytest.raises(SystemExit):
        cp._resolve_labels(["a.jsonl", "b.jsonl"], ["A"])


def test_build_report_two_prompts(results_a, results_b, expected_jsonl, tmp_path):
    report = cp.build_report(
        results=[str(results_a), str(results_b)],
        labels=["A", "B"],
        expected_path=str(expected_jsonl),
        log_dirs=[None, None],
        judge=None,
    )
    assert report["labels"] == ["A", "B"]
    assert len(report["per_prompt"]) == 2
    # Prompt A: p1, p2 right; p3 wrong → 2/3 = 0.6667
    assert report["per_prompt"][0]["local_acc"] == pytest.approx(2 / 3)
    # Prompt B: p1, p3 right; p2 wrong → 2/3 = 0.6667
    assert report["per_prompt"][1]["local_acc"] == pytest.approx(2 / 3)
    # Fallback counts come from the `unable_to_determine` answer string.
    assert report["per_prompt"][0]["fallback"] == 1
    assert report["per_prompt"][1]["fallback"] == 1

    # per_problem rows: p1 won by both; p2 won only by A; p3 won only by B.
    pp = {row["problem_id"]: row for row in report["per_problem"]}
    assert pp["p1"]["winner"] == ["A", "B"]
    assert pp["p2"]["winner"] == ["A"]
    assert pp["p3"]["winner"] == ["B"]


def test_build_report_accuracy_matrix(results_a, results_b, expected_jsonl):
    report = cp.build_report(
        results=[str(results_a), str(results_b)],
        labels=["A", "B"],
        expected_path=str(expected_jsonl),
        log_dirs=[None, None],
        judge=None,
    )
    matrix = report["deltas"]["accuracy_matrix"]
    assert len(matrix) == 2
    # Equal accuracy → 0pp delta both ways.
    assert matrix[0][1] == pytest.approx(0.0, abs=0.01)
    assert matrix[1][0] == pytest.approx(0.0, abs=0.01)
    # Diagonal is exactly 0.0.
    assert matrix[0][0] == 0.0
    assert matrix[1][1] == 0.0


def test_main_writes_json_and_prints_table(results_a, results_b, expected_jsonl, tmp_path, capsys):
    out = tmp_path / "report.json"
    rc = cp.main([
        "--results", str(results_a),
        "--label", "A",
        "--results", str(results_b),
        "--label", "B",
        "--expected", str(expected_jsonl),
        "--report", str(out),
    ])
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["labels"] == ["A", "B"]
    captured = capsys.readouterr()
    assert "Prompt comparison" in captured.out
    assert "| label |" in captured.out


def test_main_markdown_flag_writes_md(results_a, results_b, expected_jsonl, tmp_path):
    out = tmp_path / "report.json"
    rc = cp.main([
        "--results", str(results_a),
        "--label", "A",
        "--results", str(results_b),
        "--label", "B",
        "--expected", str(expected_jsonl),
        "--report", str(out),
        "--markdown",
    ])
    assert rc == 0
    md = out.with_suffix(".md")
    assert md.exists()
    text = md.read_text()
    assert "# Prompt comparison" in text
    assert "| A |" in text
    assert "| B |" in text


def test_build_report_rejects_duplicate_labels(results_a, results_b, expected_jsonl):
    with pytest.raises(SystemExit, match="must be unique"):
        cp.build_report(
            results=[str(results_a), str(results_b)],
            labels=["same", "same"],
            expected_path=str(expected_jsonl),
            log_dirs=[None, None],
            judge=None,
        )
