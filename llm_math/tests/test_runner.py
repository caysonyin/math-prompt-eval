"""Tests for the parallel batch runner."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from llm_math.runner import run_parallel_batch
from llm_math.vendor import MathSolution


class FakeSolver:
    """Solver that returns a deterministic MathSolution; records call count."""

    call_count = 0
    lock = threading.Lock()

    def __init__(self):
        self.last_run_log: Dict[str, Any] = {}

    def solve(self, problem_text: str, problem_id: str, raw_metadata: Optional[Dict[str, Any]] = None):
        with FakeSolver.lock:
            FakeSolver.call_count += 1
        solution = MathSolution(problem_id=problem_id, answer="42", answer_type="numeric")
        self.last_run_log = {"problem_id": problem_id, "call_count": FakeSolver.call_count}
        return solution


@pytest.fixture(autouse=True)
def _reset_call_count():
    FakeSolver.call_count = 0
    yield
    FakeSolver.call_count = 0


def _problems(n: int):
    return [
        {"problem_id": f"p{i}", "problem_text": f"problem {i}", "raw_metadata": {}}
        for i in range(1, n + 1)
    ]


def test_basic_run(tmp_path: Path):
    out = tmp_path / "results.jsonl"
    logs = tmp_path / "logs"
    summary = run_parallel_batch(
        solver_factory=FakeSolver,
        problems=_problems(5),
        output_path=out,
        log_dir=logs,
        workers=2,
    )
    assert summary["processed_this_run"] == 5
    assert summary["results_in_jsonl"] == 5
    assert summary["fallback_count"] == 0
    assert len(list(logs.glob("*.json"))) == 5
    # Each line is a valid MathSolution JSON
    for line in out.read_text().strip().split("\n"):
        obj = json.loads(line)
        assert obj["answer"] == "42"


def test_resume_skips_already_present(tmp_path: Path):
    out = tmp_path / "results.jsonl"
    out.write_text(
        json.dumps({"problem_id": "p1", "answer": "99", "answer_type": "numeric"}) + "\n"
    )
    logs = tmp_path / "logs"
    summary = run_parallel_batch(
        solver_factory=FakeSolver,
        problems=_problems(3),
        output_path=out,
        log_dir=logs,
        workers=1,
        resume=True,
    )
    assert summary["skipped_by_resume"] == 1
    assert summary["processed_this_run"] == 2
    # p1 was skipped (not re-solved); p2 and p3 should be in the output
    lines = out.read_text().strip().split("\n")
    pids = [json.loads(line)["problem_id"] for line in lines]
    assert "p1" in pids  # preserved from pre-existing file
    assert "p2" in pids
    assert "p3" in pids


def test_solver_exception_does_not_kill_run(tmp_path: Path):
    """An exception in one solver call should be caught; other problems continue."""

    class SometimesBadSolver:
        def __init__(self):
            self.last_run_log = {}
            self.n = 0

        def solve(self, problem_text, problem_id, raw_metadata=None):
            self.n += 1
            if problem_id == "p2":
                raise RuntimeError("synthetic failure")
            from llm_math.vendor import MathSolution
            return MathSolution(problem_id=problem_id, answer="ok", answer_type="text")

    out = tmp_path / "results.jsonl"
    logs = tmp_path / "logs"
    summary = run_parallel_batch(
        solver_factory=SometimesBadSolver,
        problems=_problems(3),
        output_path=out,
        log_dir=logs,
        workers=1,
    )
    assert summary["processed_this_run"] == 3
    assert summary["fallback_count"] == 1
    pids = [json.loads(line)["problem_id"] for line in out.read_text().strip().split("\n")]
    assert "p1" in pids and "p3" in pids
    # p2 should still appear as a fallback (unable_to_determine)
    p2_line = next(line for line in out.read_text().strip().split("\n") if '"p2"' in line)
    p2_obj = json.loads(p2_line)
    assert p2_obj["answer"] == "unable_to_determine"
