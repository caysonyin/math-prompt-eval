"""Tests for llm_math.io."""

from __future__ import annotations

import json
from pathlib import Path

from llm_math.io import (
    ID_FIELDS,
    RpmLimiter,
    load_problems,
    read_existing_results,
    load_and_validate_results,
    write_problem_log,
    resolve_api_config,
)
from llm_math.vendor import MathSolution, solution_to_json


def test_load_problems_jsonl(sample_jsonl: Path):
    problems = load_problems(str(sample_jsonl))
    assert len(problems) == 3
    assert problems[0]["problem_id"] == "s1"
    assert problems[0]["problem_text"] == "What is 6*7?"
    assert "raw_metadata" in problems[0]


def test_load_problems_supports_aliases(tmp_path: Path):
    p = tmp_path / "alt.jsonl"
    p.write_text(
        json.dumps({"qid": "x1", "question": "What is 2+2?", "domain": "other"})
        + "\n"
    )
    rows = load_problems(str(p))
    assert rows[0]["problem_id"] == "x1"
    assert rows[0]["problem_text"] == "What is 2+2?"


def test_load_problems_unsupported_format(tmp_path: Path):
    p = tmp_path / "foo.txt"
    p.write_text("hello")
    with __import__("pytest").raises(ValueError):
        load_problems(str(p))


def test_id_fields_covers_common_aliases():
    assert "problem_id" in ID_FIELDS
    assert "id" in ID_FIELDS
    assert "question_id" in ID_FIELDS
    assert "qid" in ID_FIELDS


def test_read_existing_results_round_trip(tmp_path: Path):
    p = tmp_path / "results.jsonl"
    p.write_text(
        json.dumps({"problem_id": "a", "answer": "1"}) + "\n"
        + json.dumps({"problem_id": "b", "answer": "2"}) + "\n"
    )
    rows = read_existing_results(p)
    assert rows == {"a": {"problem_id": "a", "answer": "1"}, "b": {"problem_id": "b", "answer": "2"}}


def test_read_existing_results_missing_file(tmp_path: Path):
    p = tmp_path / "nope.jsonl"
    assert read_existing_results(p) == {}


def test_load_and_validate_results(tmp_path: Path):
    p = tmp_path / "results.jsonl"
    sol1 = MathSolution(problem_id="a", answer="1", answer_type="numeric")
    sol2 = MathSolution(problem_id="b", answer="x", answer_type="formula")
    p.write_text(solution_to_json(sol1, indent=None) + "\n" + solution_to_json(sol2, indent=None) + "\n")

    good, errors = load_and_validate_results(p)
    assert errors == []
    assert [r["problem_id"] for r in good] == ["a", "b"]


def test_load_and_validate_results_records_schema_errors(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"problem_id": "a", "answer": 123}\n')  # answer should be str
    good, errors = load_and_validate_results(p)
    assert good == []
    assert len(errors) == 1
    assert errors[0]["line"] == 1


def test_write_problem_log_sanitizes_id(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_problem_log(logs, "a/b c", {"x": 1})
    files = list(logs.glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "a_b_c.json"
    assert json.loads(files[0].read_text()) == {"x": 1}


def test_rpm_limiter_does_not_block_under_limit():
    lim = RpmLimiter(1000)
    for _ in range(5):
        lim.acquire()
    # all 5 acquires should return immediately


def test_rpm_limiter_blocks_when_full(monkeypatch):
    """Verify the limiter actually waits when over the RPM cap."""
    import time
    import llm_math.io as io_mod

    sleeps = []
    real_sleep = time.sleep

    def fake_sleep(s):
        sleeps.append(s)
        # do not actually sleep to keep tests fast; the next acquire() loop
        # will pop the oldest event so the test still terminates.
        # We need the events to age out — replace acquire() temporarily.
        return None

    monkeypatch.setattr(io_mod.time, "sleep", fake_sleep)

    lim = io_mod.RpmLimiter(2)
    lim.acquire()
    lim.acquire()
    t0 = time.monotonic()
    lim.acquire()  # this should trigger a sleep
    elapsed = time.monotonic() - t0
    assert len(sleeps) >= 1
    # We never actually waited, so elapsed should be tiny.


def test_resolve_api_config_prefers_args(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("LLM_API_BASE", "https://env.example/v1")

    import argparse
    ns = argparse.Namespace(api_key="arg-key", api_base="https://arg.example/v1")
    key, base = resolve_api_config(ns)
    assert key == "arg-key"
    assert base == "https://arg.example/v1"


def test_resolve_api_config_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    import argparse
    ns = argparse.Namespace(api_key=None, api_base=None)
    key, base = resolve_api_config(ns)
    assert key == "env-key"
    # default OpenAI endpoint
    assert "openai.com" in base
