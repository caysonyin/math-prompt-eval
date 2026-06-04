"""Tests for the run.py CLI entrypoint and nested output path resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_math import run as run_module
from llm_math.run import _default_nested_paths, _resolve_paths, main, build_arg_parser


# ---- _default_nested_paths ---------------------------------------------------

def _ns(**overrides):
    """Build a minimal argparse.Namespace with the fields the helpers read."""
    base = {
        "prompt": "naive_v1",
        "model": "intern-s1",
        "base_output_dir": "outputs",
        "run_ts": None,
        "output": None,
        "results_json": None,
        "log_dir": None,
        "summary": None,
    }
    base.update(overrides)
    import argparse
    return argparse.Namespace(**base)


def test_default_nested_paths_format():
    args = _ns()
    with patch("llm_math.run.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260604_120000"
        paths = _default_nested_paths(args)
    assert paths["output"] == Path("outputs/naive_v1/intern-s1/20260604_120000/results.jsonl")
    assert paths["results_json"] == Path("outputs/naive_v1/intern-s1/20260604_120000/results.json")
    assert paths["log_dir"] == Path("outputs/naive_v1/intern-s1/20260604_120000/logs")
    assert paths["summary"] == Path("outputs/naive_v1/intern-s1/20260604_120000/run_summary.json")


def test_default_nested_paths_uses_run_ts_override():
    args = _ns(run_ts="20260101_000000", model="gpt-4o-mini")
    paths = _default_nested_paths(args)
    assert paths["output"] == Path("outputs/naive_v1/gpt-4o-mini/20260101_000000/results.jsonl")


def test_default_nested_paths_sanitizes_unsafe_names():
    """Model / prompt names with slashes or spaces are turned into underscore-delimited segments."""
    args = _ns(prompt="weird/name", model="model with spaces")
    paths = _default_nested_paths(args)
    segments = paths["output"].parts
    assert "/" not in segments[1]
    assert "/" not in segments[2]
    assert " " not in paths["output"].as_posix()


# ---- _resolve_paths ----------------------------------------------------------

def test_resolve_paths_with_explicit_output_keeps_legacy_layout():
    args = _ns(output="custom_results.jsonl", results_json="custom_results.json",
               log_dir="custom_logs", summary="custom_summary.json")
    out, log_dir, results_json, summary = _resolve_paths(args)
    assert out == Path("custom_results.jsonl")
    # When all four are explicit, they're honored verbatim.
    assert log_dir == Path("custom_logs")
    assert results_json == Path("custom_results.json")
    assert summary == Path("custom_summary.json")


def test_resolve_paths_explicit_output_others_default_to_parent():
    """Legacy: explicit --output only; the other three default to <output.parent>/<sibling>."""
    args = _ns(output="runs/foo.jsonl")
    out, log_dir, results_json, summary = _resolve_paths(args)
    assert out == Path("runs/foo.jsonl")
    assert log_dir == Path("runs/logs")
    assert results_json == Path("runs/foo.json")
    assert summary == Path("runs/run_summary.json")


def test_resolve_paths_no_output_uses_nested():
    args = _ns()
    with patch("llm_math.run.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20260604_120000"
        out, log_dir, results_json, summary = _resolve_paths(args)
    assert out == Path("outputs/naive_v1/intern-s1/20260604_120000/results.jsonl")
    assert log_dir == Path("outputs/naive_v1/intern-s1/20260604_120000/logs")


# ---- main() end-to-end with --dry-run ----------------------------------------

def test_main_dry_run_creates_nested_paths(tmp_path: Path, monkeypatch, sample_jsonl: Path, capsys):
    """`--dry-run --limit 2` should write to outputs/{prompt}/{model}/{ts}/..."""
    monkeypatch.chdir(tmp_path)
    # Use a fixed run-ts so we can assert exact paths.
    rc = main([
        "--input", str(sample_jsonl),
        "--dry-run",
        "--limit", "2",
        "--model", "fake-model",
        "--prompt", "boxed_v1",
        "--run-ts", "20260604_120000",
        "--workers", "1",
    ])
    assert rc == 0
    base = tmp_path / "outputs" / "boxed_v1" / "fake-model" / "20260604_120000"
    assert (base / "results.jsonl").exists()
    assert (base / "results.json").exists()
    assert (base / "run_summary.json").exists()
    assert (base / "logs").is_dir()

    # The JSONL should have 2 lines (--limit 2)
    lines = (base / "results.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2

    # Summary file should mention the prompt and model
    summary = json.loads((base / "run_summary.json").read_text())
    assert summary["prompt"] == "boxed_v1"
    assert summary["model"] == "fake-model"
    assert summary["processed_this_run"] == 2


def test_main_dry_run_resume_skips_existing(tmp_path: Path, monkeypatch, sample_jsonl: Path):
    """--resume with an existing results.jsonl should skip IDs already present."""
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "outputs" / "naive_v1" / "m" / "20260604_120000"
    base.mkdir(parents=True)
    (base / "results.jsonl").write_text(
        json.dumps({"problem_id": "s1", "answer": "42", "answer_type": "numeric"}) + "\n"
    )
    (base / "logs").mkdir()
    rc = main([
        "--input", str(sample_jsonl),
        "--dry-run",
        "--limit", "2",
        "--model", "m",
        "--prompt", "naive_v1",
        "--run-ts", "20260604_120000",
        "--workers", "1",
        "--resume",
    ])
    assert rc == 0
    lines = (base / "results.jsonl").read_text().strip().split("\n")
    # s1 was preserved, s2 was newly added
    pids = [json.loads(line)["problem_id"] for line in lines]
    assert pids == ["s1", "s2"]
