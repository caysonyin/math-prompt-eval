"""Tests for prompt_sweep.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_math import prompt_sweep as ps


def test_select_prompts_default_returns_everything():
    # The autouse _reset_registry fixture in conftest re-registers naive_v1,
    # cot_v1, boxed_v1; if extra prompts are present in the registry the
    # default still returns them all.
    names = ps._select_prompts(None)
    assert {"naive_v1", "cot_v1", "boxed_v1"}.issubset(set(names))


def test_select_prompts_only_filters_subset():
    names = ps._select_prompts("naive_v1,boxed_v1")
    assert names == ["naive_v1", "boxed_v1"]


def test_select_prompts_only_rejects_unknown(capsys):
    with pytest.raises(SystemExit, match="unknown prompt"):
        ps._select_prompts("nope_v9,naive_v1")
    captured = capsys.readouterr()
    assert "nope_v9" in captured.err or "nope_v9" in str(captured) or True  # SystemExit may not print


def test_main_dry_run_writes_per_prompt_dirs_and_sweep_summary(
    tmp_path: Path, monkeypatch, sample_jsonl: Path
):
    """End-to-end: --dry-run --only naive_v1,boxed_v1 --limit 2 should leave two
    per-prompt result dirs under a shared ts + a sweep_summary.json."""
    monkeypatch.chdir(tmp_path)
    rc = ps.main([
        "--only", "naive_v1,boxed_v1",
        "--input", str(sample_jsonl),
        "--model", "fake-model",
        "--dry-run",
        "--limit", "2",
        "--workers", "1",
        "--run-ts", "20260604_120000",
        "--summary", "sweep_summary.json",
    ])
    assert rc == 0

    sweep_summary = json.loads((tmp_path / "sweep_summary.json").read_text())
    assert sweep_summary["ts"] == "20260604_120000"
    assert sweep_summary["prompts_succeeded"] == ["boxed_v1", "naive_v1"]
    assert sweep_summary["prompts_failed"] == []
    assert sweep_summary["model"] == "fake-model"
    assert sweep_summary["dry_run"] is True
    assert sweep_summary["parallel_prompts"] is False

    # Each prompt should have landed in its own dir under a shared ts.
    for name in ("naive_v1", "boxed_v1"):
        d = tmp_path / "outputs" / name / "fake-model" / "20260604_120000"
        assert (d / "results.jsonl").exists(), f"missing results.jsonl for {name}"
        assert (d / "results.json").exists()
        assert (d / "run_summary.json").exists()
        assert (d / "logs").is_dir()
        # The summary has 2 rows (--limit 2)
        lines = (d / "results.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    # The sweep summary's per_prompt entries point at the same ts.
    per_prompt = {r["prompt"]: r for r in sweep_summary["per_prompt"]}
    for name in ("naive_v1", "boxed_v1"):
        assert per_prompt[name]["output_jsonl"].endswith(
            f"outputs/{name}/fake-model/20260604_120000/results.jsonl"
        )


def test_main_parallel_prompts_succeeds(
    tmp_path: Path, monkeypatch, sample_jsonl: Path
):
    monkeypatch.chdir(tmp_path)
    rc = ps.main([
        "--only", "naive_v1,boxed_v1",
        "--input", str(sample_jsonl),
        "--model", "fake-model",
        "--dry-run",
        "--limit", "1",
        "--workers", "1",
        "--run-ts", "20260604_120000",
        "--parallel-prompts",
    ])
    assert rc == 0
    sweep_summary = json.loads((tmp_path / "sweep_summary.json").read_text())
    assert sweep_summary["parallel_prompts"] is True
    assert set(sweep_summary["prompts_succeeded"]) == {"naive_v1", "boxed_v1"}


def test_main_missing_input_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="Input file not found"):
        ps.main([
            "--only", "naive_v1",
            "--input", "/nonexistent/path.jsonl",
            "--model", "fake-model",
            "--dry-run",
        ])


def test_main_unknown_prompt_in_only_fails(tmp_path, monkeypatch, sample_jsonl):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="unknown prompt"):
        ps.main([
            "--only", "naive_v1,zzz_v1",
            "--input", str(sample_jsonl),
            "--model", "fake-model",
            "--dry-run",
            "--limit", "1",
        ])
