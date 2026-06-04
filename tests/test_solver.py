"""Tests for BareLLMSolver — uses a fake LLM client."""

from __future__ import annotations

import pytest

from runtime.solver import BareLLMSolver
from shared.vendor import MathSolution


class FakeClient:
    model = "fake-model"
    temperature = 0.0
    max_tokens = 4096

    def __init__(self, response: str, raise_exc: bool = False):
        self.response = response
        self.raise_exc = raise_exc

    def chat(self, messages, response_format_json=True):
        if self.raise_exc:
            raise RuntimeError("boom")
        return self.response


def _build_solver(response: str, raise_exc: bool = False, prompt: str = "naive_v1") -> BareLLMSolver:
    s = BareLLMSolver(
        model="fake",
        api_key="k",
        api_base="https://api.example/v1/chat/completions",
        prompt_name=prompt,
        rpm_limit=10000,
    )
    s._client = FakeClient(response, raise_exc=raise_exc)
    return s


def test_happy_path_returns_parsed_solution():
    s = _build_solver(
        '{"problem_id": "t1", "answer": "42", "answer_type": "numeric", "reasoning_summary": "6*7"}'
    )
    result = s.solve("What is 6*7?", "t1")
    assert isinstance(result, MathSolution)
    assert result.answer == "42"
    assert result.answer_type == "numeric"
    assert s.last_run_log["parse_ok"] is True


def test_bad_json_returns_fallback():
    s = _build_solver("not valid json {")
    result = s.solve("foo", "t2")
    assert result.answer == "unable_to_determine"
    assert s.last_run_log["parse_ok"] is False
    assert "error" in s.last_run_log


def test_client_exception_returns_fallback():
    s = _build_solver("", raise_exc=True)
    result = s.solve("foo", "t3")
    assert result.answer == "unable_to_determine"
    assert s.last_run_log["parse_ok"] is False
    assert "RuntimeError" in s.last_run_log["error"]


def test_boxed_prompt_runs_extract_boxed():
    s = _build_solver("The answer is $\\boxed{42}$.", prompt="boxed_v1")
    result = s.solve("What is 6*7?", "t4")
    # boxed_v1 wraps the boxed answer in a minimal JSON; answer should be 42
    assert result.answer == "42"
    assert s.last_run_log["parse_ok"] is True
    # Post-processing should be recorded
    assert "post_processed" in s.last_run_log


def test_solver_records_messages_in_log():
    s = _build_solver('{"problem_id": "t1", "answer": "1"}')
    s.solve("What is 1?", "t1")
    log = s.last_run_log
    assert log["prompt_name"] == "naive_v1"
    assert "messages" in log
    assert log["messages"][0]["role"] == "system"
    assert log["messages"][1]["role"] == "user"
    assert "What is 1?" in log["messages"][1]["content"]


def test_solver_uses_selected_prompt_template():
    s = _build_solver('{"answer": "1"}', prompt="cot_v1")
    s.solve("foo", "t1")
    # cot_v1 system should mention "step-by-step" or similar
    assert "step" in s.last_run_log["messages"][0]["content"].lower() or \
           "reasoning" in s.last_run_log["messages"][0]["content"].lower()


def test_solver_unknown_prompt_name_raises():
    with pytest.raises(KeyError):
        BareLLMSolver(
            model="fake",
            api_key="k",
            api_base="https://api.example/v1/chat/completions",
            prompt_name="does_not_exist",
        )
