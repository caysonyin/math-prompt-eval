"""Bare-LLM math solver.

One chat completion per problem. No tools, no retries on the LLM side, no
verification, no candidate selection. Parse failures are caught and
surfaced as a `fallback_solution` with the error message attached.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

from shared.io import RpmLimiter
from runtime.llm_client import BareLLMClient
from runtime.prompts import get_prompt
from shared.vendor import (
    MathSolution,
    extract_boxed,
    fallback_solution,
    parse_and_validate,
)


class BareLLMSolver:
    """A single-shot, no-scaffolding LLM math solver.

    `solve()` does:
        1. acquire an RPM token (rate limiting)
        2. render the selected prompt template
        3. one chat-completion call
        4. parse the response (with `extract_boxed` post-processing for text
           prompts); on any failure return a `fallback_solution`
        5. stash a structured `last_run_log` for the per-problem log writer
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        api_base: str,
        prompt_name: str = "naive_v1",
        rpm_limit: int = 80,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> None:
        self.prompt_name = prompt_name
        self._prompt = get_prompt(prompt_name)
        self._client = BareLLMClient(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._limiter = RpmLimiter(rpm_limit)
        self._last_run_log: Dict[str, Any] = {}

    def solve(
        self,
        problem_text: str,
        problem_id: str,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> MathSolution:
        started_at = time.time()
        log: Dict[str, Any] = {
            "problem_id": problem_id,
            "prompt_name": self.prompt_name,
            "raw_problem": problem_text,
            "raw_metadata": raw_metadata or {},
            "started_at": started_at,
        }

        try:
            self._limiter.acquire()
            user_prompt = self._prompt.user_template.format(problem=problem_text)
            messages = [
                {"role": "system", "content": self._prompt.system},
                {"role": "user", "content": user_prompt},
            ]
            log["messages"] = messages
            log["request"] = {
                "model": self._client.model,
                "temperature": self._client.temperature,
                "max_tokens": self._client.max_tokens,
                "response_format_json": self._prompt.response_format == "json_object",
            }

            raw_response = self._client.chat(
                messages,
                response_format_json=self._prompt.response_format == "json_object",
            )
            log["raw_response"] = raw_response[:2000]
            log["raw_response_truncated"] = len(raw_response) > 2000
            log["raw_response_length"] = len(raw_response)

            parse_input = self._post_process(raw_response)
            if parse_input != raw_response:
                log["post_processed"] = parse_input

            try:
                solution = parse_and_validate(parse_input, problem_id)
            except Exception as parse_exc:
                if "Invalid \\escape" in str(parse_exc):
                    repaired = re.sub(r'\\(?=[^"\\/bfnrtu])', r'\\\\', parse_input)
                    log["json_repair_applied"] = "latex_backslash_escape"
                    solution = parse_and_validate(repaired, problem_id)
                else:
                    raise
            log["parse_ok"] = True
            log["final_json"] = solution.model_dump(mode="json")
        except Exception as exc:
            solution = fallback_solution(problem_id, f"{type(exc).__name__}: {exc}")
            log["parse_ok"] = False
            log["error"] = repr(exc)
            log["final_json"] = solution.model_dump(mode="json")

        log["latency_seconds"] = round(time.time() - started_at, 3)
        log["finished_at"] = time.time()
        self._last_run_log = log
        return solution

    def _post_process(self, raw: str) -> str:
        """For text-response prompts, run `extract_boxed` to pull out the answer
        and wrap it in a minimal MathSolution JSON before parsing. The resulting
        object will be enriched with reasoning/key_steps etc. via the parse
        path, or fall back to a fallback_solution if the expression is
        unparseable.
        """
        if self._prompt.response_format != "text":
            return raw
        extracted = extract_boxed(raw)
        if extracted is None:
            # No \\boxed{...} found; pass the raw text — parse_and_validate will
            # likely fail and the caller will get a fallback_solution.
            return raw
        # Build a minimal JSON payload. The full reasoning stays in the raw
        # response in the run log; the MathSolution.answer field gets the
        # boxed expression.
        import json
        return json.dumps(
            {
                "answer": extracted,
                "reasoning_summary": raw,
            }
        )

    @property
    def last_run_log(self) -> Dict[str, Any]:
        return self._last_run_log


__all__ = ["BareLLMSolver"]
