"""Shared pytest fixtures."""

from __future__ import annotations

import importlib
from pathlib import Path
import pkgutil
import textwrap

import pytest

# Ensure the prompt registry is populated regardless of which test module
# runs first. The solver imports `from runtime.prompts import get_prompt`,
# so simply importing the package here triggers the auto-discovery of
# every prompt module (naive_v1 / cot_v1 / boxed_v1 + any future ones).
import runtime  # noqa: F401


def _reload_prompt_modules() -> None:
    """Re-import every non-`_`-prefixed module in `runtime.prompts`.

    Mirrors the discovery logic in `runtime.prompts._discover_prompts()` so
    that `reset_for_tests()` actually clears the cache rather than relying
    on a hardcoded list of names.
    """
    import runtime.prompts as prompts_pkg

    for module_info in pkgutil.iter_modules(prompts_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{prompts_pkg.__name__}.{module_info.name}"
        importlib.import_module(full_name)
        importlib.reload(importlib.import_module(full_name))


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "sample.jsonl"
    p.write_text(
        textwrap.dedent(
            """\
            {"problem_id": "s1", "problem_text": "What is 6*7?", "domain": "other", "answer_type": "numeric", "expected_answer": "42"}
            {"problem_id": "s2", "problem_text": "Max of 3x+4y.", "domain": "operations_research_optimization", "answer_type": "numeric", "expected_answer": "20"}
            {"problem_id": "s3", "problem_text": "Roots of x^4-5x^2+4=0.", "domain": "calculus_real_analysis", "answer_type": "set", "expected_answer": "{-2,-1,1,2}"}
            """
        ),
        encoding="utf-8",
    )
    return p
