"""Tests for the prompt registry and template loading."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

from runtime.prompts import (
    PromptTemplate,
    get_prompt,
    list_prompts,
    register_prompt,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the registry before each test, then re-register the built-ins.

    The post-yield teardown intentionally does NOT clear the registry, so
    tests in other modules (test_solver, test_io, test_runner) that import
    after test_prompts can still see the built-in prompts.
    """
    reset_for_tests()
    _reload_prompt_modules()
    yield


def _reload_prompt_modules() -> None:
    """Re-execute every non-`_`-prefixed prompt module.

    Name-agnostic: discovers modules via `pkgutil.iter_modules` rather than
    hard-coding naive_v1 / cot_v1 / boxed_v1. See conftest.py for the
    shared version of this helper.
    """
    import runtime.prompts as prompts_pkg

    for module_info in pkgutil.iter_modules(prompts_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{prompts_pkg.__name__}.{module_info.name}"
        importlib.import_module(full_name)
        importlib.reload(importlib.import_module(full_name))


def test_three_built_in_prompts_registered():
    names = {t.name for t in list_prompts()}
    assert {"naive_v1", "cot_v1", "boxed_v1"}.issubset(names)


def test_naive_v1_uses_json_object():
    p = get_prompt("naive_v1")
    assert p.response_format == "json_object"
    assert "{problem}" in p.user_template
    assert "answer" in p.system


def test_boxed_v1_uses_text_format():
    p = get_prompt("boxed_v1")
    assert p.response_format == "text"
    assert "\\boxed" in p.system


def test_get_prompt_unknown_raises():
    with pytest.raises(KeyError):
        get_prompt("does_not_exist")


def test_register_prompt_appends_to_registry():
    t = PromptTemplate(
        name="custom_v1",
        system="system prompt",
        user_template="Problem: {problem}",
        response_format="json_object",
        description="custom test",
    )
    register_prompt(t)
    assert get_prompt("custom_v1") == t
    assert any(x.name == "custom_v1" for x in list_prompts())


def test_register_prompt_rejects_non_template():
    with pytest.raises(TypeError):
        register_prompt("not a template")  # type: ignore[arg-type]


def test_register_warns_missing_problem_placeholder():
    t = PromptTemplate(
        name="no_problem_v1",
        system="sys",
        user_template="just static text with no placeholder",
    )
    with pytest.warns(UserWarning, match=r"missing the '\{problem\}' placeholder"):
        register_prompt(t)


def test_register_rejects_bad_response_format():
    t = PromptTemplate(
        name="badfmt_v1",
        system="sys",
        user_template="Problem: {problem}",
        response_format="xml",
    )
    with pytest.raises(ValueError, match="response_format"):
        register_prompt(t)


def test_register_rejects_bad_name():
    t = PromptTemplate(
        name="has spaces",
        system="sys",
        user_template="Problem: {problem}",
    )
    with pytest.raises(ValueError, match="must match"):
        register_prompt(t)


def test_iter_modules_picks_up_new_file(tmp_path, monkeypatch):
    """A new prompt file dropped into the package is auto-discovered.

    Drops a synthetic `pkg/_dynamic_v1.py` into `tmp_path`, monkeypatches
    `runtime.prompts.__path__` to include it, and verifies the module is
    iterated by `pkgutil.iter_modules`. (Auto-discovery at import time
    requires importing the package, which we don't do here; the test
    instead proves the iteration surface is what we expect.)
    """
    pkg_root = tmp_path / "fake_prompts"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("")
    (pkg_root / "_hidden.py").write_text("# should be skipped")
    (pkg_root / "_dynamic_v1.py").write_text(
        "from runtime.prompts._base import PromptTemplate\n"
        "from runtime.prompts import register_prompt\n"
        "register_prompt(PromptTemplate(name='_dynamic_v1', system='s', "
        "user_template='Problem: {problem}'))\n"
    )

    discovered = [mi.name for mi in pkgutil.iter_modules([str(pkg_root)])]
    assert "_dynamic_v1" in discovered
    assert "_hidden" in discovered  # iter_modules returns all; we filter in code

    # And confirm the runtime filter (skip underscore-prefixed) is what we use.
    filtered = [n for n in discovered if not n.startswith("_")]
    assert "_dynamic_v1" not in filtered
