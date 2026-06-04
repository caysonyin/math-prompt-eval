"""Prompt registry for the bare-LLM framework.

Add a new prompt template by dropping a sibling file in this directory —
no imports, no edits to this file. Each module calls `register_prompt(...)`
exactly once at import time:

    # runtime/prompts/my_prompt_v1.py
    from runtime.prompts._base import PromptTemplate
    from runtime.prompts import register_prompt

    register_prompt(PromptTemplate(
        name="my_prompt_v1",
        system="...",
        user_template="Problem: {problem}",
    ))

Discovery happens at package import time via `pkgutil.iter_modules(__path__)`,
so anything placed here is automatically wired into the `--prompt` choices of
`runtime.run` and the registry surfaced by `runtime.prompts.list_prompts()`.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List

from runtime.prompts._base import PromptTemplate, _validate


_REGISTRY: Dict[str, PromptTemplate] = {}


def register_prompt(template: PromptTemplate) -> PromptTemplate:
    """Register a PromptTemplate. Returns the template for decorator use.

    Validates `template` (response_format, name pattern, {problem} placeholder,
    system length) before insertion. Validation problems either raise
    `ValueError` (hard error: invalid response_format / name) or `warnings.warn`
    (soft: missing placeholder / oversized system / unescaped braces), so a
    new template can't silently break the output path layout, but third-party
    templates don't blow up over a missing placeholder.
    """
    if not isinstance(template, PromptTemplate):
        raise TypeError(f"register_prompt expected PromptTemplate, got {type(template).__name__}")
    _validate(template)
    if template.name in _REGISTRY:
        # Allow re-registration to support test fixtures that reset state, but
        # warn if the new template differs from the existing one.
        if _REGISTRY[template.name] != template:
            import warnings
            warnings.warn(f"prompt {template.name!r} re-registered with different content")
    _REGISTRY[template.name] = template
    return template


def get_prompt(name: str) -> PromptTemplate:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none registered>"
        raise KeyError(f"unknown prompt {name!r}; available: {available}")
    return _REGISTRY[name]


def list_prompts() -> List[PromptTemplate]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY.keys())]


def reset_for_tests() -> None:
    """Clear the registry. Used by tests that exercise the registration API."""
    _REGISTRY.clear()


def _discover_prompts() -> None:
    """Auto-import every non-`_`-prefixed module in this package.

    Importing each module triggers its module-level `register_prompt(...)` call.
    This is intentionally a side effect on package import — see the docstring
    at the top of this file for the rationale.
    """
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module_info.name}")


_discover_prompts()


__all__ = [
    "PromptTemplate",
    "register_prompt",
    "get_prompt",
    "list_prompts",
    "reset_for_tests",
]
