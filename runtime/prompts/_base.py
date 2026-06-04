"""Prompt template dataclass for the bare-LLM framework.

A `PromptTemplate` bundles everything the solver needs to render a single
chat-completion call:

  - `system`            static system prompt
  - `user_template`     user-side prompt; must contain a `{problem}` placeholder
  - `response_format`   "json_object" (parse the response as JSON via
                        `parse_and_validate`) or "text" (run `extract_boxed`
                        first, then parse the extracted expression)
  - `description`       one-line summary; surfaced by `runtime.prompts.list_prompts()`

`register_prompt()` (in `runtime.prompts`) calls `_validate()` so the registry
rejects malformed templates at registration time — the hard errors are
`response_format` outside the allowed set and names that would break the
nested output directory layout; the soft warnings (`{problem}` missing,
oversized `system`, unescaped braces) keep third-party templates from
breaking while still nudging authors toward the canonical shape.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass


_VALID_RESPONSE_FORMATS = frozenset({"json_object", "text"})
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_PROBLEM_PLACEHOLDER = "{problem}"
# Detect named placeholders like `{problem_id}` that the solver does not know
# how to fill in (only `{problem}` is passed to str.format). A `{` immediately
# preceded by another `{` is the str.format literal-escape, so we exclude
# those from the match.
_NAMED_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _validate(template: "PromptTemplate") -> None:
    if not isinstance(template, PromptTemplate):
        raise TypeError(f"expected PromptTemplate, got {type(template).__name__}")

    if template.response_format not in _VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"prompt {template.name!r}: response_format must be one of "
            f"{sorted(_VALID_RESPONSE_FORMATS)}, got {template.response_format!r}"
        )

    if not _NAME_PATTERN.match(template.name):
        raise ValueError(
            f"prompt name {template.name!r} must match {_NAME_PATTERN.pattern} "
            "(letters, digits, underscore, dash only — required to be safe in paths)"
        )

    if _PROBLEM_PLACEHOLDER not in template.user_template:
        warnings.warn(
            f"prompt {template.name!r}: user_template is missing the "
            f"'{_PROBLEM_PLACEHOLDER}' placeholder; the solver will substitute "
            "an empty string and the LLM will not see the actual problem.",
            stacklevel=3,
        )

    if len(template.system) > 4000:
        warnings.warn(
            f"prompt {template.name!r}: system prompt is {len(template.system)} "
            "chars; most chat APIs cap at 4096 and a near-limit system will "
            "leave little room for the model to think.",
            stacklevel=3,
        )

    # Only the user_template is run through str.format() (the system prompt is
    # sent verbatim), so unknown-placeholder warnings only apply there.
    for match in _NAMED_PLACEHOLDER.finditer(template.user_template):
        name = match.group(1)
        if name == "problem":
            continue
        warnings.warn(
            f"prompt {template.name!r}: user_template contains an unknown "
            f"placeholder {{{name}}}; str.format() will raise because the "
            f"solver only fills in the 'problem' keyword.",
            stacklevel=3,
        )


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user_template: str
    response_format: str = "json_object"  # or "text"
    description: str = ""


__all__ = ["PromptTemplate", "_validate"]
