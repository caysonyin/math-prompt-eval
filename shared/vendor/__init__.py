"""Vendored modules from the upstream `mathsolve-agent/math_prove/` package.

Source repository: https://github.com/caysonyin/mathsolve-agent
Original license: Apache-2.0
Vendored on: 2026-06-01

Each file under this package starts with an attribution header. The vendored
content is byte-identical to the upstream at vendoring time (no modifications);
relative imports (`from .normalizer import …`, `from .parser import …`) work
unchanged because the three files are siblings inside the `shared.vendor`
package.

Re-exports the public surface used elsewhere in this project. Pydantic models and
internal helpers in the vendored modules are intentionally not re-exported; if
you need one, import it from the specific submodule (e.g.
`shared.vendor.parser.MathSolution`).
"""

from shared.vendor.parser import (
    MathSolution,
    CandidateSolution,
    VerificationResult,
    ClassificationResult,
    SelectionResult,
    LayerCheck,
    ClaimCheck,
    fallback_solution,
    solution_to_json,
    parse_and_validate,
    parse_json_object,
)
from shared.vendor.normalizer import (
    AnswerForms,
    EquivalenceResult,
    equivalent_answers,
    normalize_answer,
    extract_boxed,
)
from shared.vendor.validator import (
    LLMJudgeConfig,
    ValidationReport,
    ValidationItem,
    _GenericLLM,
    load_result_file,
    load_expected_file,
    validate_results,
    write_validation_report,
)

__all__ = [
    "MathSolution",
    "CandidateSolution",
    "VerificationResult",
    "ClassificationResult",
    "SelectionResult",
    "LayerCheck",
    "ClaimCheck",
    "fallback_solution",
    "solution_to_json",
    "parse_and_validate",
    "parse_json_object",
    "AnswerForms",
    "EquivalenceResult",
    "equivalent_answers",
    "normalize_answer",
    "extract_boxed",
    "LLMJudgeConfig",
    "ValidationReport",
    "ValidationItem",
    "_GenericLLM",
    "load_result_file",
    "load_expected_file",
    "validate_results",
    "write_validation_report",
]
