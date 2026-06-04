"""Convert external benchmarks to the local JSONL format.

Currently supports one subcommand:

  - `interns1_504` — converts `interns1_math_18domains_504.json` (the
    504-problem / 18-domain intern-S1 evaluation set) into the local
    `{problem_id, problem_text, domain, answer_type, expected_answer,
    raw_metadata}` JSONL schema consumed by `llm_math.run` and
    `llm_math.evaluate`.

The format mirrors `math_prove.convert_benchmarks` so the workflow feels
identical when switching between the two repos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Mapping from the 504 dataset's `answer_form` values to the canonical
# MathSolution `answer_type` enum (see llm_math/vendor/parser.py for the
# allowed values). The 504 set uses coarser categories; the equivalence
# tier in normalizer.py will still try its three-tier match.
ANSWER_FORM_MAP: Dict[str, str] = {
    "integer": "numeric",
    "real_number": "numeric",
    "rational": "formula",
    "symbolic_expression": "formula",
    "function": "formula",
    "set_or_tuple": "set",
    "matrix_vector": "matrix",
    "proof_text": "proof",
    "model_solution": "text",
}

# Mapping from the 504 dataset's `math_domain` values to the canonical
# domain ids used in llm_math/vendor/parser.py:DOMAIN_VALUES. Unknown
# values pass through unchanged (will become "other" via the upstream
# alias normalizer in parser.py).
DOMAIN_MAP: Dict[str, str] = {
    "calculus_analysis": "calculus_real_analysis",
    "real_analysis_measure": "real_analysis_measure",
    "linear_algebra_matrix": "linear_algebra",
    "abstract_algebra": "algebra",
    "ode": "ode",
    "pde": "pde",
    "complex_analysis": "complex_analysis",
    "topology": "topology",
    "functional_analysis": "functional_analysis",
    "probability": "probability",
    "mathematical_statistics": "statistics",
    "combinatorics": "combinatorics",
    "graph_discrete_math": "graph_theory",
    "number_theory": "number_theory",
    "geometry_analytic_geometry": "geometry",
    "operations_research": "operations_research_optimization",
    "convex_nonlinear_optimization": "operations_research_optimization",
    "numerical_analysis": "numerical_analysis",
}


def convert_interns1_504(input_path: Path, output_path: Path) -> int:
    """Convert the 504 problem JSON to canonical JSONL. Returns the count written."""
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON array, got {type(raw).__name__}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in raw:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("id", "")).strip()
            text = str(row.get("problem", "")).strip()
            if not pid or not text:
                continue
            answer = row.get("answer")
            answer_str = "" if answer is None else str(answer)
            domain_raw = str(row.get("math_domain", "")).strip()
            domain = DOMAIN_MAP.get(domain_raw, domain_raw or "other")
            answer_form_raw = str(row.get("answer_form", "")).strip()
            answer_type = ANSWER_FORM_MAP.get(answer_form_raw, "other")

            metadata = {k: v for k, v in row.items() if k not in {"id", "problem"}}
            out.write(
                json.dumps(
                    {
                        "problem_id": pid,
                        "problem_text": text,
                        "domain": domain,
                        "answer_type": answer_type,
                        "expected_answer": answer_str,
                        "raw_metadata": metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert external benchmarks to the local JSONL format."
    )
    sub = p.add_subparsers(dest="command", required=True)

    p504 = sub.add_parser(
        "interns1_504",
        help="Convert the intern-S1 504-problem / 18-domain evaluation set.",
    )
    p504.add_argument(
        "--input", type=str, required=True,
        help="Path to interns1_math_18domains_504.json",
    )
    p504.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL path (one record per line).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "interns1_504":
        n = convert_interns1_504(Path(args.input), Path(args.output))
        print(f"Wrote {n} problems to {args.output}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
