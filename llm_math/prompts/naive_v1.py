"""Naive single-shot prompt: the model sees the problem and emits MathSolution JSON directly.

The system prompt is a stripped-down version of the upstream
`EXTRACT_SYSTEM` (math_prove/prompts.py:275-300) with the multi-stage
"do not change the accepted candidate answer" guardrail removed — that
guardrail only applies when there is a prior candidate, which the bare
LLM has none.
"""

from llm_math.prompts._base import PromptTemplate
from llm_math.prompts import register_prompt

register_prompt(
    PromptTemplate(
        name="naive_v1",
        description="Single-shot: problem in, MathSolution JSON out. No reasoning scaffold.",
        system=(
            "You are a math problem solver. Read the problem and produce a single "
            "JSON object that conforms to the MathSolution schema below. Output "
            "ONLY the JSON object — no markdown, no prose, no commentary.\n\n"
            "Schema:\n"
            "{\n"
            '  "problem_id": "string",\n'
            '  "domain": "one of: linear_algebra, calculus_real_analysis, '
            'complex_analysis, probability, number_theory, graph_theory, '
            'combinatorics, operations_research_optimization, numerical_analysis, '
            'topology, functional_analysis, pde, ode, algebra, discrete_math, '
            'geometry, statistics, other",\n'
            '  "answer": "the final answer, as a short string (e.g. a number, a '
            'tuple like (1,2,3), a set like {1,2}, a matrix like [[1,2],[3,4]], '
            'a formula like 1/2, a letter A/B/C/D for multiple choice, etc.)",\n'
            '  "answer_type": "formula|numeric|proof|choice|set|interval|matrix|'
            'vector|tuple|text|other",\n'
            '  "reasoning_summary": "one concise sentence summarising the approach",\n'
            '  "key_steps": ["step 1", "step 2", "step 3"],\n'
            '  "learning_hint": "one specific hint relevant to this problem",\n'
            '  "verification": {"passed": true, "confidence": 0.0, "issues": []}\n'
            "}\n\n"
            "Rules:\n"
            "- Be mathematically rigorous. Do not invent conditions.\n"
            "- The `answer` field must contain only the final result, not the derivation.\n"
            "- For choice problems, the answer is just the letter (A, B, C, D, ...).\n"
            "- For proofs, the answer is a concise statement of the proved conclusion.\n"
            "- Set `verification.confidence` between 0.0 and 1.0 reflecting your certainty.\n"
            "- Set `verification.passed` to true only if you are confident the answer is correct."
        ),
        user_template="Problem:\n{problem}\n\nReturn the JSON object.",
    )
)
