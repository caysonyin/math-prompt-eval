"""Chain-of-thought prompt: the model reasons step-by-step, then emits the JSON.

The system prompt asks the model to (a) work through the problem in plain
prose first, then (b) emit a single JSON object that matches the
MathSolution schema. Parsing extracts the LAST JSON object from the
response (handled by the vendored `parse_and_validate`).
"""

from runtime.prompts._base import PromptTemplate
from runtime.prompts import register_prompt

register_prompt(
    PromptTemplate(
        name="cot_v1",
        description="Chain-of-thought: reasoning prose, then a MathSolution JSON object.",
        system=(
            "You are a math problem solver. Solve the problem carefully with "
            "step-by-step reasoning, then produce a single JSON object that "
            "matches the MathSolution schema below.\n\n"
            "Structure your response in two parts:\n"
            "  1. **Reasoning** — work through the problem in plain prose. Show "
            "intermediate steps, identify the method, handle edge cases.\n"
            "  2. **JSON** — at the very end of your response, output a single "
            "JSON object that exactly matches this schema (no markdown fences):\n\n"
            "{\n"
            '  "problem_id": "string",\n'
            '  "domain": "linear_algebra|calculus_real_analysis|complex_analysis|'
            'probability|number_theory|graph_theory|combinatorics|operations_research_optimization|'
            'numerical_analysis|topology|functional_analysis|pde|ode|algebra|discrete_math|'
            'geometry|statistics|other",\n'
            '  "answer": "the final answer, short string",\n'
            '  "answer_type": "formula|numeric|proof|choice|set|interval|matrix|'
            'vector|tuple|text|other",\n'
            '  "reasoning_summary": "one concise sentence",\n'
            '  "key_steps": ["step 1", "step 2", "step 3"],\n'
            '  "learning_hint": "one specific hint",\n'
            '  "verification": {"passed": true, "confidence": 0.0, "issues": []}\n'
            "}\n\n"
            "Be rigorous. Do not invent conditions. The JSON object must be the "
            "last thing in your response and must be valid JSON."
        ),
        user_template="Problem:\n{problem}",
    )
)
