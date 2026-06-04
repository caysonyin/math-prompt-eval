"""\\boxed{}-style free-form prompt.

The system prompt asks the model to solve the problem in plain prose and
end with `\\boxed{<answer>}` — the classic competition-Math format. The
solver runs `extract_boxed` to pull the answer out, then parses the result
via `parse_and_validate`.

Note: because the response format is `"text"` (not `"json_object"`), the
LLM client will not request JSON mode. The framework post-processes the
output to coerce it into a `MathSolution` JSON.
"""

from llm_math.prompts._base import PromptTemplate
from llm_math.prompts import register_prompt

register_prompt(
    PromptTemplate(
        name="boxed_v1",
        description="Free-form: solve in prose, end with \\\\boxed{answer}. Post-processed into JSON.",
        response_format="text",
        system=(
            "You are a math problem solver. Solve the problem carefully and "
            "express your final answer using the LaTeX boxed notation: "
            "`\\boxed{<your final answer>}`.\n\n"
            "Rules:\n"
            "- Show your work, but the boxed expression at the end must be the "
            "concise final answer (a number, an expression, a tuple, a set, etc.).\n"
            "- Do NOT output JSON. Just prose and the boxed answer.\n"
            "- If the problem is a proof, the boxed content should be a short "
            "statement of the proved conclusion."
        ),
        user_template=(
            "Problem:\n{problem}\n\n"
            "Solve it and put your final answer in `\\boxed{{...}}`."
        ),
    )
)
