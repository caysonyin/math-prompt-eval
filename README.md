# math-prompt-eval

A bare-LLM single-shot math solver plus a complete prompt-engineering
framework around it. The solver makes **one** chat completion per
problem — no tools, no retries, no verifier, no candidate selection —
and emits a `MathSolution` JSON in the same schema as the upstream
`mathsolve-agent` `MathSolverAgent`. The framework wraps that solver
with a prompt template registry, a sweep runner, a comparison report
generator, and a sweep-level validator.

The aim: **make A/B prompt experiments as cheap as a single CLI
command.** Drop a new template file, run the sweep, read the report.

## Install

```bash
cd /home/caysonyin/Projects/llm
uv sync --extra dev            # adds pytest, pandas, pyarrow
```

Runtime dependencies: `pydantic>=2`, `requests`, `openpyxl`, `sympy`.
The vendored `parser.py`, `normalizer.py`, and `validator.py` come
from the upstream `mathsolve-agent/math_prove/` and are byte-identical
copies (with attribution headers) — no runtime dependency on the
upstream fork.

## Configure

Set the model and endpoint via env vars (or `--api-key` / `--api-base`
on each CLI invocation). For Intern-S1:

```bash
export OPENAI_API_KEY="<intern-s1-token, no Bearer prefix>"
export LLM_API_BASE="https://chat.intern-ai.org.cn/api/v1/chat/completions"
export MODEL_NAME="intern-s1"
```

The framework also reads `DEEPSEEK_API_KEY` (or `MODEL_API_KEY`) when
`--llm-judge` is enabled without explicit judge credentials.

## Quickstart (5 commands)

Prepare your own input JSONL file, for example `problems.jsonl`. Each
row should contain at least `problem_id` and `problem_text`; evaluation
also needs an `expected_answer` field in the expected-answer file.

```bash
# 1) Run a single demo problem (one LLM call)
llm-math-run --demo --prompt naive_v1

# 2) Run a batch on a problem set
llm-math-run --input problems.jsonl --prompt naive_v1 --limit 6

# 3) Validate the resulting JSONL against expected answers
llm-math-evaluate \
    --results outputs/naive_v1/intern-s1/20260604_120000/results.jsonl \
    --expected problems.jsonl

# 4) Sweep every registered prompt over the same input
llm-math-sweep --input problems.jsonl --limit 10

# 5) Compare all sweeps sharing a timestamp, with markdown
llm-math-compare-prompts \
    --results outputs/naive_v1/intern-s1/<ts>/results.jsonl --label naive_v1 \
    --results outputs/cot_v1/intern-s1/<ts>/results.jsonl   --label cot_v1   \
    --results outputs/boxed_v1/intern-s1/<ts>/results.jsonl --label boxed_v1 \
    --expected problems.jsonl --markdown
```

Direct module invocations such as `uv run python -m runtime.run` and
`uv run python -m evaluation.evaluate` are interchangeable with the
console-script names listed above.

## Output directory convention

Every `llm-math-run` (and every per-prompt run inside `llm-math-sweep`)
writes to a fresh, isolated directory:

```
outputs/
└── {prompt_name}/
    └── {model}/
        └── {timestamp}/             # YYYYMMDD_HHMMSS, shared across one sweep
            ├── results.jsonl        # streaming JSONL (one MathSolution per line)
            ├── results.json         # merged array, written at the end of the run
            ├── run_summary.json     # counts, fallback, latency, etc.
            ├── validation_report.json   # only after llm-math-evaluate[-sweep]
            └── logs/
                └── {problem_id}.json    # per-problem request/response log
```

This isolates every (prompt, model, timestamp) combination on disk so
prompt A/B runs never clobber each other, and `compare_prompts` can
glob across the tree.

If you pass `--output` explicitly, that path is honored verbatim and
the other three default to `<output.parent>/<sibling>` for backward
compatibility.

## Add a new prompt template

Drop a single file in `runtime/prompts/` — no imports, no edits
elsewhere. Each module calls `register_prompt(...)` once at import
time, and `pkgutil.iter_modules` auto-discovers the new file at
package import.

```python
# runtime/prompts/naive_v2.py
from runtime.prompts._base import PromptTemplate
from runtime.prompts import register_prompt

register_prompt(PromptTemplate(
    name="naive_v2",
    description="Adds a 'show your work' preamble; expects a JSON object at the end.",
    system="You are a math problem solver. Show your work briefly, then "
           "output ONLY a single JSON object matching the MathSolution schema.",
    user_template="Problem: {problem}",
))
```

`--prompt naive_v2` immediately appears as a CLI choice. Validation at
`register_prompt` time rejects malformed names and `response_format`
values, and warns on missing `{problem}` placeholders or unknown
`{...}` tokens (which would crash `str.format()` at render time).

Three built-in templates ship with the framework:

| Name       | response_format | Description                                              |
|------------|-----------------|----------------------------------------------------------|
| `naive_v1` | `json_object`   | Problem in, MathSolution JSON out. No reasoning scaffold. |
| `cot_v1`   | `json_object`   | Step-by-step reasoning prose, then a MathSolution JSON.   |
| `boxed_v1` | `text`          | Free-form prose ending in `\boxed{...}`. Post-processed.  |

## A/B prompt experiments (end-to-end)

The full workflow for a 3-prompt A/B run is three commands and one
markdown report. Example on a problem set with `--dry-run` to
illustrate the layout without burning API credits:

```bash
# 1) Sweep three prompts over the same input, sharing one timestamp.
llm-math-sweep \
    --input problems.jsonl \
    --only naive_v1,cot_v1,boxed_v1 \
    --model intern-s1 \
    --limit 100 \
    --dry-run
# → outputs/{naive_v1,cot_v1,boxed_v1}/intern-s1/<ts>/results.jsonl
# → outputs/.../sweep_summary.json

# 2) Validate every per-prompt JSONL against expected answers.
llm-math-evaluate-sweep \
    --sweep-dir outputs \
    --expected problems.jsonl
# → outputs/{prompt}/intern-s1/<ts>/validation_report.json
# → outputs/.../sweep_eval_summary.json

# 3) Cross-compare the runs, with a side-by-side markdown table.
llm-math-compare-prompts \
    --results outputs/naive_v1/intern-s1/<ts>/results.jsonl  --label naive_v1  \
    --results outputs/cot_v1/intern-s1/<ts>/results.jsonl    --label cot_v1    \
    --results outputs/boxed_v1/intern-s1/<ts>/results.jsonl  --label boxed_v1  \
    --expected problems.jsonl \
    --markdown
# → prompt_comparison_report.json
# → prompt_comparison_report.md
```

The emitted markdown looks like:

```markdown
# Prompt comparison (3 runs)

| label     | total | local_acc | judge_acc | fallback  | median_latency (s) |
|-----------|------:|----------:|----------:|-----------|-------------------:|
| naive_v1  |   100 |    79.37% |       n/a | 3 (0.6%)  |              3.42  |
| cot_v1    |   100 |    82.14% |       n/a | 1 (0.2%)  |              5.18  |
| boxed_v1  |   100 |    78.97% |       n/a | 5 (1.0%)  |              4.20  |

## Local-accuracy deltas (rows − cols, percentage points)

| row \ col   | naive_v1 | cot_v1 | boxed_v1 |
|-------------|---------:|-------:|---------:|
| naive_v1    |   +0.00  | -2.77pp | +0.40pp  |
| cot_v1      |  +2.77pp |  +0.00  | +3.17pp  |
| boxed_v1    |  -0.40pp | -3.17pp |  +0.00   |
```

The JSON report (`prompt_comparison_report.json`) carries the same
data plus `per_problem` (every pid with the labels that got it right)
and a `winner` lookup table that downstream tools can diff against a
single baseline to find regressions.

### LLM-judge tier

To upgrade the local-equivalence accuracy column to multi-judge
accuracy, pass `--llm-judge` plus up to three judge models:

```bash
llm-math-compare-prompts \
    --results a.jsonl --label A \
    --results b.jsonl --label B \
    --expected expected.jsonl \
    --llm-judge \
    --judge-model  deepseek-chat  --judge-api-key  $DEEPSEEK_API_KEY \
    --judge-api-base https://api.deepseek.com/chat/completions \
    --judge-model2 intern-s1      --judge-api-key2 $INTERN_KEY \
    --judge-api-base2 $LLM_API_BASE
```

When the judge tier is enabled, `judge_acc` is the multi-judge majority
verdict and the `winner` map is determined by `llm_judge_correct` rather
than local equivalence. With `--llm-judge` off (the default), both
fall back to the local equivalence tier.

The same flags work on `llm-math-evaluate` and `llm-math-evaluate-sweep`
for per-prompt validation.

## Dry-run mode

Smoke-test the harness without an API key:

```bash
llm-math-run --input problems.jsonl --dry-run --limit 3
llm-math-sweep --input problems.jsonl --only naive_v1 --dry-run --limit 3
```

`--dry-run` substitutes a stub solver that emits a `fallback_solution`
for every row, so the per-prompt directory layout, the sweep summary,
and the validation report all exercise end-to-end without API calls.

## Architecture

```
runtime/                     # run-model module
├── llm_client.py            # OpenAI-compatible chat client with retry + lock
├── prompts/                 # pkgutil-discovered @register_prompt('name') templates
├── solver.py                # BareLLMSolver: one prompt, one chat call, one MathSolution
├── runner.py                # ThreadPoolExecutor parallel batch runner
├── run.py                   # CLI: --demo, --dry-run, batch output layout
└── prompt_sweep.py          # CLI: run N prompts over one input
evaluation/                  # evaluation module
├── evaluate.py              # CLI: validate one results file vs. expected
├── evaluate_sweep.py        # CLI: validate every results.jsonl under outputs/
└── compare_prompts.py       # CLI: N-prompt A/B comparison report
shared/                      # shared infrastructure
├── io.py                    # load_problems, resume helpers, RpmLimiter
└── vendor/                  # vendored parser, normalizer, validator
tests/                       # pytest unit tests (62 tests)
```

`BareLLMSolver.solve()` flow:

1. `RpmLimiter.acquire()` — sliding-window 60s rate limit
2. render the selected prompt template (`get_prompt(name)`)
3. one `requests.post` to the OpenAI-compatible chat endpoint
4. for `response_format=text` prompts, run `extract_boxed` first
5. `parse_and_validate` → `MathSolution`; on any failure, `fallback_solution`
6. stash a structured `last_run_log` for the per-problem log writer

The same `MathSolution` schema feeds the vendored `validate_results`
function, so the evaluator and the comparison flow share a single
code path with the upstream.

## Tests

```bash
uv run pytest tests/ -q
```

62 unit tests cover the prompt registry (auto-discovery, validation,
placeholder warnings), IO helpers, the solver's parse / fallback /
LaTeX-escape-repair paths, the runner's resume and exception handling,
the boxed-prompt post-processing, the nested output layout in `run.py`,
and end-to-end dry-run behaviour for `compare_prompts`, `prompt_sweep`,
and `evaluate_sweep`.

## Vendoring notes

The three vendored files in `shared/vendor/` are byte-identical
copies of:

- `mathsolve-agent/math_prove/parser.py`
- `mathsolve-agent/math_prove/normalizer.py`
- `mathsolve-agent/math_prove/validator.py`

Each starts with an attribution header. The relative imports
(`from .normalizer import …`, `from .parser import …`) work unchanged
because the three files are siblings inside the `shared.vendor`
package. There is no `lagent` runtime dependency in any vendored file.

If the upstream changes, the vendored copies need to be re-synced
manually. A `test_vendor_drift.py` to flag this is a possible future
addition.

## Limitations (deferred)

- **McNemar / paired significance tests** — the comparison report shows
  accuracy deltas and a winner map; with the current data sizes
  (your problem-set size) the per-prompt deltas are usually clearly readable
  without a paired test.
- **Per-token / per-run cost tracking** — the framework records
  `latency_seconds` in `run_summary.json` but not dollar cost; computing
  it would require an explicit price table per model.
- **Automatic prompt tuning (DSPy-style)** — adding a new prompt still
  requires a human in the loop to author the template.
- **Cross-sweep resume** — `llm-math-run --resume` resumes a single
  prompt's JSONL; a killed sweep cannot be resumed as a single unit
  (re-running the sweep will re-process problems for prompts whose
  results.jsonl was never written).
- **Multi-judge ensemble scoring beyond 3 models** — the
  `LLMJudgeConfig` supports up to 3 judges, matching the upstream
  3-model majority-vote evaluator. Larger ensembles would need a
  separate `LLMJudgeConfig` variant.
- **Web UI / dashboard** — the markdown table emitted by
  `llm-math-compare-prompts --markdown` is the only built-in
  visualization.
