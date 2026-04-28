# Evaluation Notes

This project now uses two evaluation layers.

## 1. Free regression suite

Purpose:

* catch routing regressions
* catch formatting regressions
* verify deterministic fallback behavior
* avoid paid model calls

Command:

```bash
make eval-free
```

Equivalent:

```bash
uv run python scripts/run_offline_regression_suite.py --local-only
```

What it checks:

* no exceptions
* no raw JSON
* no help-text fallback for covered prompts
* non-empty answers
* expected counts where labeled
* expected ranking names
* pairwise winner checks
* approximate age wording

Status:

* the offline regression suite passes end to end on the current repo state

## 2. Paid LLM quality benchmark

Purpose:

* evaluate reasoning-heavy answers
* validate usefulness, correctness, groundedness, and safety
* check quality after prompt, routing, or answer-composition changes

Command pattern:

```bash
DATASET=panda_chatbot_llm_benchmark_v1 RUN_NAME=llm-benchmark-smoke make eval-paid
```

Requirements:

* `OPENAI_API_KEY`
* `MLFLOW_TRACKING_URI`
* `MLFLOW_EXPERIMENT_ID`
* `DATABRICKS_LLM_ENABLED=true`

Status:

* the curated paid benchmark has been run successfully in batches
* Databricks `429 Too Many Requests` made small-batch execution necessary
* the eval runner now supports dataset slicing through:
  * `--start-index`
  * `--max-examples`

Example:

```bash
uv run python scripts/run_baseline_eval.py \
  --dataset-name panda_chatbot_llm_benchmark_remaining_v1 \
  --start-index 3 \
  --max-examples 2 \
  --run-name llm-benchmark-batch2 \
  --exclude-scorer panda_tool_call_correctness
```

## Routing policy validated through evaluation

The chatbot now follows a split strategy:

* deterministic routing/rendering for:
  * profile questions
  * count questions
  * unsupported requests
  * common ranking/comparison prompts where SQL-backed explanation is sufficient
* LLM-assisted responses for:
  * richer reasoning-heavy prompts
  * prompts that benefit from conversational synthesis

This design keeps factual answers grounded and reduces hallucination risk while still using the LLM where it adds value.
