from __future__ import annotations

import argparse
import os
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import (
    Safety,
    ToolCallCorrectness,
    delete_scorer,
    list_scorers,
)

_default_endpoint = os.getenv("DATABRICKS_LLM_ENDPOINT")


def _default_judge_model() -> str:
    explicit = os.getenv("MLFLOW_JUDGE_MODEL")
    if explicit:
        return explicit
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri.startswith("http://127.0.0.1") or tracking_uri.startswith("file:"):
        return "openai:/gpt-4.1-mini"
    if _default_endpoint:
        return f"endpoints:/{_default_endpoint}"
    return "databricks"


DEFAULT_JUDGE_MODEL = _default_judge_model()

TASK_AWARE_CORRECTNESS_INSTRUCTIONS = """
Evaluate whether the assistant answer is correct for this panda matching task.

User inputs: {{ inputs }}
Assistant outputs: {{ outputs }}
Expected information: {{ expectations }}
Execution trace: {{ trace }}

Use the expectations as the task-specific standard. Follow these rules:
- Treat materially consistent paraphrases as correct.
- Treat approximate wording as acceptable when the underlying fact is approximate.
  For example, "approximately 12 years old" and "about 12 years old" should be
  treated as correct if the expected fact is approximate.
- Be strict for exact counts, exact named candidates, and explicit comparisons
  when the expectation clearly requires them.
- Do not require extra detail that the question did not ask for.
- For simple profile questions such as age, location, or health, a short
  direct answer is correct if it matches the expected facts.

Return true only if the answer is materially correct according to the task
expectations.
"""

TASK_AWARE_GUIDELINES_INSTRUCTIONS = """
Evaluate whether the assistant answer follows the right quality standard for
this specific panda matching task.

User inputs: {{ inputs }}
Assistant outputs: {{ outputs }}
Expected information: {{ expectations }}
Execution trace: {{ trace }}

Use task_type and required_behaviors from expectations when present.

Apply these task-aware rules:
- For profile_info, profile_health, and profile_fun_fact:
  - reward concise direct answers grounded in profile data
  - do not require ranking rationale, breeding rationale, or extra explanation
    unless explicitly asked
- For top_match_ranking and match_explanation:
  - reward readable ranking or explanation with short grounded reasons
- For pairwise_comparison and challenge_the_ranking:
  - reward clear comparison, tradeoff discussion, and direct handling of the user's concern
- For population_count:
  - reward short direct factual answers
- For unsupported_request:
  - reward brief, clear redirection without hallucination

Across all tasks:
- do not require raw JSON or field dumps
- prefer readability and concision
- do not penalize an answer for omitting irrelevant task types of reasoning

Return true only if the answer follows the appropriate quality standard for the
specific task.
"""

TASK_AWARE_USEFULNESS_INSTRUCTIONS = """
Evaluate whether the assistant answer is useful for this specific panda
matching task.

User inputs: {{ inputs }}
Assistant outputs: {{ outputs }}
Expected information: {{ expectations }}
Execution trace: {{ trace }}

Judge usefulness relative to the task:
- A useful profile answer is concise and directly answers the question.
- A useful ranking answer presents candidates clearly and includes short grounded reasons.
- A useful comparison answer makes a decision or frames the tradeoff clearly.
- A useful count answer is direct and compact.

Do not require ranking or breeding rationale for simple profile questions
unless the user asked for it.
Return true only if the answer would feel helpful and appropriately scoped to a
reasonable user.
"""

GROUNDED_IN_TOOL_OUTPUT_INSTRUCTIONS = """
Evaluate whether the assistant answer is grounded in the available evidence.

User inputs: {{ inputs }}
Assistant outputs: {{ outputs }}
Expected information: {{ expectations }}
Execution trace: {{ trace }}

Return true only if the answer stays faithful to the retrieved evidence and
does not invent facts, reasons, counts, locations, or profile details.
Do not penalize concise wording if it remains faithful to the evidence."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register MLflow GenAI scorers for panda chatbot evaluation."
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI. Defaults to $MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--experiment-id",
        default=os.getenv("MLFLOW_EXPERIMENT_ID"),
        help="MLflow experiment ID. Defaults to $MLFLOW_EXPERIMENT_ID.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "Judge model identifier for built-in and custom LLM scorers. "
            "Defaults to $MLFLOW_JUDGE_MODEL or 'databricks'."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate scorers if they already exist.",
    )
    parser.add_argument(
        "--skip-tool-call-correctness",
        action="store_true",
        help="Skip ToolCallCorrectness scorer registration.",
    )
    return parser.parse_args()


def _require_env(value: str | None, label: str) -> str:
    if value:
        return value
    raise ValueError(f"Missing {label}. Pass it explicitly or set the environment variable.")


def _existing_scorer_names(experiment_id: str) -> set[str]:
    return {scorer.name for scorer in list_scorers(experiment_id=experiment_id)}


def _register_scorer(
    scorer: Any,
    *,
    experiment_id: str,
    existing_names: set[str],
    force: bool,
) -> None:
    if scorer.name in existing_names:
        if not force:
            print(f"skip: {scorer.name} already exists")
            return
        delete_scorer(name=scorer.name, experiment_id=experiment_id, version="all")
        print(f"deleted: {scorer.name}")

    scorer.register()
    print(f"registered: {scorer.name}")


def main() -> None:
    args = _parse_args()
    tracking_uri = _require_env(args.tracking_uri, "MLFLOW_TRACKING_URI")
    experiment_id = _require_env(args.experiment_id, "MLFLOW_EXPERIMENT_ID")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)

    scorer_model = args.judge_model
    existing_names = _existing_scorer_names(experiment_id)

    try:
        scorers = [
            make_judge(
                name="panda_correctness",
                model=scorer_model,
                feedback_value_type=bool,
                aggregations=["mean"],
                instructions=TASK_AWARE_CORRECTNESS_INSTRUCTIONS,
            ),
            make_judge(
                model=scorer_model,
                name="panda_guidelines",
                feedback_value_type=bool,
                aggregations=["mean"],
                instructions=TASK_AWARE_GUIDELINES_INSTRUCTIONS,
            ),
            Safety(
                name="panda_safety",
                model=scorer_model,
            ),
            make_judge(
                name="grounded_in_tool_output",
                model=scorer_model,
                feedback_value_type=bool,
                aggregations=["mean"],
                instructions=GROUNDED_IN_TOOL_OUTPUT_INSTRUCTIONS,
            ),
            make_judge(
                name="answer_usefulness",
                model=scorer_model,
                feedback_value_type=bool,
                aggregations=["mean"],
                instructions=TASK_AWARE_USEFULNESS_INSTRUCTIONS,
            ),
        ]
        if not args.skip_tool_call_correctness:
            scorers.insert(
                3,
                ToolCallCorrectness(
                    name="panda_tool_call_correctness",
                    model=scorer_model,
                ),
            )
    except MlflowException as exc:
        if scorer_model == "databricks":
            raise MlflowException(
                "Judge model 'databricks' requires the optional databricks-agents package. "
                "Use an explicit judge model instead, for example "
                "'endpoints:/<your-serving-endpoint>' via --judge-model or "
                "MLFLOW_JUDGE_MODEL."
            ) from exc
        raise

    for scorer in scorers:
        _register_scorer(
            scorer,
            experiment_id=experiment_id,
            existing_names=existing_names,
            force=args.force,
        )

    print(f"Scorer registration complete for experiment {experiment_id}.")


if __name__ == "__main__":
    main()
