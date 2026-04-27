from __future__ import annotations

import argparse
import os
from typing import Any

import mlflow
import pandas as pd
from mlflow.genai import evaluate
from mlflow.genai.datasets import EvaluationDataset, get_dataset, search_datasets
from mlflow.genai.scorers import list_scorers

from panda_matching.api.routes import ChatRequest, chat_agent

DEFAULT_DATASET_NAME = "panda_chatbot_eval_v1"
SMOKE_MODE_EXCLUDED_SCORERS = {"panda_tool_call_correctness"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline MLflow GenAI evaluation for the panda chatbot."
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
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"MLflow evaluation dataset name. Defaults to {DEFAULT_DATASET_NAME!r}.",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Optional MLflow dataset ID to use when multiple datasets share the same name.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for a smaller smoke-test run.",
    )
    parser.add_argument(
        "--run-name",
        default="baseline-chatbot-eval",
        help="MLflow run name for this evaluation.",
    )
    parser.add_argument(
        "--exclude-scorer",
        action="append",
        default=[],
        help="Scorer name to exclude. Repeatable.",
    )
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help="Exclude heavy scorers automatically for fast local validation runs.",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help=(
            "Only evaluate rows whose expectations include expected_facts "
            "or expected_response."
        ),
    )
    return parser.parse_args()


def _require_env(value: str | None, label: str) -> str:
    if value:
        return value
    raise ValueError(f"Missing {label}. Pass it explicitly or set the environment variable.")


def _is_labeled_expectation(expectations: Any) -> bool:
    if not isinstance(expectations, dict):
        return False
    expected_response = expectations.get("expected_response")
    expected_facts = expectations.get("expected_facts")
    if isinstance(expected_response, str) and expected_response.strip():
        return True
    if isinstance(expected_facts, list) and any(
        isinstance(item, str) and item.strip() for item in expected_facts
    ):
        return True
    return False


def _resolve_dataset(
    *,
    dataset_name: str,
    dataset_id: str | None,
    experiment_id: str,
) -> EvaluationDataset:
    if dataset_id:
        return get_dataset(dataset_id=dataset_id)

    try:
        return get_dataset(name=dataset_name)
    except Exception as exc:
        if "Multiple datasets found with name" not in str(exc):
            raise
        datasets = search_datasets(experiment_ids=experiment_id)
        matching = [dataset for dataset in datasets if dataset.name == dataset_name]
        if not matching:
            raise
        matching.sort(
            key=lambda dataset: (
                dataset.created_time or dataset.create_time or 0,
                dataset.dataset_id,
            ),
            reverse=True,
        )
        selected = matching[0]
        print(
            "Multiple datasets matched name "
            f"{dataset_name!r}; using latest dataset_id={selected.dataset_id}."
        )
        return get_dataset(dataset_id=selected.dataset_id)


def _load_dataset_frame(
    dataset_name: str,
    dataset_id: str | None,
    experiment_id: str,
    max_examples: int | None,
    *,
    labeled_only: bool,
) -> pd.DataFrame:
    dataset = _resolve_dataset(
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
    )
    frame = dataset.to_df()
    if labeled_only:
        frame = frame[frame["expectations"].apply(_is_labeled_expectation)].copy()
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("--max-examples must be positive when provided.")
        frame = frame.head(max_examples).copy()
    return frame


def _ensure_required_judge_credentials(scorers: list[Any]) -> None:
    needs_openai = any(
        str(getattr(scorer, "model", "") or "").startswith("openai:/")
        for scorer in scorers
    )
    if needs_openai and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is required for the registered OpenAI-backed scorers. "
            "Export it in the current shell before running paid benchmarks."
        )


def _log_missing_aggregate_metrics(result: Any) -> dict[str, float]:
    tables = getattr(result, "tables", None)
    if not isinstance(tables, dict):
        return {}

    eval_results = tables.get("eval_results")
    if not isinstance(eval_results, pd.DataFrame):
        return {}

    recovered_metrics: dict[str, float] = {}
    for column in eval_results.columns:
        if not column.endswith("/value"):
            continue
        scorer_name = column[: -len("/value")]
        series = eval_results[column].dropna()
        if series.empty:
            continue

        def _normalize(value: Any) -> float | None:
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"yes", "true", "pass", "passed", "correct"}:
                    return 1.0
                if lowered in {"no", "false", "fail", "failed", "incorrect"}:
                    return 0.0
            return None

        normalized = [score for score in (_normalize(item) for item in series) if score is not None]
        if not normalized:
            continue
        metric_name = f"{scorer_name}/mean"
        metric_value = sum(normalized) / len(normalized)
        recovered_metrics[metric_name] = metric_value
        mlflow.log_metric(metric_name, metric_value)
    return recovered_metrics


@mlflow.trace(name="baseline_eval_predict")
def _predict_fn(message: str) -> str:
    result = chat_agent(ChatRequest(message=message))
    return result.response


def main() -> None:
    args = _parse_args()
    tracking_uri = _require_env(args.tracking_uri, "MLFLOW_TRACKING_URI")
    experiment_id = _require_env(args.experiment_id, "MLFLOW_EXPERIMENT_ID")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=experiment_id)

    dataset_frame = _load_dataset_frame(
        args.dataset_name,
        args.dataset_id,
        experiment_id,
        args.max_examples,
        labeled_only=args.labeled_only,
    )
    scorers = list_scorers(experiment_id=experiment_id)
    if not scorers:
        raise ValueError(
            "No scorers registered for this experiment. "
            "Run scripts/register_mlflow_scorers.py first."
        )
    excluded = set(args.exclude_scorer)
    if args.smoke_mode:
        excluded.update(SMOKE_MODE_EXCLUDED_SCORERS)
    if args.exclude_scorer or args.smoke_mode:
        scorers = [scorer for scorer in scorers if scorer.name not in excluded]
    if not scorers:
        raise ValueError("No scorers remain after applying --exclude-scorer filters.")
    if dataset_frame.empty:
        raise ValueError("No dataset rows remain after applying the current filters.")
    _ensure_required_judge_credentials(scorers)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_param("dataset_name", args.dataset_name)
        mlflow.log_param("num_examples", len(dataset_frame))
        mlflow.log_param("smoke_mode", args.smoke_mode)
        mlflow.log_param("labeled_only", args.labeled_only)
        if excluded:
            mlflow.log_param("excluded_scorers", ",".join(sorted(excluded)))
        if args.max_examples is not None:
            mlflow.log_param("max_examples", args.max_examples)

        result = evaluate(
            data=dataset_frame,
            scorers=scorers,
            predict_fn=_predict_fn,
        )

        print(f"Evaluation run id: {run.info.run_id}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Examples evaluated: {len(dataset_frame)}")

        metrics = getattr(result, "metrics", None)
        recovered_metrics = _log_missing_aggregate_metrics(result)
        if isinstance(metrics, dict):
            for key, value in recovered_metrics.items():
                metrics.setdefault(key, value)
            print("Metrics:")
            for key in sorted(metrics):
                print(f"  {key}: {metrics[key]}")
        elif recovered_metrics:
            print("Metrics:")
            for key in sorted(recovered_metrics):
                print(f"  {key}: {recovered_metrics[key]}")

        tables = getattr(result, "tables", None)
        if isinstance(tables, dict):
            print("Result tables:")
            for key in sorted(tables):
                print(f"  {key}")


if __name__ == "__main__":
    main()
