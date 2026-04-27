from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.genai import evaluate
from mlflow.genai.scorers import scorer

from panda_matching.api.routes import ChatRequest, chat_agent

DEFAULT_DATASET_FILE = "data/evaluation/panda_chatbot_eval_v1.json"
APPROXIMATE_WORDS = ("approximately", "about", "around", "roughly")
RAW_JSON_PATTERNS = (
    re.compile(r"^\s*[\[{]"),
    re.compile(r"```json", re.IGNORECASE),
    re.compile(r'"(?:panda_name|matches|source_view|score_breakdown_json)"\s*:'),
)
DEBUG_TEXT_PATTERNS = ("intent=", "session_id", "show raw data", "source_view")
COMPARATIVE_WORDS = (
    "better",
    "stronger",
    "higher",
    "above",
    "rank",
    "top",
    "best",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a free offline regression suite across the full panda chatbot dataset."
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
        "--dataset-file",
        default=DEFAULT_DATASET_FILE,
        help=f"Local JSON dataset file. Defaults to {DEFAULT_DATASET_FILE!r}.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for a smaller local run.",
    )
    parser.add_argument(
        "--run-name",
        default="offline-regression-suite",
        help="MLflow run name for this evaluation.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run the offline regression suite without MLflow tracking for a faster loop.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit with code 0 even when one or more regression checks fail.",
    )
    return parser.parse_args()


def _require_env(value: str | None, label: str) -> str:
    if value:
        return value
    raise ValueError(f"Missing {label}. Pass it explicitly or set the environment variable.")


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _response_text(outputs: Any) -> str:
    if isinstance(outputs, dict):
        response = outputs.get("response")
        return response if isinstance(response, str) else ""
    return outputs if isinstance(outputs, str) else ""


def _extract_exception(outputs: Any) -> str | None:
    if isinstance(outputs, dict):
        error = outputs.get("exception")
        return error if isinstance(error, str) and error.strip() else None
    return None


def _load_dataset_frame(dataset_file: str, max_examples: int | None) -> pd.DataFrame:
    path = Path(dataset_file)
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload)
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("--max-examples must be positive when provided.")
        frame = frame.head(max_examples).copy()
    return frame


def _contains_all_names(text: str, names: list[str]) -> bool:
    normalized = _normalize_text(text)
    return all(_normalize_text(name) in normalized for name in names)


def _extract_count_requirements(expectations: dict[str, Any]) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    for fact in expectations.get("expected_facts", []):
        if not isinstance(fact, str):
            continue
        match = re.search(r"\b(\d+)\b", fact)
        if not match:
            continue
        fact_lower = fact.lower()
        if "male" in fact_lower:
            requirements.append((match.group(1), "male"))
        elif "female" in fact_lower:
            requirements.append((match.group(1), "female"))
        elif "eligible" in fact_lower:
            requirements.append((match.group(1), "eligible"))
        else:
            requirements.append((match.group(1), "pandas"))
    return requirements


def _expected_winner(expectations: dict[str, Any]) -> str | None:
    for fact in expectations.get("expected_facts", []):
        if not isinstance(fact, str):
            continue
        if "better match" in fact.lower() or "ranks above" in fact.lower():
            winner, _, _ = fact.partition(" is ")
            if winner.strip():
                return winner.strip()
    return None


@scorer(name="no_exceptions", aggregations=["mean"])
def no_exceptions(outputs: Any) -> bool:
    return _extract_exception(outputs) is None


@scorer(name="non_empty_answer", aggregations=["mean"])
def non_empty_answer(outputs: Any) -> bool:
    return bool(_response_text(outputs).strip())


@scorer(name="no_raw_json", aggregations=["mean"])
def no_raw_json(outputs: Any) -> bool:
    text = _response_text(outputs)
    if not text.strip():
        return False
    return not any(pattern.search(text) for pattern in RAW_JSON_PATTERNS)


@scorer(name="no_debug_artifacts", aggregations=["mean"])
def no_debug_artifacts(outputs: Any) -> bool:
    normalized = _normalize_text(_response_text(outputs))
    return not any(pattern in normalized for pattern in DEBUG_TEXT_PATTERNS)


@scorer(name="expected_counts_match", aggregations=["mean"])
def expected_counts_match(outputs: Any, expectations: dict[str, Any]) -> bool:
    if expectations.get("task_type") != "population_count":
        return True
    text = _normalize_text(_response_text(outputs))
    requirements = _extract_count_requirements(expectations)
    if not requirements:
        return True
    return all(number in text and keyword in text for number, keyword in requirements)


@scorer(name="expected_names_present", aggregations=["mean"])
def expected_names_present(outputs: Any, expectations: dict[str, Any]) -> bool:
    names = expectations.get("should_mention_names")
    if not isinstance(names, list) or not names:
        return True
    expected_names = [name for name in names if isinstance(name, str)]
    return _contains_all_names(_response_text(outputs), expected_names)


@scorer(name="pairwise_winner_match", aggregations=["mean"])
def pairwise_winner_match(outputs: Any, expectations: dict[str, Any]) -> bool:
    if expectations.get("task_type") != "pairwise_comparison":
        return True
    winner = _expected_winner(expectations)
    if not winner:
        return True
    text = _normalize_text(_response_text(outputs))
    return _normalize_text(winner) in text and any(word in text for word in COMPARATIVE_WORDS)


@scorer(name="approximate_age_wording", aggregations=["mean"])
def approximate_age_wording(outputs: Any, expectations: dict[str, Any]) -> bool:
    expected_facts = expectations.get("expected_facts", [])
    approximate_fact = next(
        (
            fact
            for fact in expected_facts
            if isinstance(fact, str) and any(word in fact.lower() for word in APPROXIMATE_WORDS)
        ),
        None,
    )
    if not approximate_fact:
        return True
    number_match = re.search(r"\b(\d+)\b", approximate_fact)
    if not number_match:
        return True
    text = _normalize_text(_response_text(outputs))
    return number_match.group(1) in text and any(word in text for word in APPROXIMATE_WORDS)


PROGRAMMATIC_SCORERS = [
    no_exceptions,
    non_empty_answer,
    no_raw_json,
    no_debug_artifacts,
    expected_counts_match,
    expected_names_present,
    pairwise_winner_match,
    approximate_age_wording,
]
PROGRAMMATIC_SCORER_NAMES = {scorer.name for scorer in PROGRAMMATIC_SCORERS}


@mlflow.trace(name="offline_regression_predict")
def _predict_fn(message: str) -> dict[str, Any]:
    try:
        result = chat_agent(ChatRequest(message=message))
    except Exception as exc:  # pragma: no cover - explicit regression capture
        return {
            "response": "",
            "intent": None,
            "exception": f"{type(exc).__name__}: {exc}",
        }
    return {
        "response": result.response,
        "intent": result.intent,
        "exception": None,
    }


def _build_failure_report(
    eval_results: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    failures: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    scorer_columns = [
        column
        for column in eval_results.columns
        if column.endswith("/value")
        and column[: -len("/value")] in PROGRAMMATIC_SCORER_NAMES
    ]
    for scorer_column in scorer_columns:
        scorer_name = scorer_column[: -len("/value")]
        counts[scorer_name] = 0

    for _, row in eval_results.iterrows():
        request_payload = row.get("request")
        message = request_payload.get("message") if isinstance(request_payload, dict) else None
        response_payload = row.get("response")
        response = _response_text(response_payload)
        for scorer_column in scorer_columns:
            scorer_name = scorer_column[: -len("/value")]
            value = row.get(scorer_column)
            if value is False or value == 0:
                counts[scorer_name] += 1
                failures.append(
                    {
                        "message": message,
                        "scorer": scorer_name,
                        "response": response,
                    }
                )
    return failures, counts


def _invoke_programmatic_scorer(
    scorer_obj: Any,
    *,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    expectations: dict[str, Any],
) -> bool:
    kwargs: dict[str, Any] = {}
    for name in inspect.signature(scorer_obj).parameters:
        if name == "inputs":
            kwargs[name] = inputs
        elif name == "outputs":
            kwargs[name] = outputs
        elif name == "expectations":
            kwargs[name] = expectations
    return bool(scorer_obj(**kwargs))


def _run_local_only(
    dataset_frame: pd.DataFrame,
    *,
    allow_failures: bool,
) -> None:
    failures: list[dict[str, Any]] = []
    totals = {scorer.name: 0 for scorer in PROGRAMMATIC_SCORERS}

    for _, row in dataset_frame.iterrows():
        inputs = row["inputs"]
        expectations = row["expectations"]
        message = inputs["message"]
        outputs = _predict_fn(message)
        for scorer_obj in PROGRAMMATIC_SCORERS:
            passed = _invoke_programmatic_scorer(
                scorer_obj,
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
            )
            totals[scorer_obj.name] += 1 if passed else 0
            if not passed:
                failures.append(
                    {
                        "message": message,
                        "scorer": scorer_obj.name,
                        "response": _response_text(outputs),
                    }
                )

    print(f"Examples evaluated: {len(dataset_frame)}")
    print("Metrics:")
    for scorer_name in sorted(totals):
        metric = totals[scorer_name] / len(dataset_frame)
        print(f"  {scorer_name}/mean: {metric}")

    failure_counts = {scorer.name: 0 for scorer in PROGRAMMATIC_SCORERS}
    for failure in failures:
        failure_counts[failure["scorer"]] += 1

    print("Failure counts:")
    for scorer_name in sorted(failure_counts):
        print(f"  {scorer_name}: {failure_counts[scorer_name]}")

    if failures:
        artifact_path = Path("tmp/offline_regression_failures.json")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(failures, indent=2))
        print("Failure details written to tmp/offline_regression_failures.json")
        print("First failures:")
        for failure in failures[:10]:
            print(
                f"  [{failure['scorer']}] {failure['message']} -> "
                f"{failure['response']!r}"
            )
        if not allow_failures:
            raise SystemExit(1)


def main() -> None:
    args = _parse_args()
    dataset_frame = _load_dataset_frame(args.dataset_file, args.max_examples)
    if dataset_frame.empty:
        raise ValueError("The offline regression dataset is empty.")

    previous_databricks_enabled = os.environ.get("DATABRICKS_LLM_ENABLED")
    os.environ["DATABRICKS_LLM_ENABLED"] = "false"

    try:
        if args.local_only:
            _run_local_only(
                dataset_frame,
                allow_failures=args.allow_failures,
            )
            return

        tracking_uri = _require_env(args.tracking_uri, "MLFLOW_TRACKING_URI")
        experiment_id = _require_env(args.experiment_id, "MLFLOW_EXPERIMENT_ID")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(run_name=args.run_name) as run:
            mlflow.log_param("dataset_file", args.dataset_file)
            mlflow.log_param("num_examples", len(dataset_frame))
            mlflow.log_param("databricks_llm_enabled", "false")
            if args.max_examples is not None:
                mlflow.log_param("max_examples", args.max_examples)

            result = evaluate(
                data=dataset_frame,
                scorers=PROGRAMMATIC_SCORERS,
                predict_fn=_predict_fn,
            )

            print(f"Evaluation run id: {run.info.run_id}")
            print(f"Dataset file: {args.dataset_file}")
            print(f"Examples evaluated: {len(dataset_frame)}")

            metrics = getattr(result, "metrics", None)
            if isinstance(metrics, dict):
                print("Metrics:")
                for key in sorted(metrics):
                    print(f"  {key}: {metrics[key]}")

            tables = getattr(result, "tables", {})
            eval_results = tables.get("eval_results")
            if not isinstance(eval_results, pd.DataFrame):
                raise ValueError("MLflow did not return an eval_results table.")

            failures, failure_counts = _build_failure_report(eval_results)
            if failures:
                artifact_path = Path("tmp/offline_regression_failures.json")
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(json.dumps(failures, indent=2))
                mlflow.log_artifact(str(artifact_path))

            print("Failure counts:")
            for scorer_name in sorted(failure_counts):
                print(f"  {scorer_name}: {failure_counts[scorer_name]}")

            if failures:
                print("First failures:")
                for failure in failures[:10]:
                    print(
                        f"  [{failure['scorer']}] {failure['message']} -> "
                        f"{failure['response']!r}"
                    )

            if failures and not args.allow_failures:
                raise SystemExit(1)
    finally:
        if previous_databricks_enabled is None:
            os.environ.pop("DATABRICKS_LLM_ENABLED", None)
        else:
            os.environ["DATABRICKS_LLM_ENABLED"] = previous_databricks_enabled


if __name__ == "__main__":
    main()
