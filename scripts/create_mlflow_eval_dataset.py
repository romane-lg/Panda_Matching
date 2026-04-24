from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_CASES_PATH = Path("data/evaluation/panda_chatbot_eval_v1.json")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of cases in {path}")
    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Case {idx} is not a JSON object")
        if "inputs" not in item or not isinstance(item["inputs"], dict):
            raise ValueError(f"Case {idx} must contain an object field named 'inputs'")
        expectations = item.get("expectations", {})
        if expectations is not None and not isinstance(expectations, dict):
            raise ValueError(f"Case {idx} expectations must be an object if provided")
        cases.append(
            {
                "inputs": item["inputs"],
                "expectations": expectations,
            }
        )
    return cases


def _print_summary(cases: list[dict[str, Any]]) -> None:
    counts = Counter(
        str(case.get("expectations", {}).get("task_type", "unknown"))
        for case in cases
    )
    print(f"Loaded {len(cases)} evaluation cases")
    for task_type, count in sorted(counts.items()):
        print(f"  {task_type}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create or update an MLflow GenAI evaluation dataset "
            "from starter chatbot cases."
        )
    )
    parser.add_argument(
        "--cases-file",
        default=str(DEFAULT_CASES_PATH),
        help="Path to the JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--name",
        default="panda_chatbot_eval_v1",
        help="MLflow dataset name to create or update.",
    )
    parser.add_argument(
        "--experiment-id",
        default=os.getenv("MLFLOW_EXPERIMENT_ID"),
        help="MLflow experiment ID. Defaults to $MLFLOW_EXPERIMENT_ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the dataset without calling MLflow.",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases_file)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    cases = _load_cases(cases_path)
    _print_summary(cases)

    if args.dry_run:
        print("Dry run only. No dataset was created in MLflow.")
        return

    if not args.experiment_id:
        raise ValueError(
            "Missing experiment ID. Pass --experiment-id or set MLFLOW_EXPERIMENT_ID."
        )

    import mlflow
    from mlflow.genai.datasets import create_dataset, get_dataset, search_datasets

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    existing = [
        dataset
        for dataset in search_datasets(experiment_ids=args.experiment_id)
        if dataset.name == args.name
    ]
    if existing:
        existing.sort(
            key=lambda dataset: (
                dataset.created_time or dataset.create_time or 0,
                dataset.dataset_id,
            ),
            reverse=True,
        )
        dataset = get_dataset(dataset_id=existing[0].dataset_id)
        print(f"Updating existing dataset '{args.name}' via dataset_id={dataset.dataset_id}.")
    else:
        dataset = create_dataset(
            name=args.name,
            experiment_id=args.experiment_id,
            tags={
                "app": "panda-matching",
                "dataset_type": "chatbot_eval",
                "version": "v1",
            },
        )
    dataset.merge_records(cases)
    print(f"Dataset '{args.name}' updated successfully.")
    print(f"Experiment ID: {args.experiment_id}")


if __name__ == "__main__":
    main()
