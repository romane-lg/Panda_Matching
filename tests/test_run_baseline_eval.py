from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_baseline_eval", _MODULE_PATH)
assert _SPEC is not None
run_baseline_eval = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(run_baseline_eval)


class _FakeDataset:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_df(self) -> pd.DataFrame:
        return self._frame.copy()


def test_load_dataset_frame_applies_start_index_and_max_examples(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {"message": "a", "expectations": {}},
            {"message": "b", "expectations": {}},
            {"message": "c", "expectations": {}},
            {"message": "d", "expectations": {}},
        ]
    )
    monkeypatch.setattr(
        run_baseline_eval,
        "_resolve_dataset",
        lambda **_kwargs: _FakeDataset(frame),
    )

    sliced = run_baseline_eval._load_dataset_frame(
        "dataset",
        None,
        "1",
        2,
        start_index=1,
        labeled_only=False,
    )

    assert list(sliced["message"]) == ["b", "c"]


def test_load_dataset_frame_labeled_only_then_start_index(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {"message": "a", "expectations": {}},
            {"message": "b", "expectations": {"expected_facts": ["fact"]}},
            {"message": "c", "expectations": {"expected_response": "ok"}},
        ]
    )
    monkeypatch.setattr(
        run_baseline_eval,
        "_resolve_dataset",
        lambda **_kwargs: _FakeDataset(frame),
    )

    sliced = run_baseline_eval._load_dataset_frame(
        "dataset",
        None,
        "1",
        None,
        start_index=1,
        labeled_only=True,
    )

    assert list(sliced["message"]) == ["c"]


def test_load_dataset_frame_rejects_negative_start_index(monkeypatch) -> None:
    frame = pd.DataFrame([{"message": "a", "expectations": {}}])
    monkeypatch.setattr(
        run_baseline_eval,
        "_resolve_dataset",
        lambda **_kwargs: _FakeDataset(frame),
    )

    with pytest.raises(ValueError, match="--start-index must be zero or greater"):
        run_baseline_eval._load_dataset_frame(
            "dataset",
            None,
            "1",
            None,
            start_index=-1,
            labeled_only=False,
        )
