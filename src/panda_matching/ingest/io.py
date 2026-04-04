from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]
        raise ValueError("JSON import file must be an object or list of objects")

    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Each JSONL line must be a JSON object")
            records.append(row)
        return records

    raise ValueError(f"Unsupported import format: {suffix}")
