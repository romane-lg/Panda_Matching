from pathlib import Path

from panda_matching.ingest.io import load_records


def test_load_json_array(tmp_path: Path) -> None:
    payload = '[{"id":1},{"id":2}]'
    file = tmp_path / "data.json"
    file.write_text(payload, encoding="utf-8")
    assert len(load_records(file)) == 2


def test_load_jsonl(tmp_path: Path) -> None:
    payload = '{"id":1}\n{"id":2}\n'
    file = tmp_path / "data.jsonl"
    file.write_text(payload, encoding="utf-8")
    assert len(load_records(file)) == 2
