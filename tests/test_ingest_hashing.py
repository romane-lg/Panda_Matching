from panda_matching.ingest.pipeline import checksum_payload, hash_record


def test_hash_record_stable_ordering() -> None:
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    assert hash_record(first) == hash_record(second)


def test_checksum_payload_changes_with_content() -> None:
    base = [{"id": 1}]
    changed = [{"id": 2}]
    assert checksum_payload(base) != checksum_payload(changed)
