from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult

from panda_matching.db.models import ImportedRecord, ImportRun
from panda_matching.db.session import get_sessionmaker


def checksum_payload(records: list[dict[str, Any]]) -> str:
    normalized = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_record(record: dict[str, Any]) -> str:
    normalized = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_import(source: str, records: list[dict[str, Any]]) -> uuid.UUID:
    session_maker = get_sessionmaker()
    run_id = uuid.uuid4()
    checksum = checksum_payload(records)

    with session_maker() as session:
        run = ImportRun(
            id=run_id,
            source=source,
            checksum=checksum,
            status="running",
            rows_loaded=0,
        )
        session.add(run)
        session.commit()

        try:
            payload_rows = [
                {
                    "source": source,
                    "record_hash": hash_record(record),
                    "payload": record,
                }
                for record in records
            ]

            inserted_count = 0
            if payload_rows:
                stmt = insert(ImportedRecord).values(payload_rows)
                stmt = stmt.on_conflict_do_nothing(index_elements=["record_hash"])
                result = session.execute(stmt)
                if isinstance(result, CursorResult):
                    inserted_count = int(result.rowcount or 0)

            run.status = "success"
            run.rows_loaded = inserted_count
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
            return run_id
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
            raise
