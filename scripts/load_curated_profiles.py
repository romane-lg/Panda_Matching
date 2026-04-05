from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import psycopg

DEFAULT_CURATED_JSON = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "curated"
    / "famous_pandas_profiles_2026_04_05.json"
)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _resolve_source_id(conn: psycopg.Connection[Any], aliases: list[str]) -> str | None:
    normalized_aliases = [_normalize_name(alias) for alias in aliases if alias.strip()]
    if not normalized_aliases:
        return None

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT source_id
            FROM core.panda_profiles
            WHERE lower(trim(name)) = ANY(%s)
            ORDER BY ingested_at DESC NULLS LAST, source_id
            LIMIT 1
            """,
            (normalized_aliases,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return str(row["source_id"])


def load_curated_profiles(data_path: Path) -> int:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Curated profiles JSON must be a list")

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    db_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    upserted = 0
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            for item in payload:
                if not isinstance(item, dict):
                    continue
                canonical_name = str(item.get("canonical_name") or "").strip()
                aliases = item.get("aliases") or []
                if not canonical_name:
                    continue
                if not isinstance(aliases, list):
                    aliases = [canonical_name]
                aliases = [str(x).strip() for x in aliases if str(x).strip()]
                if canonical_name not in aliases:
                    aliases.insert(0, canonical_name)

                source_id = _resolve_source_id(conn, aliases)
                personality_tags = item.get("personality_tags") or []
                personality_tags_text: str | None
                if isinstance(personality_tags, list):
                    personality_tags_text = ",".join(
                        str(t).strip() for t in personality_tags if str(t).strip()
                    )
                else:
                    personality_tags_text = str(personality_tags).strip() or None
                health_notes = (str(item.get("health_notes") or "").strip() or None)
                agent_summary = (str(item.get("agent_summary") or "").strip() or None)
                source_label = str(item.get("source_label") or "curated profile").strip()

                cur.execute(
                    """
                    INSERT INTO core.panda_profile_overrides (
                        canonical_name,
                        aliases,
                        source_id,
                        personality_tags,
                        health_notes,
                        agent_summary,
                        source_label
                    )
                    VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s)
                    ON CONFLICT (canonical_name)
                    DO UPDATE SET
                        aliases = EXCLUDED.aliases,
                        source_id = EXCLUDED.source_id,
                        personality_tags = EXCLUDED.personality_tags,
                        health_notes = EXCLUDED.health_notes,
                        agent_summary = EXCLUDED.agent_summary,
                        source_label = EXCLUDED.source_label,
                        updated_at = NOW()
                    """,
                    (
                        canonical_name,
                        json.dumps(aliases),
                        source_id,
                        personality_tags_text,
                        health_notes,
                        agent_summary,
                        source_label,
                    ),
                )
                upserted += 1
        conn.commit()
    return upserted


def main() -> None:
    configured = os.getenv("CURATED_PANDAS_JSON")
    data_path = Path(configured).expanduser().resolve() if configured else DEFAULT_CURATED_JSON
    if not data_path.exists():
        raise FileNotFoundError(f"Curated JSON not found: {data_path}")
    loaded = load_curated_profiles(data_path)
    print(f"Upserted {loaded} curated panda override profiles from {data_path}")


if __name__ == "__main__":
    main()
