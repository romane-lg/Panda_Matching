from __future__ import annotations

import json
import os
import re
from typing import Any, cast
from urllib.request import Request, urlopen

import psycopg

PANDAS_URL = "https://blackandwhitebear.com/data/pandas.json"


def fetch_rows() -> list[dict[str, Any]]:
    req = Request(
        PANDAS_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=30) as r:  # noqa: S310
        payload = json.loads(r.read().decode("utf-8"))
        return cast(list[dict[str, Any]], payload)


def build_description(row: dict[str, Any]) -> str | None:
    fact = (row.get("funFact") or "").strip()
    if fact:
        return fact
    return None


def is_twin(row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            (row.get("funFact") or ""),
            (row.get("lifeJourney") or ""),
        ]
    ).lower()
    return "twin" in text


def personality_tags(row: dict[str, Any]) -> str | None:
    text = " ".join(
        [
            (row.get("funFact") or ""),
            (row.get("lifeJourney") or ""),
        ]
    ).lower()
    keyword_map = {
        "playful": "playful",
        "independent": "independent",
        "beloved": "beloved",
        "fan favorite": "fan-favorite",
        "favorite": "fan-favorite",
        "gentle": "gentle",
        "curious": "curious",
        "calm": "calm",
        "energetic": "energetic",
        "miracle": "resilient",
        "independent than": "independent",
    }
    tags: list[str] = []
    for key, tag in keyword_map.items():
        if key in text and tag not in tags:
            tags.append(tag)
    if not tags:
        return None
    return ",".join(tags)


def _extract_notes(row: dict[str, Any], keywords: list[str]) -> str | None:
    chunks = [
        (row.get("funFact") or "").strip(),
        (row.get("lifeJourney") or "").strip(),
    ]
    text = " ".join([c for c in chunks if c]).strip()
    if not text:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            cleaned = sentence.strip()
            if cleaned and cleaned not in selected:
                selected.append(cleaned)
    if not selected:
        return None
    return " ".join(selected)


def breeding_notes(row: dict[str, Any]) -> str | None:
    return _extract_notes(
        row,
        [
            "cubs",
            "cub",
            "mother",
            "father",
            "twin",
            "artificial insemination",
            "born",
            "birth",
            "breeding",
            "gave birth",
        ],
    )


def health_notes(row: dict[str, Any]) -> str | None:
    return _extract_notes(
        row,
        [
            "old age",
            "died",
            "death",
            "clinical",
            "surviving",
            "health",
            "unexpectedly",
            "elderly",
        ],
    )


def main() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    # Accept SQLAlchemy-style DSN from project env.
    db_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    rows = fetch_rows()
    updates = []
    for row in rows:
        source_id = row.get("id")
        if source_id is None:
            continue
        journey = (row.get("lifeJourney") or "").strip() or None
        updates.append(
            (
                build_description(row),
                journey,
                is_twin(row),
                personality_tags(row),
                breeding_notes(row),
                health_notes(row),
                int(source_id),
            )
        )

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE core.panda_profiles
                SET description = %s,
                    life_journey = %s,
                    is_twin = %s,
                    personality_tags = %s,
                    breeding_notes = %s,
                    health_notes = %s
                WHERE source_id = %s
                """,
                updates,
            )
        conn.commit()

    print(f"Updated enrichment columns for {len(updates)} pandas")


if __name__ == "__main__":
    main()
