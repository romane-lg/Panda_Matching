from __future__ import annotations

import json
import re
import subprocess
from datetime import date, datetime
from typing import Any, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from panda_matching.db.models import PandaProfile
from panda_matching.db.session import get_sessionmaker

PANDAS_URL = "https://blackandwhitebear.com/data/pandas.json"


def _base_name(name: str | None) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", name or "").strip()


def _normalized_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _natural_key_from_values(
    *,
    name: str | None,
    birth_date: str | None,
    birth_year: int | None,
    sex: str | None,
) -> tuple[str, str, int, str]:
    return (
        _normalized_text(_base_name(name)),
        (birth_date or "").strip(),
        int(birth_year or 0),
        _normalized_text(sex),
    )


def _natural_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return _natural_key_from_values(
        name=row.get("name"),
        birth_date=row.get("birth_date"),
        birth_year=row.get("birth_year"),
        sex=row.get("sex"),
    )


def _quality_score(row: dict[str, Any]) -> int:
    fields = [
        "name",
        "chinese_name",
        "sex",
        "birth_date",
        "birth_year",
        "current_location",
        "zoo_or_facility",
        "city_region",
        "country",
        "mother",
        "father",
        "status",
    ]
    score = sum(1 for field in fields if row.get(field))
    payload = row.get("raw_payload")
    if isinstance(payload, dict):
        score += len(payload)
    return score


def _country_from_location(location: str | None) -> str | None:
    if not location:
        return None
    return location.split(",")[-1].strip() or None


def _split_location(location: str | None) -> tuple[str | None, str | None, str | None]:
    if not location:
        return (None, None, None)
    parts = [x.strip() for x in location.split(",") if x.strip()]
    if not parts:
        return (None, None, None)

    facility = parts[0]
    country = parts[-1]
    city_region = (
        ", ".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) == 2 else None)
    )
    return (facility, city_region, country)


def _age_years(
    birth_date: str | None,
    birth_year: int | None,
    death_date: str | None,
) -> int | None:
    try:
        if birth_date:
            born = datetime.strptime(birth_date, "%Y-%m-%d").date()
            end = datetime.strptime(death_date, "%Y-%m-%d").date() if death_date else date.today()
            return end.year - born.year - ((end.month, end.day) < (born.month, born.day))
        if birth_year:
            end_year = int(death_date[:4]) if death_date else date.today().year
            return end_year - int(birth_year)
    except Exception:
        return None
    return None


def _ownership(name: str, country: str | None) -> tuple[bool | None, str]:
    if name == "Xin Xin" and country == "Mexico":
        return (False, "Mexico-Owned")
    if country in {"China", "Hong Kong", "Macau"}:
        return (False, "China-Owned (Domestic)")
    if country:
        return (True, "China-Owned (On Loan)")
    return (None, "Unknown")


def _count_offspring(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for key in ("mother", "father"):
            parent = (row.get(key) or "").strip()
            if not parent or parent.lower() == "unknown":
                continue
            normalized = _base_name(parent)
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def fetch_pandas() -> list[dict[str, Any]]:
    request = Request(
        PANDAS_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
            return cast(list[dict[str, Any]], payload)
    except HTTPError as exc:
        if exc.code != 403:
            raise
        # Fallback for hosts that block urllib-based clients.
        raw = subprocess.check_output(  # noqa: S603
            [
                "curl",
                "-sL",
                "-A",
                (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                PANDAS_URL,
            ],
            text=True,
        )
        payload = json.loads(raw)
        return cast(list[dict[str, Any]], payload)


def load_profiles() -> int:
    raw_rows = fetch_pandas()
    offspring_counts = _count_offspring(raw_rows)

    upsert_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        name = _base_name(raw.get("name"))
        location = raw.get("currentLocation")
        facility, city_region, country = _split_location(location)
        on_loan, ownership_category = _ownership(name, _country_from_location(location))

        upsert_rows.append(
            {
                "source_id": str(raw["id"]),
                "name": name,
                "chinese_name": raw.get("chineseName"),
                "sex": raw.get("sex"),
                "age_years": _age_years(
                    raw.get("birthDate"),
                    raw.get("birthYear"),
                    raw.get("deathDate"),
                ),
                "birth_date": raw.get("birthDate"),
                "birth_year": raw.get("birthYear"),
                "current_location": location,
                "zoo_or_facility": facility,
                "city_region": city_region,
                "country": country,
                "mother": raw.get("mother"),
                "father": raw.get("father"),
                "babies_had_count": offspring_counts.get(name, 0),
                "on_loan": on_loan,
                "ownership_category": ownership_category,
                "status": raw.get("status"),
                "raw_payload": raw,
            }
        )

    session_maker = get_sessionmaker()
    with session_maker() as session:
        existing_key_to_source_id: dict[tuple[str, str, int, str], str] = {}
        existing_rows = session.execute(
            select(
                PandaProfile.source_id,
                PandaProfile.name,
                PandaProfile.birth_date,
                PandaProfile.birth_year,
                PandaProfile.sex,
            )
        )
        for source_id, name, birth_date, birth_year, sex in existing_rows:
            key = _natural_key_from_values(
                name=name,
                birth_date=birth_date.isoformat() if birth_date else None,
                birth_year=birth_year,
                sex=sex,
            )
            existing_key_to_source_id.setdefault(key, source_id)

        deduped_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        for row in upsert_rows:
            key = _natural_key(row)
            canonical_source_id = existing_key_to_source_id.get(key)
            if canonical_source_id:
                row["source_id"] = canonical_source_id

            prior = deduped_by_key.get(key)
            if prior is None or _quality_score(row) > _quality_score(prior):
                deduped_by_key[key] = row

        deduped_by_source_id: dict[str, dict[str, Any]] = {}
        for row in deduped_by_key.values():
            source_id = str(row["source_id"])
            prior = deduped_by_source_id.get(source_id)
            if prior is None or _quality_score(row) > _quality_score(prior):
                deduped_by_source_id[source_id] = row

        final_rows = list(deduped_by_source_id.values())
        if not final_rows:
            return 0

        stmt = insert(PandaProfile).values(final_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["source_id"],
            set_={
                "name": stmt.excluded.name,
                "chinese_name": stmt.excluded.chinese_name,
                "sex": stmt.excluded.sex,
                "age_years": stmt.excluded.age_years,
                "birth_date": stmt.excluded.birth_date,
                "birth_year": stmt.excluded.birth_year,
                "current_location": stmt.excluded.current_location,
                "zoo_or_facility": stmt.excluded.zoo_or_facility,
                "city_region": stmt.excluded.city_region,
                "country": stmt.excluded.country,
                "mother": stmt.excluded.mother,
                "father": stmt.excluded.father,
                "babies_had_count": stmt.excluded.babies_had_count,
                "on_loan": stmt.excluded.on_loan,
                "ownership_category": stmt.excluded.ownership_category,
                "status": stmt.excluded.status,
                "raw_payload": stmt.excluded.raw_payload,
                "ingested_at": datetime.now(),
            },
        )
        session.execute(stmt)
        session.commit()

    return len(final_rows)
