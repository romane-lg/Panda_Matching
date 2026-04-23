from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session


def rows(session: Session, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    result = session.execute(text(sql), params)
    return [dict(row) for row in result.mappings().all()]


def scalar_int(session: Session, sql: str, params: dict[str, Any] | None = None) -> int:
    row = session.execute(text(sql), params or {}).mappings().one()
    return int(row["n"])


def relation_exists(session: Session, schema: str, relation: str) -> bool:
    row = session.execute(
        text("SELECT to_regclass(:fq_name) IS NOT NULL AS exists_flag"),
        {"fq_name": f"{schema}.{relation}"},
    ).mappings().one()
    return bool(row["exists_flag"])


def relation_columns(session: Session, schema: str, relation: str) -> set[str]:
    result = rows(
        session,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        """,
        {"schema_name": schema, "table_name": relation},
    )
    return {str(row["column_name"]) for row in result}


def pick_relation(session: Session, schema: str, candidates: list[str]) -> str:
    for relation in candidates:
        if relation_exists(session, schema=schema, relation=relation):
            return relation
    raise HTTPException(
        status_code=500,
        detail=f"No expected relation found in {schema}: {candidates}",
    )


def top_matches_data(session: Session, panda_name: str, k: int) -> dict[str, Any]:
    ranked_view = pick_relation(
        session,
        schema="core",
        candidates=[
            "ranked_directional_recommended_matches_v2",
            "ranked_directional_recommended_matches",
        ],
    )
    columns = relation_columns(session, schema="core", relation=ranked_view)
    focal_col = "focal_panda_name" if "focal_panda_name" in columns else "name"
    rank_col = None
    if "recommendation_rank_v2" in columns:
        rank_col = "recommendation_rank_v2"
    elif "recommendation_rank" in columns:
        rank_col = "recommendation_rank"

    if rank_col is None:
        raise HTTPException(
            status_code=500,
            detail=f"{ranked_view} is missing a recommendation rank column",
        )

    sql = f"""
        SELECT *
        FROM core.{ranked_view}
        WHERE lower({focal_col}) = lower(:panda_name)
        ORDER BY {rank_col}
        LIMIT :k
    """
    matches = rows(session, sql, {"panda_name": panda_name, "k": k})
    return {
        "panda_name": panda_name,
        "count": len(matches),
        "matches": matches,
        "source_view": ranked_view,
    }


def explain_match_data(session: Session, focal_id: str, candidate_id: str) -> dict[str, Any]:
    relation = pick_relation(
        session,
        schema="core",
        candidates=[
            "directional_recommended_matches",
            "ranked_directional_recommended_matches",
        ],
    )
    cols = relation_columns(session, schema="core", relation=relation)

    if "focal_panda_id" in cols and "candidate_panda_id" in cols:
        where_sql = "focal_panda_id = :focal_id AND candidate_panda_id = :candidate_id"
    elif "panda_1_id" in cols and "panda_2_id" in cols:
        where_sql = "(panda_1_id = :focal_id AND panda_2_id = :candidate_id)"
    else:
        raise HTTPException(
            status_code=500,
            detail=f"{relation} is missing expected ID columns for explanation",
        )

    sql = f"SELECT * FROM core.{relation} WHERE {where_sql} LIMIT 1"
    result = rows(session, sql, {"focal_id": focal_id, "candidate_id": candidate_id})
    if not result:
        raise HTTPException(
            status_code=404,
            detail=(
                "No directional match found for "
                f"focal_id={focal_id} candidate_id={candidate_id}"
            ),
        )
    return {"focal_id": focal_id, "candidate_id": candidate_id, "explanation": result[0]}


def blockers_data(session: Session, focal_id: str) -> dict[str, Any]:
    pair_view = pick_relation(session, schema="core", candidates=["candidate_pairs"])
    cols = relation_columns(session, schema="core", relation=pair_view)

    if not {"panda_1_id", "panda_2_id"}.issubset(cols):
        raise HTTPException(
            status_code=500,
            detail="candidate_pairs is missing panda_1_id/panda_2_id",
        )

    reason_col = "pair_risk_reason" if "pair_risk_reason" in cols else None
    elig_col = "pair_eligibility" if "pair_eligibility" in cols else None

    if reason_col:
        blocker_filter = f"COALESCE(NULLIF({reason_col}, ''), 'ok') <> 'ok'"
    elif elig_col:
        blocker_filter = f"COALESCE({elig_col}, '') NOT IN ('eligible', 'true', 'TRUE')"
    else:
        raise HTTPException(
            status_code=500,
            detail="candidate_pairs missing pair_risk_reason and pair_eligibility",
        )

    sql = f"""
        SELECT *
        FROM core.{pair_view}
        WHERE (panda_1_id = :focal_id OR panda_2_id = :focal_id)
          AND {blocker_filter}
        ORDER BY panda_1_id, panda_2_id
    """
    blocker_rows = rows(session, sql, {"focal_id": focal_id})
    return {"focal_id": focal_id, "count": len(blocker_rows), "blockers": blocker_rows}


def best_overall_match_data(session: Session, k: int = 1) -> dict[str, Any]:
    ranked_view = pick_relation(
        session,
        schema="core",
        candidates=[
            "ranked_directional_recommended_matches_v2",
            "ranked_directional_recommended_matches",
        ],
    )
    cols = relation_columns(session, schema="core", relation=ranked_view)

    if "final_score_v2" in cols:
        order_col = "final_score_v2"
    elif "recommendation_score" in cols:
        order_col = "recommendation_score"
    else:
        raise HTTPException(
            status_code=500,
            detail=f"{ranked_view} has no supported score column for global ranking",
        )

    sql = f"""
        SELECT *
        FROM core.{ranked_view}
        ORDER BY {order_col} DESC
        LIMIT :k
    """
    match_rows = rows(session, sql, {"k": k})
    return {
        "count": len(match_rows),
        "matches": match_rows,
        "source_view": ranked_view,
        "score_column": order_col,
    }


def find_panda_id_by_name(session: Session, panda_name: str) -> str | None:
    result = rows(
        session,
        """
        SELECT source_id
        FROM core.panda_profiles
        WHERE lower(trim(name)) = lower(trim(:panda_name))
        ORDER BY ingested_at DESC NULLS LAST, source_id
        LIMIT 1
        """,
        {"panda_name": panda_name},
    )
    if not result:
        return None
    return str(result[0]["source_id"])


def find_panda_name_by_substring(session: Session, name_hint: str) -> str | None:
    result = rows(
        session,
        """
        SELECT name
        FROM core.panda_profiles
        WHERE lower(name) LIKE lower(:name_pattern)
        ORDER BY char_length(name), name
        LIMIT 1
        """,
        {"name_pattern": f"%{name_hint.strip()}%"},
    )
    if not result:
        return None
    return str(result[0]["name"])


def panda_profile_data(session: Session, panda_name: str) -> dict[str, Any] | None:
    relation = (
        "panda_profiles_agent"
        if relation_exists(session, "core", "panda_profiles_agent")
        else "panda_profiles"
    )
    cols = relation_columns(session, "core", relation)
    personality_col = (
        "personality_tags_agent" if "personality_tags_agent" in cols else "personality_tags"
    )
    health_col = "health_notes_agent" if "health_notes_agent" in cols else "health_notes"
    desc_col = "description_agent" if "description_agent" in cols else "description"

    sql = f"""
        SELECT
            source_id, name, chinese_name, sex, age_years, birth_date,
            zoo_or_facility, city_region, country, status, babies_had_count,
            {personality_col} AS personality_text,
            {health_col} AS health_text,
            {desc_col} AS description_text
        FROM core.{relation}
        WHERE lower(trim(name)) = lower(trim(:panda_name))
        ORDER BY source_id
        LIMIT 1
    """
    result = rows(session, sql, {"panda_name": panda_name})
    if not result:
        return None
    profile = result[0]
    profile["source_relation"] = relation
    return profile


def ensure_chat_session(session: Session, session_id: str) -> None:
    session.execute(
        text(
            """
            INSERT INTO core.chat_sessions (session_id)
            VALUES (:session_id)
            ON CONFLICT (session_id)
            DO UPDATE SET updated_at = NOW()
            """
        ),
        {"session_id": session_id},
    )


def load_chat_state(session: Session, session_id: str) -> dict[str, str]:
    result = rows(
        session,
        """
        SELECT state_json
        FROM core.chat_state
        WHERE session_id = :session_id
        LIMIT 1
        """,
        {"session_id": session_id},
    )
    if not result:
        return {}
    state_json = result[0].get("state_json")
    if not isinstance(state_json, dict):
        return {}

    memory: dict[str, str] = {}
    for key, value in state_json.items():
        if isinstance(value, str):
            memory[str(key)] = value
    return memory


def save_chat_state(session: Session, session_id: str, memory: dict[str, str]) -> None:
    session.execute(
        text(
            """
            INSERT INTO core.chat_state (session_id, state_json, updated_at)
            VALUES (:session_id, CAST(:state_json AS jsonb), NOW())
            ON CONFLICT (session_id)
            DO UPDATE SET
                state_json = EXCLUDED.state_json,
                updated_at = NOW()
            """
        ),
        {"session_id": session_id, "state_json": json.dumps(memory)},
    )
    session.execute(
        text(
            """
            UPDATE core.chat_sessions
            SET updated_at = NOW()
            WHERE session_id = :session_id
            """
        ),
        {"session_id": session_id},
    )


def append_chat_message(
    session: Session,
    *,
    session_id: str,
    role: str,
    message: str,
    intent: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    session.execute(
        text(
            """
            INSERT INTO core.chat_messages (session_id, role, message, intent, data)
            VALUES (:session_id, :role, :message, :intent, CAST(:data AS jsonb))
            """
        ),
        {
            "session_id": session_id,
            "role": role,
            "message": message,
            "intent": intent,
            "data": None if data is None else json.dumps(data, default=str),
        },
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def looks_like_id(text_value: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9-]+$", text_value))


def count_for_relation(
    session: Session,
    relation: str,
    *,
    panda_id: str | None = None,
    panda_name: str | None = None,
) -> int:
    cols = relation_columns(session, schema="core", relation=relation)
    where_sql: str | None = None
    params: dict[str, Any] = {}

    if panda_id and "focal_panda_id" in cols:
        where_sql = "focal_panda_id = :panda_id"
        params = {"panda_id": panda_id}
    elif panda_id and {"panda_1_id", "panda_2_id"}.issubset(cols):
        where_sql = "(panda_1_id = :panda_id OR panda_2_id = :panda_id)"
        params = {"panda_id": panda_id}
    elif panda_name and "focal_panda_name" in cols:
        where_sql = "lower(trim(focal_panda_name)) = lower(trim(:panda_name))"
        params = {"panda_name": panda_name}
    elif panda_name and "name" in cols:
        where_sql = "lower(trim(name)) = lower(trim(:panda_name))"
        params = {"panda_name": panda_name}
    elif panda_name and "panda_name" in cols:
        where_sql = "lower(trim(panda_name)) = lower(trim(:panda_name))"
        params = {"panda_name": panda_name}

    if where_sql is None:
        return 0

    row = session.execute(
        text(f"SELECT COUNT(*) AS n FROM core.{relation} WHERE {where_sql}"),
        params,
    ).mappings().one()
    return int(row["n"])


def diagnose_no_matches(session: Session, panda_name: str) -> dict[str, Any]:
    profiles = rows(
        session,
        """
        SELECT source_id, name, sex, age_years, babies_had_count, status, on_loan, ingested_at
        FROM core.panda_profiles
        WHERE lower(trim(name)) = lower(trim(:panda_name))
        ORDER BY ingested_at DESC NULLS LAST, source_id
        """,
        {"panda_name": panda_name},
    )
    if not profiles:
        return {
            "summary": f"I couldn't find a panda named '{panda_name}' in core.panda_profiles.",
            "waterfall": {},
            "profile": None,
        }

    profile = profiles[0]
    focal_id = str(profile["source_id"])

    eligible_relation = None
    for candidate in ["breedeable_pandas", "breedable_pandas"]:
        if relation_exists(session, schema="core", relation=candidate):
            eligible_relation = candidate
            break

    eligible_count = 0
    if eligible_relation is not None:
        eligible_count = count_for_relation(
            session,
            eligible_relation,
            panda_id=focal_id,
            panda_name=panda_name,
        )

    candidate_count = 0
    if relation_exists(session, schema="core", relation="candidate_pairs"):
        candidate_count = count_for_relation(session, "candidate_pairs", panda_id=focal_id)

    directional_count = 0
    if relation_exists(session, schema="core", relation="directional_recommended_matches"):
        directional_count = count_for_relation(
            session,
            "directional_recommended_matches",
            panda_id=focal_id,
            panda_name=panda_name,
        )

    ranked_count = 0
    if relation_exists(session, schema="core", relation="ranked_directional_recommended_matches"):
        ranked_count = count_for_relation(
            session,
            "ranked_directional_recommended_matches",
            panda_id=focal_id,
            panda_name=panda_name,
        )

    reasons: list[str] = []
    age_years = profile.get("age_years")
    status = (profile.get("status") or "").lower()

    if status and status != "alive":
        reasons.append(f"status is '{profile.get('status')}', not alive")
    if isinstance(age_years, int) and age_years < 5:
        reasons.append(f"age is {age_years}, below breeding threshold")
    if eligible_relation and eligible_count == 0:
        reasons.append(f"not present in core.{eligible_relation}")
    if candidate_count == 0:
        reasons.append("no candidate pairs were generated for this focal panda")

    if reasons:
        summary = (
            f"{profile.get('name')} has no ranked matches because " + "; ".join(reasons) + "."
        )
    else:
        summary = (
            f"{profile.get('name')} has no ranked matches. "
            "This likely drops out between candidate/score/rank stages."
        )

    return {
        "summary": summary,
        "waterfall": {
            "profile_rows": len(profiles),
            "eligible_rows": eligible_count,
            "candidate_pairs_rows": candidate_count,
            "directional_rows": directional_count,
            "ranked_rows": ranked_count,
        },
        "profile": profile,
    }


def curated_override_for_name(session: Session, panda_name: str) -> dict[str, Any] | None:
    if not relation_exists(session, schema="core", relation="panda_profile_overrides"):
        return None
    result = rows(
        session,
        """
        SELECT
            canonical_name, aliases, personality_tags, health_notes,
            agent_summary, source_label, updated_at
        FROM core.panda_profile_overrides
        WHERE lower(trim(canonical_name)) = lower(trim(:panda_name))
        LIMIT 1
        """,
        {"panda_name": panda_name},
    )
    if result:
        return result[0]

    result = rows(
        session,
        """
        SELECT
            canonical_name, aliases, personality_tags, health_notes,
            agent_summary, source_label, updated_at
        FROM core.panda_profile_overrides
        WHERE EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(aliases) AS alias_name
            WHERE lower(trim(alias_name)) = lower(trim(:panda_name))
        )
        LIMIT 1
        """,
        {"panda_name": panda_name},
    )
    return result[0] if result else None
