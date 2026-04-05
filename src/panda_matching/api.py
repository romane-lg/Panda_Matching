from __future__ import annotations

import re
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from panda_matching.db.session import get_sessionmaker

app = FastAPI(title="Panda Matching API", version="0.1.0")
CHAT_MEMORY: dict[str, dict[str, str]] = {}


def _rows(session: Session, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    result = session.execute(text(sql), params)
    return [dict(row) for row in result.mappings().all()]


def _scalar_int(session: Session, sql: str, params: dict[str, Any] | None = None) -> int:
    row = session.execute(text(sql), params or {}).mappings().one()
    return int(row["n"])


def _relation_exists(session: Session, schema: str, relation: str) -> bool:
    row = session.execute(
        text("SELECT to_regclass(:fq_name) IS NOT NULL AS exists_flag"),
        {"fq_name": f"{schema}.{relation}"},
    ).mappings().one()
    return bool(row["exists_flag"])


def _relation_columns(session: Session, schema: str, relation: str) -> set[str]:
    rows = _rows(
        session,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        """,
        {"schema_name": schema, "table_name": relation},
    )
    return {str(row["column_name"]) for row in rows}


def _pick_relation(session: Session, schema: str, candidates: list[str]) -> str:
    for relation in candidates:
        if _relation_exists(session, schema=schema, relation=relation):
            return relation
    raise HTTPException(
        status_code=500,
        detail=f"No expected relation found in {schema}: {candidates}",
    )


def _top_matches_data(session: Session, panda_name: str, k: int) -> dict[str, Any]:
    ranked_view = _pick_relation(
        session,
        schema="core",
        candidates=[
            "ranked_directional_recommended_matches_v2",
            "ranked_directional_recommended_matches",
        ],
    )
    columns = _relation_columns(session, schema="core", relation=ranked_view)
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
    data = _rows(session, sql, {"panda_name": panda_name, "k": k})
    return {
        "panda_name": panda_name,
        "count": len(data),
        "matches": data,
        "source_view": ranked_view,
    }


def _explain_match_data(session: Session, focal_id: str, candidate_id: str) -> dict[str, Any]:
    rel = _pick_relation(
        session,
        schema="core",
        candidates=[
            "directional_recommended_matches",
            "ranked_directional_recommended_matches",
        ],
    )
    cols = _relation_columns(session, schema="core", relation=rel)

    if "focal_panda_id" in cols and "candidate_panda_id" in cols:
        where_sql = "focal_panda_id = :focal_id AND candidate_panda_id = :candidate_id"
    elif "panda_1_id" in cols and "panda_2_id" in cols:
        where_sql = "(panda_1_id = :focal_id AND panda_2_id = :candidate_id)"
    else:
        raise HTTPException(
            status_code=500,
            detail=f"{rel} is missing expected ID columns for explanation",
        )

    sql = f"SELECT * FROM core.{rel} WHERE {where_sql} LIMIT 1"
    rows = _rows(session, sql, {"focal_id": focal_id, "candidate_id": candidate_id})
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                "No directional match found for "
                f"focal_id={focal_id} candidate_id={candidate_id}"
            ),
        )
    return {"focal_id": focal_id, "candidate_id": candidate_id, "explanation": rows[0]}


def _blockers_data(session: Session, focal_id: str) -> dict[str, Any]:
    pair_view = _pick_relation(session, schema="core", candidates=["candidate_pairs"])
    cols = _relation_columns(session, schema="core", relation=pair_view)

    required = {"panda_1_id", "panda_2_id"}
    if not required.issubset(cols):
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
    blockers = _rows(session, sql, {"focal_id": focal_id})
    return {"focal_id": focal_id, "count": len(blockers), "blockers": blockers}


def _best_overall_match_data(session: Session, k: int = 1) -> dict[str, Any]:
    ranked_view = _pick_relation(
        session,
        schema="core",
        candidates=[
            "ranked_directional_recommended_matches_v2",
            "ranked_directional_recommended_matches",
        ],
    )
    cols = _relation_columns(session, schema="core", relation=ranked_view)

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
    rows = _rows(session, sql, {"k": k})
    return {
        "count": len(rows),
        "matches": rows,
        "source_view": ranked_view,
        "score_column": order_col,
    }


def _find_panda_id_by_name(session: Session, panda_name: str) -> str | None:
    rows = _rows(
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
    if not rows:
        return None
    return str(rows[0]["source_id"])


def _find_panda_name_by_substring(session: Session, name_hint: str) -> str | None:
    rows = _rows(
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
    if not rows:
        return None
    return str(rows[0]["name"])


def _panda_profile_data(session: Session, panda_name: str) -> dict[str, Any] | None:
    relation = (
        "panda_profiles_agent"
        if _relation_exists(session, "core", "panda_profiles_agent")
        else "panda_profiles"
    )
    cols = _relation_columns(session, "core", relation)
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
    rows = _rows(session, sql, {"panda_name": panda_name})
    if not rows:
        return None
    row = rows[0]
    row["source_relation"] = relation
    return row


def _extract_name_from_question(message: str) -> str | None:
    patterns = [
        r"^(?:who is|tell me(?:\s+more)? about|profile of)\s+(.+)$",
        r"^(?:how old is|where is|health of|personality of)\s+(.+)$",
        r"^(?:what is the health of|what is the personality of)\s+(.+)$",
    ]
    for pattern in patterns:
        matched = re.match(pattern, message.strip(), flags=re.IGNORECASE)
        if matched:
            return matched.group(1).strip(" ?.")
    return None


def _analytics_chat_response(
    session: Session,
    session_id: str,
    message: str,
    memory: dict[str, str],
) -> ChatResponse | None:
    lower = message.strip().lower()

    eligible_pat = re.match(r"^(?:how many|count)\s+eligible\s+pandas\??$", lower)
    if eligible_pat:
        relation = _pick_relation(session, "core", ["breedeable_pandas", "breedable_pandas"])
        total = _scalar_int(session, f"SELECT COUNT(*) AS n FROM core.{relation}")
        return ChatResponse(
            session_id=session_id,
            intent="analytics_count_eligible",
            response=f"There are {total} eligible pandas in core.{relation}.",
            data={"relation": relation, "count": total},
        )

    all_pat = re.match(r"^(?:how many|count)\s+pandas\??$", lower)
    if all_pat:
        total = _scalar_int(session, "SELECT COUNT(*) AS n FROM core.panda_profiles")
        return ChatResponse(
            session_id=session_id,
            intent="analytics_count_pandas",
            response=f"There are {total} pandas in core.panda_profiles.",
            data={"relation": "core.panda_profiles", "count": total},
        )

    status_pat = re.match(r"^(?:how many|count)\s+(alive|deceased|dead)\s+pandas\??$", lower)
    if status_pat:
        status = status_pat.group(1)
        normalized = "deceased" if status == "dead" else status
        total = _scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(status, '')) = :status
            """,
            {"status": normalized},
        )
        return ChatResponse(
            session_id=session_id,
            intent="analytics_count_status",
            response=f"There are {total} pandas with status '{normalized}'.",
            data={"status": normalized, "count": total},
        )

    curated_pat = re.match(r"^(?:how many|count)\s+curated\s+(?:profiles|pandas)\??$", lower)
    if curated_pat:
        relation_exists = _relation_exists(session, "core", "panda_profile_overrides")
        if not relation_exists:
            return ChatResponse(
                session_id=session_id,
                intent="analytics_curated_missing",
                response="Curated override table is not available yet in this database.",
                data={"relation": "core.panda_profile_overrides", "count": 0},
            )
        total = _scalar_int(session, "SELECT COUNT(*) AS n FROM core.panda_profile_overrides")
        return ChatResponse(
            session_id=session_id,
            intent="analytics_count_curated",
            response=f"There are {total} curated panda override profiles.",
            data={"relation": "core.panda_profile_overrides", "count": total},
        )

    matches_pat = re.match(
        r"^(?:how many|count)\s+matches(?:\s+for)?\s+(.+)$",
        message,
        re.IGNORECASE,
    )
    if matches_pat:
        panda_name = matches_pat.group(1).strip(" ?.")
        resolved_name = _find_panda_name_by_substring(session, panda_name) or panda_name
        top = _top_matches_data(session, panda_name=resolved_name, k=50)
        memory["last_panda_name"] = resolved_name
        return ChatResponse(
            session_id=session_id,
            intent="analytics_count_matches_for_panda",
            response=f"{resolved_name} has {top['count']} ranked matches in {top['source_view']}.",
            data={
                "panda_name": resolved_name,
                "count": top["count"],
                "source_view": top["source_view"],
            },
        )

    return None


def _looks_like_id(text_value: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9-]+$", text_value))


def _count_for_relation(
    session: Session,
    relation: str,
    *,
    panda_id: str | None = None,
    panda_name: str | None = None,
) -> int:
    cols = _relation_columns(session, schema="core", relation=relation)

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


def _diagnose_no_matches(session: Session, panda_name: str) -> dict[str, Any]:
    profiles = _rows(
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
        if _relation_exists(session, schema="core", relation=candidate):
            eligible_relation = candidate
            break

    eligible_count = 0
    if eligible_relation is not None:
        eligible_count = _count_for_relation(
            session,
            eligible_relation,
            panda_id=focal_id,
            panda_name=panda_name,
        )

    candidate_count = 0
    if _relation_exists(session, schema="core", relation="candidate_pairs"):
        candidate_count = _count_for_relation(session, "candidate_pairs", panda_id=focal_id)

    directional_count = 0
    if _relation_exists(session, schema="core", relation="directional_recommended_matches"):
        directional_count = _count_for_relation(
            session,
            "directional_recommended_matches",
            panda_id=focal_id,
            panda_name=panda_name,
        )

    ranked_count = 0
    if _relation_exists(session, schema="core", relation="ranked_directional_recommended_matches"):
        ranked_count = _count_for_relation(
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


def _curated_override_for_name(session: Session, panda_name: str) -> dict[str, Any] | None:
    if not _relation_exists(session, schema="core", relation="panda_profile_overrides"):
        return None
    rows = _rows(
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
    if rows:
        return rows[0]

    rows = _rows(
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
    return rows[0] if rows else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/chat", response_class=HTMLResponse)
def chat_ui() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Panda Matching Chat</title>
    <style>
      :root {
        --bg: #f7f4ed;
        --card: #fffdf8;
        --ink: #1f2937;
        --muted: #6b7280;
        --accent: #1f7a5c;
        --border: #e5dccd;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(1200px 500px at 10% -10%, #d7efe6 0%, transparent 60%),
          radial-gradient(800px 400px at 90% -5%, #f4e7c6 0%, transparent 60%),
          var(--bg);
      }
      .wrap {
        max-width: 900px;
        margin: 32px auto;
        padding: 0 16px;
      }
      .panel {
        border: 1px solid var(--border);
        background: var(--card);
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,.06);
        overflow: hidden;
      }
      .head {
        padding: 16px 20px;
        border-bottom: 1px solid var(--border);
      }
      .head h1 {
        margin: 0;
        font-size: 20px;
      }
      .head p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 14px;
      }
      .log {
        height: 58vh;
        overflow: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .msg {
        max-width: 85%;
        padding: 12px 14px;
        border-radius: 12px;
        line-height: 1.35;
        white-space: pre-wrap;
      }
      .user {
        align-self: flex-end;
        background: #ddf2eb;
        border: 1px solid #bce5d8;
      }
      .bot {
        align-self: flex-start;
        background: #fff;
        border: 1px solid var(--border);
      }
      .meta {
        display: block;
        margin-top: 6px;
        font-size: 12px;
        color: var(--muted);
      }
      .composer {
        border-top: 1px solid var(--border);
        padding: 12px;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 10px;
      }
      input[type="text"] {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        font-size: 14px;
      }
      button {
        border: 0;
        border-radius: 10px;
        padding: 0 16px;
        background: var(--accent);
        color: white;
        font-weight: 600;
        cursor: pointer;
      }
      button:disabled { opacity: .6; cursor: not-allowed; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="panel">
        <div class="head">
          <h1>Panda Matching Chat</h1>
          <p>Try: "top 5 matches for Ai Bao", "explain 16 81", "blockers for 16".</p>
        </div>
        <div id="log" class="log"></div>
        <form id="form" class="composer">
          <input id="input" type="text" placeholder="Ask about panda matches..." />
          <button id="send" type="submit">Send</button>
        </form>
      </div>
    </div>
    <script>
      const log = document.getElementById("log");
      const form = document.getElementById("form");
      const input = document.getElementById("input");
      const send = document.getElementById("send");
      let sessionId = null;

      function addMessage(kind, text, meta = "") {
        const div = document.createElement("div");
        div.className = `msg ${kind}`;
        div.textContent = text;
        if (meta) {
          const small = document.createElement("span");
          small.className = "meta";
          small.textContent = meta;
          div.appendChild(small);
        }
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
      }

      async function sendMessage(message) {
        send.disabled = true;
        try {
          const res = await fetch("/agent/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId })
          });
          const payload = await res.json();
          if (!res.ok) {
            addMessage("bot", payload.detail || "Request failed");
            return;
          }
          sessionId = payload.session_id;
          const body = payload.data ? JSON.stringify(payload.data, null, 2) : "";
          addMessage(
            "bot",
            payload.response + (body ? "\\n\\n" + body : ""),
            `intent=${payload.intent}`
          );
        } catch (err) {
          addMessage("bot", "Network error while calling /agent/chat");
        } finally {
          send.disabled = false;
        }
      }

      addMessage("bot", "Chat ready. Type 'help' for supported commands.");
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = input.value.trim();
        if (!message) return;
        addMessage("user", message);
        input.value = "";
        await sendMessage(message);
      });
    </script>
  </body>
</html>
"""


@app.get("/matches/top")
def get_top_matches(
    panda_name: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
) -> dict[str, Any]:
    session_maker = get_sessionmaker()
    with session_maker() as session:
        return _top_matches_data(session, panda_name=panda_name, k=k)


@app.get("/matches/explain")
def explain_match(
    focal_id: str = Query(..., min_length=1),
    candidate_id: str = Query(..., min_length=1),
) -> dict[str, Any]:
    session_maker = get_sessionmaker()
    with session_maker() as session:
        return _explain_match_data(session, focal_id=focal_id, candidate_id=candidate_id)


@app.get("/matches/blockers")
def get_blockers(
    focal_id: str = Query(..., min_length=1),
) -> dict[str, Any]:
    session_maker = get_sessionmaker()
    with session_maker() as session:
        return _blockers_data(session, focal_id=focal_id)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    response: str
    data: dict[str, Any] | None = None


@app.post("/agent/chat", response_model=ChatResponse)
def chat_agent(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    message = payload.message.strip()
    lower = message.lower()
    memory = CHAT_MEMORY.setdefault(session_id, {})

    session_maker = get_sessionmaker()
    with session_maker() as session:
        explain_match_pattern = re.match(r"^(?:explain|why)\s+(\S+)\s+(\S+)$", lower)
        top_pattern = re.search(
            r"(?:top|best)\s*(\d+)?\s*matches(?:\s+for)?\s+(.+)$",
            message,
            flags=re.IGNORECASE,
        )
        global_best_pattern = re.search(
            r"(?:best overall|overall best|best match across all|across all pandas|global best)",
            lower,
        )
        blockers_pattern = re.match(r"^blockers(?:\s+for)?\s+(.+)$", message, flags=re.IGNORECASE)
        profile_name = _extract_name_from_question(message)
        asks_age = "how old" in lower or "age of" in lower
        asks_location = "where is" in lower or "location of" in lower
        asks_health = "health" in lower
        asks_personality = "personality" in lower

        if global_best_pattern:
            result = _best_overall_match_data(session, k=1)
            if result["count"] == 0:
                return ChatResponse(
                    session_id=session_id,
                    intent="best_overall_empty",
                    response="I could not find any ranked matches in the current scoring view.",
                    data=result,
                )
            top = result["matches"][0]
            response = (
                "Best overall match across all focal pandas loaded. "
                "This is the highest-scoring directional pair in the current ranking view."
            )
            return ChatResponse(
                session_id=session_id,
                intent="best_overall",
                response=response,
                data={"best_match": top, "ranking_meta": result},
            )

        analytics_response = _analytics_chat_response(
            session=session,
            session_id=session_id,
            message=message,
            memory=memory,
        )
        if analytics_response is not None:
            return analytics_response

        if top_pattern:
            k_raw = top_pattern.group(1)
            panda_name = top_pattern.group(2).strip()
            k = int(k_raw) if k_raw else 5
            result = _top_matches_data(session, panda_name=panda_name, k=k)
            memory["last_panda_name"] = panda_name
            if result["count"] == 0:
                diagnosis = _diagnose_no_matches(session, panda_name=panda_name)
                curated = _curated_override_for_name(session, panda_name=panda_name)
                curated_bits: list[str] = []
                if curated:
                    if curated.get("personality_tags"):
                        curated_bits.append(f"personality: {curated['personality_tags']}")
                    if curated.get("health_notes"):
                        curated_bits.append(f"health: {curated['health_notes']}")
                curated_text = (
                    " Curated profile context -> " + " | ".join(curated_bits) + "."
                    if curated_bits
                    else ""
                )
                return ChatResponse(
                    session_id=session_id,
                    intent="top_matches_no_results",
                    response=diagnosis["summary"] + curated_text,
                    data={
                        "requested_top_k": k,
                        "panda_name": panda_name,
                        "matches": [],
                        "diagnosis": diagnosis,
                        "curated_profile": curated,
                    },
                )
            return ChatResponse(
                session_id=session_id,
                intent="top_matches",
                response=(
                    f"I found {result['count']} ranked matches for {panda_name}. "
                    "Returning the top candidates with score context."
                ),
                data=result,
            )

        if explain_match_pattern:
            focal_id = explain_match_pattern.group(1)
            candidate_id = explain_match_pattern.group(2)
            result = _explain_match_data(session, focal_id=focal_id, candidate_id=candidate_id)
            memory["last_focal_id"] = focal_id
            memory["last_candidate_id"] = candidate_id
            return ChatResponse(
                session_id=session_id,
                intent="explain_match",
                response=f"Loaded explanation for {focal_id} vs {candidate_id}.",
                data=result,
            )

        if blockers_pattern:
            focal_ref = blockers_pattern.group(1).strip()
            focal_id = (
                focal_ref
                if _looks_like_id(focal_ref)
                else _find_panda_id_by_name(session, focal_ref)
            )
            if not focal_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not resolve focal panda: {focal_ref}",
                )
            result = _blockers_data(session, focal_id=focal_id)
            memory["last_focal_id"] = focal_id
            return ChatResponse(
                session_id=session_id,
                intent="blockers",
                response=f"I found {result['count']} blocker rows for focal_id={focal_id}.",
                data=result,
            )

        if profile_name:
            resolved_name = _find_panda_name_by_substring(session, profile_name) or profile_name
            profile = _panda_profile_data(session, panda_name=resolved_name)
            if not profile:
                return ChatResponse(
                    session_id=session_id,
                    intent="panda_info_not_found",
                    response=f"I could not find a panda matching '{profile_name}'.",
                    data={"requested_name": profile_name},
                )

            memory["last_panda_name"] = str(profile.get("name") or resolved_name)
            if asks_age:
                response = (
                    f"{profile['name']} is approximately {profile.get('age_years')} years old."
                    if profile.get("age_years") is not None
                    else f"I do not have a computed age for {profile['name']}."
                )
            elif asks_location:
                location_bits = [
                    str(profile.get("zoo_or_facility") or "").strip(),
                    str(profile.get("city_region") or "").strip(),
                    str(profile.get("country") or "").strip(),
                ]
                location_text = ", ".join(bit for bit in location_bits if bit)
                response = (
                    f"{profile['name']} is currently listed at {location_text}."
                    if location_text
                    else f"I do not have a current location for {profile['name']}."
                )
            elif asks_health:
                response = (
                    f"Health notes for {profile['name']}: {profile.get('health_text')}"
                    if profile.get("health_text")
                    else f"I do not have health notes for {profile['name']}."
                )
            elif asks_personality:
                response = (
                    f"Personality notes for {profile['name']}: {profile.get('personality_text')}"
                    if profile.get("personality_text")
                    else f"I do not have personality notes for {profile['name']}."
                )
            else:
                age_text = (
                    str(profile.get("age_years"))
                    if profile.get("age_years") is not None
                    else "unknown"
                )
                response = (
                    f"{profile['name']} is a {profile.get('sex') or 'unknown-sex'} panda, "
                    f"status={profile.get('status') or 'unknown'}, "
                    f"age={age_text}."
                )

            return ChatResponse(
                session_id=session_id,
                intent="panda_info",
                response=response,
                data={"profile": profile},
            )

        if lower in {"help", "commands"}:
            return ChatResponse(
                session_id=session_id,
                intent="help",
                response=(
                    "Commands: "
                    "'top 5 matches for Bao Li' (or 'give me the top 5 matches for Bao Li'), "
                    "'explain <focal_id> <candidate_id>', "
                    "'blockers for <focal_id|name>', "
                    "'how many eligible pandas', 'count alive pandas', "
                    "'who is <name>', 'how old is <name>', 'where is <name>', "
                    "'health of <name>', 'personality of <name>'."
                ),
                data={"memory": memory},
            )

        if lower in {"top matches", "top"} and "last_panda_name" in memory:
            panda_name = memory["last_panda_name"]
            result = _top_matches_data(session, panda_name=panda_name, k=5)
            return ChatResponse(
                session_id=session_id,
                intent="top_matches",
                response=f"Found {result['count']} matches for {panda_name}.",
                data=result,
            )

        return ChatResponse(
            session_id=session_id,
            intent="fallback",
            response=(
                "I can help with matches. Try: "
                "'best overall panda match', "
                "'top 5 matches for Bao Li', "
                "'explain <focal_id> <candidate_id>', "
                "'blockers for <focal_id|name>', "
                "'how many eligible pandas', "
                "'who is <name>', 'health of <name>'."
            ),
            data={"memory": memory},
        )


def run() -> None:
    import uvicorn

    uvicorn.run("panda_matching.api:app", host="0.0.0.0", port=8000, reload=False)
