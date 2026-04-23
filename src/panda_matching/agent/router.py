from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from panda_matching.agent.llm import llm_compose_answer, llm_enabled, llm_plan_message
from panda_matching.agent.tools import (
    best_overall_match_data,
    blockers_data,
    curated_override_for_name,
    diagnose_no_matches,
    explain_match_data,
    find_panda_id_by_name,
    find_panda_name_by_substring,
    looks_like_id,
    panda_profile_data,
    pick_relation,
    relation_exists,
    scalar_int,
    top_matches_data,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    intent: str
    response: str
    data: dict[str, Any] | None = None


def llm_chat_response(
    session: Session,
    *,
    message: str,
    memory: dict[str, str],
) -> AgentTurn | None:
    if not llm_enabled():
        return None

    try:
        plan = llm_plan_message(message=message, memory=memory)
    except Exception:
        logger.exception("LLM planning failed; falling back to deterministic chat path")
        return None
    if not plan:
        return None

    mode = str(plan.get("mode") or "").lower()
    if mode == "respond":
        response_text = str(plan.get("response") or "").strip()
        if not response_text:
            return None
        return AgentTurn(intent="llm_respond", response=response_text, data=plan)

    if mode != "tool":
        return None

    tool = str(plan.get("tool") or "").strip().lower()
    args_raw = plan.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}

    if tool == "top_matches":
        panda_name = str(args.get("panda_name") or "").strip()
        if not panda_name:
            return None
        k = int(args.get("k") or 5)
        k = max(1, min(k, 20))
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        result = top_matches_data(session, panda_name=resolved_name, k=k)
        memory["last_panda_name"] = resolved_name
        response_text = llm_compose_answer(message, tool, result)
        return AgentTurn(intent="llm_top_matches", response=response_text, data=result)

    if tool == "panda_profile":
        panda_name = str(args.get("name") or "").strip()
        if not panda_name:
            return None
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        profile = panda_profile_data(session, panda_name=resolved_name)
        if not profile:
            return AgentTurn(
                intent="llm_panda_info_not_found",
                response=f"I could not find a panda matching '{panda_name}'.",
                data={"requested_name": panda_name},
            )
        memory["last_panda_name"] = str(profile.get("name") or resolved_name)
        response_text = llm_compose_answer(message, tool, {"profile": profile})
        return AgentTurn(
            intent="llm_panda_info",
            response=response_text,
            data={"profile": profile},
        )

    if tool == "explain_match":
        focal_id = str(args.get("focal_id") or "").strip()
        candidate_id = str(args.get("candidate_id") or "").strip()
        if not focal_id or not candidate_id:
            return None
        result = explain_match_data(session, focal_id=focal_id, candidate_id=candidate_id)
        memory["last_focal_id"] = focal_id
        memory["last_candidate_id"] = candidate_id
        response_text = llm_compose_answer(message, tool, result)
        return AgentTurn(intent="llm_explain_match", response=response_text, data=result)

    if tool == "blockers":
        focal_ref = str(args.get("focal_ref") or "").strip()
        if not focal_ref:
            return None
        blocker_focal_id: str | None = (
            focal_ref
            if looks_like_id(focal_ref)
            else find_panda_id_by_name(session, focal_ref)
        )
        if not blocker_focal_id:
            return AgentTurn(
                intent="llm_blockers_not_found",
                response=f"I could not resolve focal panda '{focal_ref}'.",
                data={"requested_focal_ref": focal_ref},
            )
        result = blockers_data(session, focal_id=blocker_focal_id)
        memory["last_focal_id"] = blocker_focal_id
        response_text = llm_compose_answer(message, tool, result)
        return AgentTurn(intent="llm_blockers", response=response_text, data=result)

    if tool == "best_overall":
        result = best_overall_match_data(session, k=1)
        response_text = llm_compose_answer(message, tool, result)
        return AgentTurn(intent="llm_best_overall", response=response_text, data=result)

    if tool == "count_eligible":
        relation = pick_relation(session, "core", ["breedeable_pandas", "breedable_pandas"])
        total = scalar_int(session, f"SELECT COUNT(*) AS n FROM core.{relation}")
        data = {"relation": relation, "count": total}
        response_text = llm_compose_answer(message, tool, data)
        return AgentTurn(intent="llm_count_eligible", response=response_text, data=data)

    if tool == "count_pandas":
        total = scalar_int(session, "SELECT COUNT(*) AS n FROM core.panda_profiles")
        data = {"relation": "core.panda_profiles", "count": total}
        response_text = llm_compose_answer(message, tool, data)
        return AgentTurn(intent="llm_count_pandas", response=response_text, data=data)

    if tool == "count_status":
        status = str(args.get("status") or "").strip().lower()
        if status == "dead":
            status = "deceased"
        if status not in {"alive", "deceased"}:
            return None
        total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(status, '')) = :status
            """,
            {"status": status},
        )
        data = {"status": status, "count": total}
        response_text = llm_compose_answer(message, tool, data)
        return AgentTurn(intent="llm_count_status", response=response_text, data=data)

    if tool == "count_sex":
        sex = str(args.get("sex") or "").strip().lower()
        if sex in {"m", "man", "boy"}:
            sex = "male"
        if sex in {"f", "woman", "girl"}:
            sex = "female"
        if sex not in {"male", "female"}:
            return None
        total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(sex, '')) = :sex
            """,
            {"sex": sex},
        )
        data = {"sex": sex, "count": total}
        response_text = llm_compose_answer(message, tool, data)
        return AgentTurn(intent="llm_count_sex", response=response_text, data=data)

    if tool == "count_curated":
        if not relation_exists(session, "core", "panda_profile_overrides"):
            data = {"relation": "core.panda_profile_overrides", "count": 0}
        else:
            total = scalar_int(session, "SELECT COUNT(*) AS n FROM core.panda_profile_overrides")
            data = {"relation": "core.panda_profile_overrides", "count": total}
        response_text = llm_compose_answer(message, tool, data)
        return AgentTurn(intent="llm_count_curated", response=response_text, data=data)

    if tool == "count_matches_for_panda":
        panda_name = str(args.get("panda_name") or "").strip()
        if not panda_name:
            return None
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        top = top_matches_data(session, panda_name=resolved_name, k=50)
        memory["last_panda_name"] = resolved_name
        data = {
            "panda_name": resolved_name,
            "count": top["count"],
            "source_view": top["source_view"],
        }
        response_text = llm_compose_answer(message, tool, data)
        return AgentTurn(intent="llm_count_matches_for_panda", response=response_text, data=data)

    return None


def extract_name_from_question(message: str) -> str | None:
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


def analytics_chat_response(
    session: Session,
    message: str,
    memory: dict[str, str],
) -> AgentTurn | None:
    lower = message.strip().lower()

    eligible_pat = re.match(r"^(?:how many|count)\s+eligible\s+pandas\??$", lower)
    if eligible_pat:
        relation = pick_relation(session, "core", ["breedeable_pandas", "breedable_pandas"])
        total = scalar_int(session, f"SELECT COUNT(*) AS n FROM core.{relation}")
        return AgentTurn(
            intent="analytics_count_eligible",
            response=f"There are {total} eligible pandas in core.{relation}.",
            data={"relation": relation, "count": total},
        )

    all_pat = re.match(r"^(?:how many|count)\s+pandas\??$", lower)
    if all_pat:
        total = scalar_int(session, "SELECT COUNT(*) AS n FROM core.panda_profiles")
        return AgentTurn(
            intent="analytics_count_pandas",
            response=f"There are {total} pandas in core.panda_profiles.",
            data={"relation": "core.panda_profiles", "count": total},
        )

    status_pat = re.match(r"^(?:how many|count)\s+(alive|deceased|dead)\s+pandas\??$", lower)
    if status_pat:
        status = status_pat.group(1)
        normalized = "deceased" if status == "dead" else status
        total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(status, '')) = :status
            """,
            {"status": normalized},
        )
        return AgentTurn(
            intent="analytics_count_status",
            response=f"There are {total} pandas with status '{normalized}'.",
            data={"status": normalized, "count": total},
        )

    sex_pat = re.match(
        (
            r"^(?:how many|count)\s+(male|female)\s+pandas"
            r"(?:\s+are\s+there(?:\s+in\s+my\s+data)?)?\??$"
        ),
        lower,
    )
    if sex_pat:
        sex = sex_pat.group(1)
        total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(sex, '')) = :sex
            """,
            {"sex": sex},
        )
        return AgentTurn(
            intent="analytics_count_sex",
            response=f"There are {total} {sex} pandas in core.panda_profiles.",
            data={"sex": sex, "count": total},
        )

    curated_pat = re.match(r"^(?:how many|count)\s+curated\s+(?:profiles|pandas)\??$", lower)
    if curated_pat:
        if not relation_exists(session, "core", "panda_profile_overrides"):
            return AgentTurn(
                intent="analytics_curated_missing",
                response="Curated override table is not available yet in this database.",
                data={"relation": "core.panda_profile_overrides", "count": 0},
            )
        total = scalar_int(session, "SELECT COUNT(*) AS n FROM core.panda_profile_overrides")
        return AgentTurn(
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
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        top = top_matches_data(session, panda_name=resolved_name, k=50)
        memory["last_panda_name"] = resolved_name
        return AgentTurn(
            intent="analytics_count_matches_for_panda",
            response=f"{resolved_name} has {top['count']} ranked matches in {top['source_view']}.",
            data={
                "panda_name": resolved_name,
                "count": top["count"],
                "source_view": top["source_view"],
            },
        )

    return None


def route_chat_message(session: Session, message: str, memory: dict[str, str]) -> AgentTurn:
    lower = message.lower()

    llm_response = llm_chat_response(session, message=message, memory=memory)
    if llm_response is not None:
        return llm_response

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
    profile_name = extract_name_from_question(message)
    asks_age = "how old" in lower or "age of" in lower
    asks_location = "where is" in lower or "location of" in lower
    asks_health = "health" in lower
    asks_personality = "personality" in lower

    if global_best_pattern:
        result = best_overall_match_data(session, k=1)
        if result["count"] == 0:
            return AgentTurn(
                intent="best_overall_empty",
                response="I could not find any ranked matches in the current scoring view.",
                data=result,
            )
        top = result["matches"][0]
        response = (
            "Best overall match across all focal pandas loaded. "
            "This is the highest-scoring directional pair in the current ranking view."
        )
        return AgentTurn(
            intent="best_overall",
            response=response,
            data={"best_match": top, "ranking_meta": result},
        )

    analytics_response = analytics_chat_response(session=session, message=message, memory=memory)
    if analytics_response is not None:
        return analytics_response

    if top_pattern:
        k_raw = top_pattern.group(1)
        panda_name = top_pattern.group(2).strip()
        k = int(k_raw) if k_raw else 5
        result = top_matches_data(session, panda_name=panda_name, k=k)
        memory["last_panda_name"] = panda_name
        if result["count"] == 0:
            diagnosis = diagnose_no_matches(session, panda_name=panda_name)
            curated = curated_override_for_name(session, panda_name=panda_name)
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
            return AgentTurn(
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
        return AgentTurn(
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
        result = explain_match_data(session, focal_id=focal_id, candidate_id=candidate_id)
        memory["last_focal_id"] = focal_id
        memory["last_candidate_id"] = candidate_id
        return AgentTurn(
            intent="explain_match",
            response=f"Loaded explanation for {focal_id} vs {candidate_id}.",
            data=result,
        )

    if blockers_pattern:
        focal_ref = blockers_pattern.group(1).strip()
        focal_id = (
            focal_ref
            if looks_like_id(focal_ref)
            else find_panda_id_by_name(session, focal_ref)
        )
        if not focal_id:
            raise ValueError(f"Could not resolve focal panda: {focal_ref}")
        result = blockers_data(session, focal_id=focal_id)
        memory["last_focal_id"] = focal_id
        return AgentTurn(
            intent="blockers",
            response=f"I found {result['count']} blocker rows for focal_id={focal_id}.",
            data=result,
        )

    if profile_name:
        resolved_name = find_panda_name_by_substring(session, profile_name) or profile_name
        profile = panda_profile_data(session, panda_name=resolved_name)
        if not profile:
            return AgentTurn(
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

        return AgentTurn(intent="panda_info", response=response, data={"profile": profile})

    if lower in {"help", "commands"}:
        return AgentTurn(
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
        result = top_matches_data(session, panda_name=panda_name, k=5)
        return AgentTurn(
            intent="top_matches",
            response=f"Found {result['count']} matches for {panda_name}.",
            data=result,
        )

    return AgentTurn(
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
