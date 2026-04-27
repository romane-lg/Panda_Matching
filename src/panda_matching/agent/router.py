from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from panda_matching.agent.llm import llm_compose_answer, llm_enabled, llm_plan_message
from panda_matching.agent.tools import (
    best_overall_match_data,
    blockers_data,
    compare_candidates_for_focal_data,
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
from panda_matching.observability import SpanType, trace

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    intent: str
    response: str
    data: dict[str, Any] | None = None


def _rank_number(row: dict[str, Any]) -> int | None:
    for key in ("recommendation_rank_v2", "recommendation_rank"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _score_number(row: dict[str, Any]) -> float | None:
    for key in ("final_score_v2", "recommendation_score", "base_score"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(str(value))
        except (TypeError, ValueError):
            continue
    return None


def _find_candidate_row(
    matches: list[dict[str, Any]],
    candidate_name: str,
) -> dict[str, Any] | None:
    lowered = candidate_name.strip().lower()
    for row in matches:
        name = str(row.get("candidate_panda_name") or "").strip()
        if name.lower() == lowered:
            return row
    for row in matches:
        name = str(row.get("candidate_panda_name") or "").strip()
        if lowered in name.lower():
            return row
    return None


def _clean_factor(text: Any) -> str:
    return str(text or "").strip().strip(".")


def _format_rank_score(row: dict[str, Any]) -> str:
    rank = _rank_number(row)
    score = _score_number(row)
    parts: list[str] = []
    if rank is not None:
        parts.append(f"#{rank}")
    if score is not None:
        parts.append(f"score {score:.3f}")
    return ", ".join(parts)


def _format_component_sentence(row: dict[str, Any]) -> str | None:
    bits: list[str] = []
    age_gap_years = row.get("age_gap_years")
    bio_component = row.get("bio_component")
    behavior_component = row.get("behavior_component")
    logistics_component = row.get("logistics_component")

    if age_gap_years not in (None, ""):
        bits.append(f"an age gap of {age_gap_years} years")
    if bio_component not in (None, ""):
        bits.append(f"a bio component of {bio_component}")
    if behavior_component not in (None, ""):
        bits.append(f"a behavior component of {behavior_component}")
    if logistics_component not in (None, ""):
        bits.append(f"a logistics component of {logistics_component}")

    if not bits:
        return None
    if len(bits) == 1:
        return bits[0]
    if len(bits) == 2:
        return f"{bits[0]} and {bits[1]}"
    return ", ".join(bits[:-1]) + f", and {bits[-1]}"


def _deterministic_top_matches_response(
    *,
    focal_name: str,
    matches: list[dict[str, Any]],
    requested_k: int,
) -> str:
    if not matches:
        return f"I could not find any ranked matches for {focal_name}."

    selected = matches[:requested_k]
    top_names = [
        str(row.get("candidate_panda_name") or "").strip()
        for row in selected
        if row.get("candidate_panda_name")
    ]
    lead = selected[0]
    lead_name = str(lead.get("candidate_panda_name") or "the current leader").strip()
    lead_positive = _clean_factor(lead.get("top_positive_factors"))
    lead_negative = _clean_factor(lead.get("top_negative_factors"))
    lead_score = _score_number(lead)

    if len(top_names) == 1:
        intro = f"{focal_name}'s current top match is {top_names[0]}."
    elif len(top_names) == 2:
        intro = f"{focal_name}'s current top matches are {top_names[0]} and {top_names[1]}."
    else:
        intro = (
            f"{focal_name}'s current top {len(top_names)} matches are "
            + ", ".join(top_names[:-1])
            + f", and {top_names[-1]}."
        )

    lead_sentence = f"{lead_name} leads the list"
    if lead_score is not None:
        lead_sentence += f" with a score of {lead_score:.3f}"
    if lead_positive:
        lead_sentence += f", helped most by {lead_positive}"
    lead_sentence += "."

    followup: list[str] = []
    if len(selected) > 1:
        runner_up = selected[1]
        runner_name = str(runner_up.get("candidate_panda_name") or "the next candidate").strip()
        runner_score = _score_number(runner_up)
        if lead_score is not None and runner_score is not None:
            gap = lead_score - runner_score
            followup.append(
                f"{runner_name} is next, trailing by {gap:.3f} points, "
                "so the top of the list is fairly tight."
            )
        else:
            followup.append(f"{runner_name} sits right behind in the current ranking.")

    if lead_negative and lead_negative.lower() != "no major penalties":
        followup.append(
            f"The main tradeoff even for {lead_name} is {lead_negative}, "
            "but the stronger positives still keep the pair at the top."
        )
    elif lead_negative:
        followup.append(
            f"There are no major penalties flagged for {lead_name} in the current scoring view."
        )

    return " ".join([intro, lead_sentence, *followup]).strip()


def _deterministic_pairwise_response(
    *,
    focal_name: str,
    result: dict[str, Any],
) -> str:
    better = result["better_match"]
    assert better is not None

    candidate_a = result["candidate_a"]
    candidate_b = result["candidate_b"]
    better_name = str(better.get("candidate_panda_name") or "the stronger candidate").strip()

    def _summary(row: dict[str, Any] | None) -> str | None:
        if row is None:
            return None
        name = str(row.get("candidate_panda_name") or "").strip()
        if not name:
            return None
        positive = _clean_factor(row.get("top_positive_factors"))
        negative = _clean_factor(row.get("top_negative_factors"))
        rank = _rank_number(row)
        score = _score_number(row)
        sentence = name
        if rank is not None and score is not None:
            sentence += f" sits at #{rank} with a score of {score:.3f}"
        elif rank is not None:
            sentence += f" sits at #{rank}"
        if positive:
            sentence += f", and its clearest strength is {positive}"
        sentence += "."
        if negative and negative.lower() != "no major penalties":
            sentence += f" The main drawback in the row is {negative}."
        return sentence

    better_rank = _rank_number(better)
    better_score = _score_number(better)
    other = candidate_b if better is candidate_a else candidate_a
    other_name = (
        str(other.get("candidate_panda_name") or "").strip()
        if other is not None
        else "the other candidate"
    )
    other_score = _score_number(other) if other is not None else None

    intro = (
        f"Between {result['candidate_a_name']} and {result['candidate_b_name']}, "
        f"{better_name} looks like the stronger match for {focal_name} right now."
    )
    comparison = ""
    if better_rank is not None and better_score is not None:
        comparison = (
            f"{better_name} is currently ranked #{better_rank} "
            f"with a score of {better_score:.3f}."
        )
        if other_score is not None:
            comparison += (
                f" That puts {other_name} behind by {better_score - other_score:.3f} points, "
                "so the margin is real but not huge."
            )

    pieces = [intro]
    if comparison:
        pieces.append(comparison)
    better_summary = _summary(better)
    other_summary = _summary(other)
    if better_summary:
        pieces.append(better_summary)
    if other_summary:
        pieces.append(f"For comparison, {other_summary}")
    return " ".join(pieces)


def _deterministic_no_match_response(
    *,
    diagnosis: dict[str, Any],
    curated: dict[str, Any] | None = None,
) -> str:
    summary = str(diagnosis.get("summary") or "").strip()
    waterfall = diagnosis.get("waterfall") or {}
    profile = diagnosis.get("profile") or {}

    details: list[str] = []
    eligible_rows = waterfall.get("eligible_rows")
    candidate_rows = waterfall.get("candidate_pairs_rows")
    directional_rows = waterfall.get("directional_rows")
    ranked_rows = waterfall.get("ranked_rows")

    if eligible_rows == 0:
        details.append("the panda does not appear in the eligible breeding set")
    if candidate_rows == 0:
        details.append("no candidate pairs were generated at all")
    elif directional_rows == 0:
        details.append("candidate pairs exist, but none survive into directional scoring")
    elif ranked_rows == 0:
        details.append("scored directional pairs exist, but none make it into the ranked output")

    if profile.get("status") and str(profile["status"]).lower() != "alive":
        details.append(f"the recorded status is {profile['status']}")

    if curated and curated.get("health_notes"):
        details.append(f"curated health context says {curated['health_notes']}")

    if not details:
        return summary

    return (
        f"{summary} Looking at the pipeline, this usually means that "
        + "; ".join(details)
        + "."
    )


def _deterministic_rank_explanation(
    *,
    focal_name: str,
    candidate_name: str,
    candidate_row: dict[str, Any],
    top_row: dict[str, Any] | None,
) -> str:
    rank = _rank_number(candidate_row)
    score = _score_number(candidate_row)
    positive = _clean_factor(candidate_row.get("top_positive_factors"))
    negative = _clean_factor(candidate_row.get("top_negative_factors"))
    if rank is not None and score is not None:
        intro = f"{candidate_name} ranks #{rank} for {focal_name} with a score of {score:.3f}."
    elif rank is not None:
        intro = f"{candidate_name} ranks #{rank} for {focal_name}."
    else:
        intro = f"{candidate_name} is one of the top-ranked matches for {focal_name}."

    strengths: list[str] = []
    if positive:
        strengths.append(f"the biggest strength is {positive}")

    component_sentence = _format_component_sentence(candidate_row)
    if component_sentence:
        strengths.append("the score is also being supported by " + component_sentence)

    comparison: str | None = None
    if top_row is not None and top_row is not candidate_row:
        top_name = str(top_row.get("candidate_panda_name") or "the top candidate").strip()
        top_score = _score_number(top_row)
        if score is not None and top_score is not None:
            gap = top_score - score
            comparison = (
                f"It sits behind {top_name} by only {gap:.3f} points, so the gap is narrow "
                "rather than a major drop in compatibility."
            )
        else:
            comparison = f"It sits just behind {top_name} in the ranking."

    caveat: str | None = None
    if negative and negative.lower() != "no major penalties":
        caveat = (
            f"The main tradeoff is {negative}, but that drawback is not "
            "strong enough to cancel out the positives."
        )
    elif negative:
        caveat = "There are no major penalties flagged in the current scoring view."

    if not strengths and not comparison and not caveat:
        return intro

    pieces = [intro]
    if strengths:
        pieces.append("What stands out most is that " + "; ".join(strengths) + ".")
    if comparison:
        pieces.append(comparison)
    if caveat:
        pieces.append(caveat)
    return " ".join(pieces)


def _resolve_subject_pair(
    session: Session,
    combined_subject: str,
) -> tuple[str, str] | None:
    words = combined_subject.strip().split()
    if len(words) < 2:
        return None

    for split_idx in range(1, len(words)):
        candidate_hint = " ".join(words[:split_idx]).strip()
        focal_hint = " ".join(words[split_idx:]).strip()
        if not candidate_hint or not focal_hint:
            continue
        resolved_candidate = find_panda_name_by_substring(session, candidate_hint)
        resolved_focal = find_panda_name_by_substring(session, focal_hint)
        if resolved_candidate and resolved_focal:
            return resolved_candidate, resolved_focal
    return None


def _profile_fun_fact_response(profile: dict[str, Any]) -> str:
    name = str(profile.get("name") or "This panda")
    babies = profile.get("babies_had_count")
    location_bits = [
        str(profile.get("zoo_or_facility") or "").strip(),
        str(profile.get("country") or "").strip(),
    ]
    location_text = ", ".join(bit for bit in location_bits if bit)
    description = str(profile.get("description_text") or "").strip()
    personality = str(profile.get("personality_text") or "").strip()

    if babies not in (None, ""):
        if description:
            return (
                f"Fun fact: {name} is a mother of {babies} cubs. "
                f"According to the profile, {description.lower()}"
            )
        if location_text:
            return (
                f"Fun fact: {name} is a mother of {babies} cubs and is currently listed at "
                f"{location_text}."
            )
        return f"Fun fact: {name} is a mother of {babies} cubs."
    if description:
        return f"Fun fact: {description}"
    if personality:
        if location_text:
            return (
                f"Fun fact: {name} is known for being {personality} and is currently listed at "
                f"{location_text}."
            )
        return f"Fun fact: {name} is known for being {personality}."
    if location_text:
        return f"Fun fact: {name} lives at {location_text}."
    return f"Fun fact: {name} is part of the panda matching dataset."


@trace(name="llm_chat_response", span_type=SpanType.AGENT)
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

    try:
        return _execute_llm_tool(
            session=session,
            message=message,
            memory=memory,
            tool=tool,
            args=args,
        )
    except HTTPException:
        logger.exception("LLM tool execution failed; falling back to deterministic chat path")
        return None
    except Exception:
        logger.exception("Unexpected LLM tool failure; falling back to deterministic chat path")
        return None


def _execute_llm_tool(
    session: Session,
    *,
    message: str,
    memory: dict[str, str],
    tool: str,
    args: dict[str, Any],
) -> AgentTurn | None:

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
        if "fun fact" in message.lower():
            response_text = _profile_fun_fact_response(profile)
        else:
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

    if tool == "compare_candidates":
        focal_panda_name = str(args.get("focal_panda_name") or "").strip()
        candidate_a_name = str(args.get("candidate_a_name") or "").strip()
        candidate_b_name = str(args.get("candidate_b_name") or "").strip()
        if not focal_panda_name or not candidate_a_name or not candidate_b_name:
            return None
        resolved_focal = (
            find_panda_name_by_substring(session, focal_panda_name) or focal_panda_name
        )
        result = compare_candidates_for_focal_data(
            session,
            focal_panda_name=resolved_focal,
            candidate_a_name=candidate_a_name,
            candidate_b_name=candidate_b_name,
        )
        memory["last_panda_name"] = resolved_focal
        response_text = llm_compose_answer(message, tool, result)
        return AgentTurn(intent="llm_compare_candidates", response=response_text, data=result)

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
        r"^(?:who is|tell me(?:\s+more)? about|profile of)\s+(.+?)(?:[.?!].*)?$",
        r"^(?:tell me\s+a\s+fun\s+fact\s+about)\s+(.+)$",
        r"^(?:how old is|where is|health of|personality of)\s+(.+)$",
        r"^(?:what is the health of|what is the personality of)\s+(.+)$",
        r"^is\s+(.+?)\s+(?:healthy|in good health(?: conditions)?)\??$",
        r"^how is\s+(.+?)'s\s+health\??$",
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

    eligible_pat = re.match(
        r"^(?:how many|count)\s+eligible\s+pandas(?:\s+are\s+there)?\??$",
        lower,
    )
    if eligible_pat:
        relation = pick_relation(session, "core", ["breedeable_pandas", "breedable_pandas"])
        total = scalar_int(session, f"SELECT COUNT(*) AS n FROM core.{relation}")
        return AgentTurn(
            intent="analytics_count_eligible",
            response=f"There are {total} eligible pandas in core.{relation}.",
            data={"relation": relation, "count": total},
        )

    all_pat = re.match(
        r"^(?:how many|count)\s+pandas(?:\s+are\s+there)?(?:\s+in\s+total)?\??$",
        lower,
    )
    if all_pat:
        total = scalar_int(session, "SELECT COUNT(*) AS n FROM core.panda_profiles")
        return AgentTurn(
            intent="analytics_count_pandas",
            response=f"There are {total} pandas in core.panda_profiles.",
            data={"relation": "core.panda_profiles", "count": total},
        )

    sex_split_pat = re.match(
        (
            r"^(?:how many|count)\s+males?\s+to\s+females?\s+"
            r"(?:are\s+(?:there\s+)?)?(?:in\s+the\s+dataset|in\s+my\s+data)\??$"
        ),
        lower,
    )
    if sex_split_pat:
        male_total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(sex, '')) = 'male'
            """,
        )
        female_total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(sex, '')) = 'female'
            """,
        )
        return AgentTurn(
            intent="analytics_count_sex_split",
            response=(
                f"There are {male_total} male pandas and {female_total} female pandas "
                "in core.panda_profiles."
            ),
            data={"male_count": male_total, "female_count": female_total},
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


def unsupported_chat_response(message: str) -> AgentTurn | None:
    lower = message.strip().lower()
    if "poem" in lower and "panda" in lower:
        return AgentTurn(
            intent="unsupported_request",
            response=(
                "I focus on panda matching and profile questions. "
                "I cannot write creative pieces here, but I can help with matches, "
                "profiles, health notes, and ranking explanations."
            ),
            data=None,
        )
    if "weather" in lower:
        return AgentTurn(
            intent="unsupported_request",
            response=(
                "I focus on panda matching and profile questions, so I cannot answer "
                "weather questions here."
            ),
            data=None,
        )
    if "breeding season" in lower:
        breeding_match = re.match(
            r"^is\s+(.+?)\s+in\s+breeding\s+season(?:\s+right\s+now)?\??$",
            message.strip(),
            flags=re.IGNORECASE,
        )
        profile_name = (
            breeding_match.group(1).strip(" ?.")
            if breeding_match
            else extract_name_from_question(message)
        )
        if profile_name:
            return AgentTurn(
                intent="breeding_season_uncertain",
                response=(
                    f"I do not have live breeding-season timing data for {profile_name} in the "
                    "current profile and matching tables, so I cannot confirm that reliably."
                ),
                data=None,
            )
        return AgentTurn(
            intent="breeding_season_uncertain",
            response=(
                "I do not have live breeding-season timing data in the current profile and "
                "matching tables, so I cannot confirm that reliably."
            ),
            data=None,
        )
    return None


@trace(name="route_chat_message", span_type=SpanType.AGENT)
def route_chat_message(session: Session, message: str, memory: dict[str, str]) -> AgentTurn:
    lower = message.lower()

    explain_match_pattern = re.match(r"^(?:explain|why)\s+(\S+)\s+(\S+)$", lower)
    top_pattern = re.search(
        r"(?:top|best)\s*(\d+)?\s*matches(?:\s+for)?\s+(.+)$",
        message,
        flags=re.IGNORECASE,
    )
    possessive_top_pattern = re.match(
        r"^(?:who\s+are\s+)?(.+?)'s\s+(?:top|best)\s*(\d+)?\s+matches\??$",
        message.strip(),
        flags=re.IGNORECASE,
    )
    first_match_explanation_pattern = re.match(
        r"^why\s+is\s+(.+)\s+(.+?)'s\s+first\s+top\s+match\??$",
        message.strip(),
        flags=re.IGNORECASE,
    )
    second_match_explanation_pattern = re.match(
        r"^why\s+is\s+(.+?)\s+only\s+second\s+for\s+(.+?)\??$",
        message.strip(),
        flags=re.IGNORECASE,
    )
    no_matches_pattern = re.match(
        r"^why\s+does\s+(.+?)\s+have\s+no\s+matches\??$",
        message.strip(),
        flags=re.IGNORECASE,
    )
    pairwise_pattern = re.match(
        r"^is\s+(.+?)\s+or\s+(.+?)\s+better\s+for\s+(.+?)(?:,?\s*and why)?\??$",
        message.strip(),
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
    asks_fun_fact = "fun fact" in lower

    unsupported_response = unsupported_chat_response(message)
    if unsupported_response is not None:
        return unsupported_response

    analytics_response = analytics_chat_response(session=session, message=message, memory=memory)
    if analytics_response is not None:
        return analytics_response

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
        elif asks_fun_fact:
            response = _profile_fun_fact_response(profile)
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

    llm_response = llm_chat_response(session, message=message, memory=memory)
    if llm_response is not None:
        return llm_response

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

    if pairwise_pattern:
        candidate_a_name = pairwise_pattern.group(1).strip(" ?.")
        candidate_b_name = pairwise_pattern.group(2).strip(" ?.")
        focal_panda_name = pairwise_pattern.group(3).strip(" ?.")
        resolved_focal = find_panda_name_by_substring(session, focal_panda_name) or focal_panda_name
        result = compare_candidates_for_focal_data(
            session,
            focal_panda_name=resolved_focal,
            candidate_a_name=candidate_a_name,
            candidate_b_name=candidate_b_name,
        )
        memory["last_panda_name"] = resolved_focal
        if result["missing_candidates"]:
            missing = ", ".join(result["missing_candidates"])
            return AgentTurn(
                intent="compare_candidates_partial",
                response=(
                    f"I could not find ranked match rows for {missing} under {resolved_focal}. "
                    "Try asking for the top matches first or use exact candidate names."
                ),
                data=result,
            )
        better = result["better_match"]
        assert better is not None
        return AgentTurn(
            intent="compare_candidates",
            response=_deterministic_pairwise_response(
                focal_name=resolved_focal,
                result=result,
            ),
            data=result,
        )

    if top_pattern:
        k_raw = top_pattern.group(1)
        panda_name = top_pattern.group(2).strip()
        k = int(k_raw) if k_raw else 5
        result = top_matches_data(session, panda_name=panda_name, k=k)
        memory["last_panda_name"] = panda_name
        if result["count"] == 0:
            diagnosis = diagnose_no_matches(session, panda_name=panda_name)
            curated = curated_override_for_name(session, panda_name=panda_name)
            return AgentTurn(
                intent="top_matches_no_results",
                response=_deterministic_no_match_response(
                    diagnosis=diagnosis,
                    curated=curated,
                ),
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
            response=_deterministic_top_matches_response(
                focal_name=panda_name,
                matches=result["matches"],
                requested_k=k,
            ),
            data=result,
        )

    if possessive_top_pattern:
        panda_name = possessive_top_pattern.group(1).strip()
        k_raw = possessive_top_pattern.group(2)
        k = int(k_raw) if k_raw else 5
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        result = top_matches_data(session, panda_name=resolved_name, k=k)
        memory["last_panda_name"] = resolved_name
        if result["count"] == 0:
            return AgentTurn(
                intent="top_matches_no_results",
                response=_deterministic_no_match_response(
                    diagnosis=diagnose_no_matches(session, panda_name=resolved_name),
                    curated=curated_override_for_name(session, panda_name=resolved_name),
                ),
                data=result,
            )
        return AgentTurn(
            intent="top_matches",
            response=_deterministic_top_matches_response(
                focal_name=resolved_name,
                matches=result["matches"],
                requested_k=k,
            ),
            data=result,
        )

    if first_match_explanation_pattern:
        combined_subject = first_match_explanation_pattern.group(1).strip(" ?.")
        focal_name = first_match_explanation_pattern.group(2).strip(" ?.")
        resolved_pair = _resolve_subject_pair(session, f"{combined_subject} {focal_name}")
        if resolved_pair is not None:
            candidate_name, resolved_name = resolved_pair
        else:
            candidate_name = combined_subject
            resolved_name = find_panda_name_by_substring(session, focal_name) or focal_name
        result = top_matches_data(session, panda_name=resolved_name, k=10)
        matches = result["matches"]
        candidate_row = _find_candidate_row(matches, candidate_name)
        top_row = matches[0] if matches else None
        if not candidate_row:
            return AgentTurn(
                intent="match_explanation_not_found",
                response=(
                    f"I could not find {candidate_name} in the current top matches "
                    f"for {resolved_name}."
                ),
                data=result,
            )
        memory["last_panda_name"] = resolved_name
        return AgentTurn(
            intent="match_explanation",
            response=_deterministic_rank_explanation(
                focal_name=resolved_name,
                candidate_name=str(candidate_row.get("candidate_panda_name") or candidate_name),
                candidate_row=candidate_row,
                top_row=top_row,
            ),
            data={
                "panda_name": resolved_name,
                "candidate_name": candidate_name,
                "match": candidate_row,
            },
        )

    if second_match_explanation_pattern:
        candidate_name = second_match_explanation_pattern.group(1).strip(" ?.")
        focal_name = second_match_explanation_pattern.group(2).strip(" ?.")
        resolved_name = find_panda_name_by_substring(session, focal_name) or focal_name
        result = top_matches_data(session, panda_name=resolved_name, k=10)
        matches = result["matches"]
        candidate_row = _find_candidate_row(matches, candidate_name)
        top_row = matches[0] if matches else None
        if not candidate_row:
            return AgentTurn(
                intent="match_explanation_not_found",
                response=(
                    f"I could not find {candidate_name} in the current top matches "
                    f"for {resolved_name}."
                ),
                data=result,
            )
        memory["last_panda_name"] = resolved_name
        return AgentTurn(
            intent="match_explanation",
            response=_deterministic_rank_explanation(
                focal_name=resolved_name,
                candidate_name=str(candidate_row.get("candidate_panda_name") or candidate_name),
                candidate_row=candidate_row,
                top_row=top_row,
            ),
            data={
                "panda_name": resolved_name,
                "candidate_name": candidate_name,
                "match": candidate_row,
            },
        )

    if no_matches_pattern:
        panda_name = no_matches_pattern.group(1).strip(" ?.")
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        top = top_matches_data(session, panda_name=resolved_name, k=3)
        if top["count"] > 0:
            lead = top["matches"][0]
            lead_name = str(lead.get("candidate_panda_name") or "the current top match").strip()
            lead_score = _score_number(lead)
            lead_positive = _clean_factor(lead.get("top_positive_factors"))
            response = f"{resolved_name} actually does have ranked matches in the current data."
            if lead_score is not None:
                response += (
                    f" Right now {lead_name} is the top match "
                    f"with a score of {lead_score:.3f}"
                )
            else:
                response += f" Right now {lead_name} is the top match"
            if lead_positive:
                response += f", helped by {lead_positive}"
            response += "."
            return AgentTurn(
                intent="no_match_diagnosis",
                response=response,
                data={"panda_name": resolved_name, "matches": top["matches"]},
            )

        diagnosis = diagnose_no_matches(session, panda_name=resolved_name)
        curated = curated_override_for_name(session, panda_name=resolved_name)
        return AgentTurn(
            intent="no_match_diagnosis",
            response=_deterministic_no_match_response(
                diagnosis=diagnosis,
                curated=curated,
            ),
            data={
                "panda_name": resolved_name,
                "matches": [],
                "diagnosis": diagnosis,
                "curated_profile": curated,
            },
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
