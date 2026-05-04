from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from panda_matching.agent.llm import (
    LLMRateLimitError,
    llm_compose_answer,
    llm_enabled,
    llm_plan_message,
)
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
from panda_matching.observability import SpanType, start_span, trace

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    intent: str
    response: str
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class IntentDecision:
    category: str
    confidence: float
    names: tuple[str, ...] = ()
    needs_clarification: bool = False
    reason: str = ""


def _name_like(text: str) -> str:
    return text.strip(" ?.!")


def _clean_panda_name_hint(text: str) -> str:
    cleaned = text.strip(" ?.")
    cleaned = re.sub(
        r"\s+(?:and\s+)?(?:why|explain|with\s+explanation|with\s+details)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ?.")


def _normalize_location_hint(location: str) -> tuple[str, str, str]:
    cleaned = location.strip(" ?.")
    lowered = cleaned.lower()
    aliases = {
        "america": ("USA", "USA"),
        "usa": ("USA", "USA"),
        "us": ("USA", "USA"),
        "u.s.": ("USA", "USA"),
        "u.s.a.": ("USA", "USA"),
        "united states": ("USA", "USA"),
        "the united states": ("USA", "USA"),
        "south korea": "South Korea",
        "korea": "South Korea",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
    }
    alias = aliases.get(lowered, cleaned)
    if isinstance(alias, tuple):
        query_value, display_value = alias
    else:
        query_value = " ".join(part.capitalize() for part in str(alias).split())
        display_value = " ".join(part.capitalize() for part in str(alias).split())
    return query_value, display_value, cleaned


def classify_chat_intent(message: str, memory: dict[str, str]) -> IntentDecision:
    query_text = re.split(r"[.?!]\s+", message.strip(), maxsplit=1)[0].strip()
    lower = query_text.lower()

    if re.match(r"^(?:what|how)\s+about\s+.+$", query_text, flags=re.IGNORECASE):
        last_query_kind = memory.get("last_query_kind", "")
        if last_query_kind == "top_matches":
            return IntentDecision(
                category="ranking",
                confidence=0.8,
                reason="follow-up ranking request",
            )

    if re.match(
        r"^why\s+is\s+.+?'s\s+first\s+top\s+match\??$",
        query_text,
        flags=re.IGNORECASE,
    ) or re.match(
        r"^why\s+is\s+.+?\s+only\s+second\s+for\s+.+?\??$",
        query_text,
        flags=re.IGNORECASE,
    ):
        return IntentDecision(
            category="explanation",
            confidence=0.9,
            reason="rank explanation language",
        )

    pair_match = re.match(
        r"^(?:would|could|should|is|are)\s+(.+?)\s+and\s+(.+?)\s+"
        r"(?:make\s+)?(?:a\s+)?(?:good|great|strong|compatible|suitable)?\s*"
        r"(?:match|pair|pairing|breeding\s+pair|compatible)\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    if pair_match:
        return IntentDecision(
            category="match",
            confidence=0.9,
            names=(_name_like(pair_match.group(1)), _name_like(pair_match.group(2))),
            reason="two panda names compared for compatibility",
        )

    pairwise_match = re.match(
        r"^is\s+(.+?)\s+or\s+(.+?)\s+better\s+for\s+(.+?)(?:,?\s*and why)?\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    if pairwise_match:
        return IntentDecision(
            category="match",
            confidence=0.9,
            names=(
                _name_like(pairwise_match.group(1)),
                _name_like(pairwise_match.group(2)),
                _name_like(pairwise_match.group(3)),
            ),
            reason="candidate comparison for one focal panda",
        )

    explicit_top = re.search(
        r"(?:top|best)\s*(\d+)?\s*matches(?:\s+for)?\s+(.+)$",
        query_text,
        flags=re.IGNORECASE,
    )
    possessive_top = re.match(
        r"^(?:who\s+are\s+)?(.+?)'s\s+(?:top|best)\s*(\d+)?\s+matches\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    name_first_top = re.match(
        r"^(?:(?:who\s+are|whoa\s+re|who\s+rae|show\s+me|give\s+me)\s+)?"
        r"(.+?)\s+(?:top|best)\s*(\d+)?\s+matches\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    global_best = re.search(
        (
            r"(?:best overall|overall best|best match across all|across all pandas|"
            r"global best|most eligible panda|most eligible match|best panda match|"
            r"best overall pair|recommend(?:\s+me)?\s+a\s+panda match|"
            r"tell me about a panda match)"
        ),
        lower,
    )
    best_for = re.match(
        r"^(?:who\s+is\s+|what\s+is\s+|show\s+me\s+|find\s+)?(?:the\s+)?"
        r"(?:best|top|ideal|strongest|most\s+compatible)\s+"
        r"(?:match|partner|pairing)(?:\s+for)?\s+(.+?)\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    if explicit_top or possessive_top or name_first_top or best_for or global_best:
        return IntentDecision(category="ranking", confidence=0.9, reason="ranking language")

    if _is_vague_best_match_query(query_text):
        return IntentDecision(
            category="ranking",
            confidence=0.65,
            names=(memory["last_panda_name"],) if "last_panda_name" in memory else (),
            needs_clarification="last_panda_name" not in memory,
            reason="vague ranking request",
        )

    if re.match(r"^(?:explain|why)\b", lower) or "blocker" in lower or "breakdown" in lower:
        return IntentDecision(
            category="explanation",
            confidence=0.8,
            reason="why/explain language",
        )

    if re.search(r"\b(health|healthy|condition|alive|dead|deceased|status)\b", lower):
        return IntentDecision(
            category="health_status",
            confidence=0.85,
            reason="health/status term",
        )

    if re.search(r"\b(eligible|available|still be matched|can still be matched)\b", lower):
        return IntentDecision(category="eligibility", confidence=0.85, reason="eligibility term")

    if extract_name_from_question(message):
        return IntentDecision(category="profile", confidence=0.8, reason="profile info pattern")

    return IntentDecision(category="fallback", confidence=0.2, reason="no matching intent pattern")


def _record_llm_fallback(reason: str, phase: str) -> None:
    with start_span(
        "llm_fallback",
        span_type=SpanType.TOOL,
        attributes={
            "llm_fallback_reason": reason,
            "llm_fallback_phase": phase,
        },
    ):
        pass


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


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _format_rank_score(row: dict[str, Any]) -> str:
    rank = _rank_number(row)
    score = _score_number(row)
    parts: list[str] = []
    if rank is not None:
        parts.append(f"#{rank}")
    if score is not None:
        parts.append(f"score {score:.3f}")
    return ", ".join(parts)


def _wants_numeric_detail(message: str) -> bool:
    lowered = message.lower()
    detail_markers = (
        "detail",
        "detailed",
        "breakdown",
        "break down",
        "score",
        "scores",
        "component",
        "components",
        "numbers",
        "numeric",
        "exactly why",
    )
    return any(marker in lowered for marker in detail_markers)


def _qualitative_band(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 0.8:
        return "very strong"
    if value >= 0.65:
        return "strong"
    if value >= 0.5:
        return "fairly solid"
    if value >= 0.35:
        return "mixed"
    return "weak"


def _component_takeaway(row: dict[str, Any]) -> str | None:
    bio = _qualitative_band(_maybe_float(row.get("bio_component")))
    behavior = _qualitative_band(_maybe_float(row.get("behavior_component")))
    logistics = _qualitative_band(_maybe_float(row.get("logistics_component")))

    parts: list[str] = []
    if bio:
        parts.append(f"the biological fit looks {bio}")
    if behavior:
        parts.append(f"the behavior fit looks {behavior}")
    if logistics:
        parts.append(f"logistically the pairing looks {logistics}")

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}, and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _natural_strengths(row: dict[str, Any]) -> list[str]:
    strengths: list[str] = []

    positive = _clean_factor(row.get("top_positive_factors"))
    if positive:
        strengths.append(f"the clearest strength is {positive}")

    age_gap_years = _maybe_float(row.get("age_gap_years"))
    focal_age = _maybe_float(row.get("focal_panda_age"))
    candidate_age = _maybe_float(row.get("candidate_panda_age"))
    if age_gap_years is not None:
        strengths.append(f"the age gap is a manageable {age_gap_years:.0f} years")
    if (
        row.get("candidate_panda_sex") == "male"
        and focal_age is not None
        and candidate_age is not None
        and candidate_age > focal_age
        and int(row.get("male_older_bonus") or 0) > 0
    ):
        strengths.append(
            "the older-male / younger-female setup is also working in the pair's favor"
        )

    candidate_babies = row.get("candidate_panda_babies")
    zero_babies_bonus = int(row.get("zero_babies_bonus") or 0)
    if candidate_babies == 0 and zero_babies_bonus > 0:
        candidate_name = str(row.get("candidate_panda_name") or "the candidate")
        strengths.append(
            f"{candidate_name} has no recorded cubs yet, which can make the pairing "
            "more promising than one involving a male that has already reproduced"
        )

    health_penalty = _maybe_float(row.get("health_penalty_score"))
    if health_penalty is not None and health_penalty <= 0.1:
        strengths.append("there are very few health concerns weighing the pair down")

    component_takeaway = _component_takeaway(row)
    if component_takeaway:
        strengths.append(
            component_takeaway.replace("the biological fit looks", "biologically the pair looks")
            .replace("the behavior fit looks", "behaviorally the pair looks")
            .replace("logistically the pairing looks", "logistically the pair looks")
        )

    return strengths


def _natural_tradeoff(row: dict[str, Any]) -> str | None:
    negative = _clean_factor(row.get("top_negative_factors"))
    if negative and negative.lower() != "no major penalties":
        return (
            f"The softer spot is {negative}, so the pair looks less convincing "
            "on personality than it does on health and breeding factors."
        )

    return None


def _format_section(title: str, items: list[str]) -> str:
    cleaned = [item.strip().rstrip(".") + "." for item in items if item and item.strip()]
    if not cleaned:
        return ""
    return title + "\n" + "\n".join(f"- {item}" for item in cleaned)


def _join_natural(items: list[str]) -> str:
    cleaned = [item for item in items if item]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _profile_overview_response(profile: dict[str, Any]) -> str:
    name = str(profile.get("name") or "This panda").strip()
    sex = str(profile.get("sex") or "").strip().lower()
    status = str(profile.get("status") or "").strip().lower()
    age = profile.get("age_years")
    babies = profile.get("babies_had_count")
    location = _join_natural(
        [
            str(profile.get("zoo_or_facility") or "").strip(),
            str(profile.get("city_region") or "").strip(),
            str(profile.get("country") or "").strip(),
        ]
    )
    description = str(profile.get("description_text") or "").strip()
    personality = str(profile.get("personality_text") or "").strip()
    health = str(profile.get("health_text") or "").strip()

    identity_bits: list[str] = []
    if age not in (None, ""):
        identity_bits.append(f"{age}-year-old")
    if sex:
        identity_bits.append(sex)
    identity = " ".join(identity_bits) if identity_bits else "profiled"

    intro = f"{name} is a {identity} panda"
    if status:
        intro += f" listed as {status}"
    if location:
        intro += f" at {location}"
    intro += "."

    details: list[str] = []
    if babies not in (None, ""):
        details.append(f"The profile records {babies} cubs")
    if personality:
        details.append(f"personality notes include {personality}")
    if health:
        details.append(f"health context says {health}")

    sections = [intro]
    if description:
        sections.append(description)
    if details:
        sections.append(_join_natural(details).rstrip(".") + ".")
    return "\n\n".join(section for section in sections if section).strip()


def _pair_not_ranked_response(
    *,
    focal_name: str,
    candidate_name: str,
    focal_profile: dict[str, Any] | None,
    candidate_profile: dict[str, Any] | None,
) -> str:
    if not focal_profile or not candidate_profile:
        missing = focal_name if not focal_profile else candidate_name
        return (
            f"I could not find a complete profile for {missing}, so I cannot give a "
            "grounded pair assessment yet."
        )

    focal_sex = str(focal_profile.get("sex") or "").strip().lower()
    candidate_sex = str(candidate_profile.get("sex") or "").strip().lower()
    focal_status = str(focal_profile.get("status") or "").strip().lower()
    candidate_status = str(candidate_profile.get("status") or "").strip().lower()

    blockers: list[str] = []
    if focal_sex and candidate_sex and focal_sex == candidate_sex:
        blockers.append(f"both pandas are recorded as {focal_sex}")
    if focal_status and focal_status != "alive":
        blockers.append(f"{focal_profile.get('name') or focal_name} is recorded as {focal_status}")
    if candidate_status and candidate_status != "alive":
        blockers.append(
            f"{candidate_profile.get('name') or candidate_name} is recorded as {candidate_status}"
        )

    if blockers:
        return (
            f"{candidate_profile.get('name') or candidate_name} is not currently showing up as "
            f"a ranked breeding match for {focal_profile.get('name') or focal_name}. "
            "The likely blocker is that "
            + "; ".join(blockers)
            + "."
        )

    return (
        f"I found profiles for {focal_profile.get('name') or focal_name} and "
        f"{candidate_profile.get('name') or candidate_name}, but this pair is not present "
        "in the ranked match output. That means the pair likely dropped out during "
        "eligibility, candidate-pair generation, or scoring."
    )


def _ranked_pair_turn(
    *,
    session: Session,
    message: str,
    memory: dict[str, str],
    focal_name: str,
    candidate_name: str,
    explanation: bool = False,
) -> AgentTurn:
    resolved_focal = find_panda_name_by_substring(session, focal_name) or focal_name
    result = top_matches_data(session, panda_name=resolved_focal, k=20)
    matches = result["matches"]
    candidate_row = _find_candidate_row(matches, candidate_name)
    top_row = matches[0] if matches else None

    if not candidate_row:
        resolved_candidate_as_focal = (
            find_panda_name_by_substring(session, candidate_name) or candidate_name
        )
        reverse_result = top_matches_data(session, panda_name=resolved_candidate_as_focal, k=20)
        reverse_matches = reverse_result["matches"]
        reverse_candidate_row = _find_candidate_row(reverse_matches, resolved_focal)
        if reverse_candidate_row:
            resolved_focal = resolved_candidate_as_focal
            candidate_name = focal_name
            result = reverse_result
            matches = reverse_matches
            candidate_row = reverse_candidate_row
            top_row = matches[0] if matches else None

    if not candidate_row:
        memory["last_panda_name"] = resolved_focal
        memory["last_candidate_name"] = candidate_name
        focal_profile = panda_profile_data(session, panda_name=resolved_focal)
        candidate_profile = panda_profile_data(session, panda_name=candidate_name)
        return AgentTurn(
            intent="pairwise_match_not_found",
            response=_pair_not_ranked_response(
                focal_name=resolved_focal,
                candidate_name=candidate_name,
                focal_profile=focal_profile,
                candidate_profile=candidate_profile,
            ),
            data={
                "ranking_result": result,
                "focal_profile": focal_profile,
                "candidate_profile": candidate_profile,
            },
        )

    memory["last_panda_name"] = resolved_focal
    memory["last_candidate_name"] = str(candidate_row.get("candidate_panda_name") or candidate_name)
    if explanation:
        return AgentTurn(
            intent="match_explanation",
            response=_deterministic_rank_explanation(
                focal_name=resolved_focal,
                candidate_name=str(candidate_row.get("candidate_panda_name") or candidate_name),
                candidate_row=candidate_row,
                top_row=top_row,
                detailed=_wants_numeric_detail(message),
            ),
            data={
                "panda_name": resolved_focal,
                "candidate_name": candidate_name,
                "match": candidate_row,
                "matches": matches,
            },
        )

    return AgentTurn(
        intent="pairwise_match_opinion",
        response=_deterministic_pair_opinion_response(
            focal_name=resolved_focal,
            candidate_name=str(candidate_row.get("candidate_panda_name") or candidate_name),
            candidate_row=candidate_row,
            top_row=top_row,
        ),
        data={
            "panda_name": resolved_focal,
            "candidate_name": candidate_name,
            "match": candidate_row,
            "matches": matches,
        },
    )


def _is_vague_best_match_query(message: str) -> bool:
    lower = message.strip().lower()
    if not re.search(r"\b(match|matches|partner|breed|breeding|pair|pairing)\b", lower):
        return False
    return bool(
        re.search(
            r"\b(best|top|ideal|good|compatible|recommend|strongest|suitable)\b",
            lower,
        )
    )


def _resolve_contextual_panda_name(
    session: Session,
    raw_name: str,
    memory: dict[str, str],
) -> str:
    name = raw_name.strip(" ?.")
    if name.lower() in {"he", "she", "him", "her", "it", "they", "them"}:
        return memory.get("last_panda_name") or name
    return find_panda_name_by_substring(session, name) or name


def _remember_query_kind(memory: dict[str, str], kind: str) -> None:
    memory["last_query_kind"] = kind


def _relative_gap_reasons(
    candidate_row: dict[str, Any],
    top_row: dict[str, Any] | None,
) -> list[str]:
    if not top_row or top_row is candidate_row:
        return []

    reasons: list[str] = []
    top_name = str(top_row.get("candidate_panda_name") or "the first-ranked match")

    candidate_age_gap = _maybe_float(candidate_row.get("age_gap_years"))
    top_age_gap = _maybe_float(top_row.get("age_gap_years"))
    if (
        candidate_age_gap is not None
        and top_age_gap is not None
        and candidate_age_gap - top_age_gap >= 2
    ):
        reasons.append(f"{top_name} has the tighter age gap")

    candidate_bio = _maybe_float(candidate_row.get("bio_component"))
    top_bio = _maybe_float(top_row.get("bio_component"))
    if candidate_bio is not None and top_bio is not None and top_bio - candidate_bio >= 0.04:
        reasons.append(f"{top_name} also looks a bit stronger biologically")

    candidate_behavior = _maybe_float(candidate_row.get("behavior_component"))
    top_behavior = _maybe_float(top_row.get("behavior_component"))
    if (
        candidate_behavior is not None
        and top_behavior is not None
        and top_behavior - candidate_behavior >= 0.04
    ):
        reasons.append(f"{top_name} appears to have the better behavioral fit")

    candidate_health_penalty = _maybe_float(candidate_row.get("health_penalty_score"))
    top_health_penalty = _maybe_float(top_row.get("health_penalty_score"))
    if (
        candidate_health_penalty is not None
        and top_health_penalty is not None
        and candidate_health_penalty - top_health_penalty >= 0.02
    ):
        reasons.append(f"{top_name} carries slightly fewer health concerns")

    return reasons


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
    detailed: bool = False,
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
    strengths = _natural_strengths(lead)
    tradeoff = _natural_tradeoff(lead)

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

    intro_lines = [intro]
    if strengths:
        intro_lines.append(f"{lead_name} leads the list mainly because {strengths[0]}.")

    followup: list[str] = []
    if len(selected) > 1:
        runner_up = selected[1]
        runner_name = str(runner_up.get("candidate_panda_name") or "the next candidate").strip()
        followup.append(
            f"{runner_name} is right behind, so the top of the list is fairly tight."
        )

    if len(strengths) > 1:
        followup.extend(strengths[1:])

    sections = ["\n".join(intro_lines)]
    why_section = _format_section("Why the first match is leading:", followup[:4])
    if why_section:
        sections.append(why_section)
    if tradeoff:
        tradeoff_section = _format_section("Main hesitation:", [tradeoff])
        if tradeoff_section:
            sections.append(tradeoff_section)

    return "\n\n".join(section for section in sections if section).strip()


def _deterministic_best_overall_response(row: dict[str, Any]) -> str:
    focal_name = str(row.get("focal_panda_name") or row.get("name") or "the focal panda").strip()
    candidate_name = str(row.get("candidate_panda_name") or "the candidate panda").strip()
    rank_score = _format_rank_score(row)
    strengths = _natural_strengths(row)
    tradeoff = _natural_tradeoff(row)

    intro = (
        f"The strongest overall directional match I found is {focal_name} with "
        f"{candidate_name}."
    )
    if rank_score:
        intro += f" It is currently ranked with {rank_score}."

    sections = [intro]
    if strengths:
        sections.append(_format_section("Why this pair stands out:", strengths[:5]))
    if tradeoff:
        sections.append(_format_section("Main hesitation:", [tradeoff]))
    return "\n\n".join(section for section in sections if section).strip()


def _deterministic_pairwise_response(
    *,
    focal_name: str,
    result: dict[str, Any],
    detailed: bool = False,
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
        rank = _rank_number(row)
        strengths = _natural_strengths(row)
        tradeoff = _natural_tradeoff(row)
        sentence = name
        if rank is not None:
            sentence += f" sits at #{rank}"
        if strengths:
            strength_text = strengths[0].removeprefix("the clearest strength is ")
            sentence += f", and its clearest strength is {strength_text}"
        sentence += "."
        if len(strengths) > 1:
            sentence += " It also helps that " + "; ".join(strengths[1:]) + "."
        if tradeoff:
            sentence += f" {tradeoff}"
        return sentence

    better_rank = _rank_number(better)
    other = candidate_b if better is candidate_a else candidate_a
    other_name = (
        str(other.get("candidate_panda_name") or "").strip()
        if other is not None
        else "the other candidate"
    )

    intro = (
        f"Between {result['candidate_a_name']} and {result['candidate_b_name']}, "
        f"{better_name} looks like the stronger match for {focal_name} right now."
    )
    comparison = ""
    if better_rank is not None:
        comparison = f"{better_name} is currently ahead in the ranking."
        comparison += f" The margin over {other_name} is noticeable, but not huge."

    better_summary = _summary(better)
    other_summary = _summary(other)
    sections = [intro]
    if comparison:
        sections.append(comparison)
    if better_summary:
        sections.append(_format_section(f"Why {better_name} is ahead:", [better_summary]))
    if other_summary:
        sections.append(_format_section(f"How {other_name} compares:", [other_summary]))
    return "\n\n".join(section for section in sections if section)


def _deterministic_pair_opinion_response(
    *,
    focal_name: str,
    candidate_name: str,
    candidate_row: dict[str, Any],
    top_row: dict[str, Any] | None,
) -> str:
    rank = _rank_number(candidate_row)
    strengths = _natural_strengths(candidate_row)
    tradeoff = _natural_tradeoff(candidate_row)
    gap_reasons = _relative_gap_reasons(candidate_row, top_row)

    if rank == 1:
        intro = f"Yes — {candidate_name} looks like an excellent match for {focal_name}."
    elif rank is not None and rank <= 3:
        intro = (
            f"{candidate_name} does look like a strong match for {focal_name}, "
            "just not the very strongest one in the current ranking."
        )
    elif rank is not None and rank <= 5:
        intro = (
            f"{candidate_name} looks like a reasonable match for {focal_name}, "
            "but there are stronger options ahead of it."
        )
    else:
        intro = (
            f"{candidate_name} does not look like one of the strongest matches for "
            f"{focal_name} in the current data."
        )

    sections = [intro]
    if strengths:
        sections.append(_format_section("Why it could work:", strengths[:5]))
    why_not_items: list[str] = []
    if gap_reasons:
        why_not_items.extend(gap_reasons[:3])
    elif rank is not None and rank > 1:
        why_not_items.append("there are stronger-ranked options ahead of this pairing")
    if tradeoff:
        why_not_items.append(tradeoff)
    if why_not_items:
        sections.append(_format_section("Why it may not be the best option:", why_not_items))
    return "\n\n".join(section for section in sections if section)


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
    detailed: bool = False,
) -> str:
    rank = _rank_number(candidate_row)
    strengths = _natural_strengths(candidate_row)
    tradeoff = _natural_tradeoff(candidate_row)
    if rank == 1:
        intro = f"{candidate_name} comes out as {focal_name}'s top match right now."
    elif rank is not None:
        intro = f"{candidate_name} currently sits at #{rank} for {focal_name}."
    else:
        intro = f"{candidate_name} is one of the top-ranked matches for {focal_name}."

    comparison: str | None = None
    if top_row is not None and top_row is not candidate_row:
        top_name = str(top_row.get("candidate_panda_name") or "the top candidate").strip()
        gap_reasons = _relative_gap_reasons(candidate_row, top_row)
        if gap_reasons:
            comparison = (
                f"It sits just behind {top_name}. The main difference is that "
                + "; ".join(gap_reasons[:2])
                + "."
            )
        else:
            comparison = f"It sits just behind {top_name}, and the gap is fairly small."

    if not strengths and not comparison and not tradeoff:
        return intro

    sections = [intro]
    if strengths:
        sections.append(_format_section("Main reasons:", strengths[:5]))
    if comparison:
        sections.append(_format_section("Ranking context:", [comparison]))
    if tradeoff:
        sections.append(_format_section("Main hesitation:", [tradeoff]))
    return "\n\n".join(section for section in sections if section)


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
    except LLMRateLimitError:
        _record_llm_fallback("429", "planning")
        logger.exception(
            "LLM planning failed due to rate limit; "
            "falling back to deterministic chat path"
        )
        return None
    except Exception:
        _record_llm_fallback("planner_error", "planning")
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
    except LLMRateLimitError:
        _record_llm_fallback("429", f"tool:{tool}")
        logger.exception(
            "LLM tool execution failed due to rate limit; "
            "falling back to deterministic chat path"
        )
        return None
    except HTTPException:
        _record_llm_fallback("tool_http_error", f"tool:{tool}")
        logger.exception("LLM tool execution failed; falling back to deterministic chat path")
        return None
    except Exception:
        _record_llm_fallback("tool_error", f"tool:{tool}")
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
        r"^(?:who is|tell me(?:\s+more)? about|profile of|details on)\s+(.+?)(?:[.?!].*)?$",
        r"^(?:tell me\s+a\s+fun\s+fact\s+about)\s+(.+)$",
        r"^(?:how old is|where is|health of|personality of)\s+(.+)$",
        r"^(?:what is the health of|what is the personality of)\s+(.+)$",
        r"^is\s+(.+?)\s+(?:dead\s+or\s+alive|alive\s+or\s+dead)\??$",
        r"^is\s+(.+?)\s+(?:healthy|in good health(?: conditions)?)\??$",
        r"^is\s+(.+?)\s+(?:alive|dead|deceased)\??$",
        r"^does\s+(.+?)\s+have\s+any\s+(?:cubs|babies)\??$",
        r"^how many\s+(?:cubs|babies)\s+does\s+(.+?)\s+have\??$",
        r"^(?:status of|condition of)\s+(.+)$",
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

    youngest_pat = re.match(
        r"^who\s+is\s+(?:the\s+)?youngest\s+panda(?:\s+in\s+the\s+dataset)?\??$",
        lower,
    )
    if youngest_pat:
        row = session.execute(
            text(
                """
                SELECT name, age_years
                FROM core.panda_profiles
                WHERE age_years IS NOT NULL
                ORDER BY age_years ASC, name ASC
                LIMIT 1
                """
            )
        ).mappings().first()
        if not row:
            return AgentTurn(
                intent="analytics_youngest_panda",
                response="I could not find any pandas with recorded ages.",
                data={"name": None, "age_years": None},
            )
        return AgentTurn(
            intent="analytics_youngest_panda",
            response=(
                f"The youngest panda in core.panda_profiles is {row['name']}, "
                f"at approximately {row['age_years']} years old."
            ),
            data={"name": row["name"], "age_years": row["age_years"]},
        )

    oldest_pat = re.match(
        r"^who\s+is\s+(?:the\s+)?oldest\s+panda(?:\s+in\s+the\s+dataset)?\??$",
        lower,
    )
    if oldest_pat:
        row = session.execute(
            text(
                """
                SELECT name, age_years
                FROM core.panda_profiles
                WHERE age_years IS NOT NULL
                ORDER BY age_years DESC, name ASC
                LIMIT 1
                """
            )
        ).mappings().first()
        if not row:
            return AgentTurn(
                intent="analytics_oldest_panda",
                response="I could not find any pandas with recorded ages.",
                data={"name": None, "age_years": None},
            )
        return AgentTurn(
            intent="analytics_oldest_panda",
            response=(
                f"The oldest panda in core.panda_profiles is {row['name']}, "
                f"at approximately {row['age_years']} years old."
            ),
            data={"name": row["name"], "age_years": row["age_years"]},
        )

    most_cubs_pat = re.match(
        r"^who\s+is\s+(?:the\s+)?panda\s+with\s+the\s+most\s+(?:cubs|babies)\??$",
        lower,
    )
    if most_cubs_pat:
        row = session.execute(
            text(
                """
                SELECT name, babies_had_count
                FROM core.panda_profiles
                WHERE babies_had_count IS NOT NULL
                ORDER BY babies_had_count DESC, name ASC
                LIMIT 1
                """
            )
        ).mappings().first()
        if not row:
            return AgentTurn(
                intent="analytics_most_cubs",
                response="I could not find any pandas with recorded cub counts.",
                data={"name": None, "babies_had_count": None},
            )
        babies = int(row["babies_had_count"])
        cub_word = "cub" if babies == 1 else "cubs"
        return AgentTurn(
            intent="analytics_most_cubs",
            response=(
                f"The panda with the most recorded cubs is {row['name']}, "
                f"with {babies} {cub_word}."
            ),
            data={"name": row["name"], "babies_had_count": babies},
        )

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

    location_count_pat = re.match(
        r"^(?:how many|count)\s+pandas\s+(?:are\s+)?(?:in|from|located\s+in)\s+(.+?)\??$",
        lower,
    )
    location_any_pat = re.match(
        r"^are\s+there\s+any\s+pandas\s+(?:in|from|located\s+in)\s+(.+?)\??$",
        lower,
    )
    if location_count_pat:
        query_location, display_location, requested_location = _normalize_location_hint(
            location_count_pat.group(1),
        )
        total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(country, '')) = lower(:location)
               OR lower(coalesce(city_region, '')) = lower(:location)
               OR lower(coalesce(zoo_or_facility, '')) LIKE lower(:location_like)
            """,
            {
                "location": query_location,
                "location_like": f"%{query_location}%",
            },
        )
        return AgentTurn(
            intent="analytics_count_location",
            response=f"There are {total} pandas listed in {display_location}.",
            data={
                "requested_location": requested_location,
                "normalized_location": query_location,
                "display_location": display_location,
                "count": total,
            },
        )
    if location_any_pat:
        query_location, display_location, requested_location = _normalize_location_hint(
            location_any_pat.group(1),
        )
        total = scalar_int(
            session,
            """
            SELECT COUNT(*) AS n
            FROM core.panda_profiles
            WHERE lower(coalesce(country, '')) = lower(:location)
               OR lower(coalesce(city_region, '')) = lower(:location)
               OR lower(coalesce(zoo_or_facility, '')) LIKE lower(:location_like)
            """,
            {
                "location": query_location,
                "location_like": f"%{query_location}%",
            },
        )
        if total == 0:
            response = f"No, I do not see any pandas listed in {display_location}."
        elif total == 1:
            response = f"Yes, there is 1 panda listed in {display_location}."
        else:
            response = f"Yes, there are {total} pandas listed in {display_location}."
        return AgentTurn(
            intent="analytics_count_location",
            response=response,
            data={
                "requested_location": requested_location,
                "normalized_location": query_location,
                "display_location": display_location,
                "count": total,
            },
        )

    available_pat = re.match(
        r"^(?:who|which pandas?)\s+(?:can|could)\s+(?:still\s+)?"
        r"(?:be\s+)?(?:matched|paired|bred)\??$",
        lower,
    )
    if available_pat:
        relation = pick_relation(session, "core", ["breedeable_pandas", "breedable_pandas"])
        total = scalar_int(session, f"SELECT COUNT(*) AS n FROM core.{relation}")
        sample_rows = session.execute(
            text(
                f"""
                SELECT name
                FROM core.{relation}
                ORDER BY name
                LIMIT 10
                """
            )
        ).mappings().all()
        names = [str(row["name"]) for row in sample_rows]
        response = f"{total} pandas are currently in core.{relation}."
        if names:
            response += " Examples include " + _join_natural(names) + "."
        return AgentTurn(
            intent="eligibility_available_pandas",
            response=response,
            data={"relation": relation, "count": total, "sample_names": names},
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
    query_text = re.split(r"[.?!]\s+", message.strip(), maxsplit=1)[0].strip()
    intent_decision = classify_chat_intent(message, memory)

    explain_match_pattern = re.match(r"^(?:explain|why)\s+(\S+)\s+(\S+)$", query_text.lower())
    explain_pair_pattern = re.match(
        r"^(?:explain|why)\s+(.+?)\s+and\s+(.+?)(?:\s+(?:match|compatible|work|not work).*)?\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    explain_context_pair_pattern = re.match(
        r"^why\s+(?:are|is|would|could)\s+(?:they|that|this)\s+"
        r"(?:a\s+)?(?:good|great|bad|weak|strong|compatible)?\s*"
        r"(?:match|pair|pairing|compatible)\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    top_pattern = re.search(
        r"(?:top|best)\s*(\d+)?\s*matches(?:\s+for)?\s+(.+)$",
        query_text,
        flags=re.IGNORECASE,
    )
    possessive_top_pattern = re.match(
        r"^(?:who\s+are\s+)?(.+?)'s\s+(?:top|best)\s*(\d+)?\s+matches\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    name_first_top_pattern = re.match(
        r"^(?:(?:who\s+are|whoa\s+re|who\s+rae|show\s+me|give\s+me)\s+)?"
        r"(.+?)\s+(?:top|best)\s*(\d+)?\s+matches\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    what_about_pattern = re.match(
        r"^(?:what|how)\s+about\s+(.+?)\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    first_match_explanation_pattern = re.match(
        r"^why\s+is\s+(.+)\s+(.+?)'s\s+first\s+top\s+match\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    second_match_explanation_pattern = re.match(
        r"^why\s+is\s+(.+?)\s+only\s+second\s+for\s+(.+?)\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    no_matches_pattern = re.match(
        r"^why\s+does\s+(.+?)\s+have\s+no\s+matches\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    best_for_pattern = re.match(
        r"^(?:who\s+is\s+|what\s+is\s+|show\s+me\s+|find\s+)?(?:the\s+)?"
        r"(?:best|top|ideal|strongest|most\s+compatible)\s+"
        r"(?:match|partner|pairing)(?:\s+for)?\s+(.+?)\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    breeding_partner_pattern = re.match(
        r"^(?:who\s+should|who\s+could|who\s+would)\s+(.+?)\s+"
        r"(?:breed|mate|pair)(?:\s+with)?\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    great_match_pattern = re.match(
        r"^do\s+you\s+think\s+(.+?)\s+and\s+(.+?)\s+would\s+make\s+a\s+great\s+match(?:,?\s*why(?:\s+and\s+why\s+not)?)?\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    broad_pair_match_pattern = re.match(
        r"^(?:would|could|should|is|are)\s+(.+?)\s+and\s+(.+?)\s+"
        r"(?:make\s+)?(?:a\s+)?(?:good|great|strong|compatible|suitable)?\s*"
        r"(?:match|pair|pairing|breeding\s+pair|compatible)(?:,?\s*why(?:\s+and\s+why\s+not)?)?\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    pairwise_pattern = re.match(
        r"^is\s+(.+?)\s+or\s+(.+?)\s+better\s+for\s+(.+?)(?:,?\s*and why)?\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    global_best_pattern = re.search(
        (
            r"(?:best overall|overall best|best match across all|across all pandas|"
            r"global best|most eligible panda|most eligible match|best panda match|"
            r"best overall pair|recommend(?:\s+me)?\s+a\s+panda match|"
            r"tell me about a panda match)"
        ),
        lower,
    )
    blockers_pattern = re.match(r"^blockers(?:\s+for)?\s+(.+)$", query_text, flags=re.IGNORECASE)
    vague_best_match = (
        _is_vague_best_match_query(query_text)
        and not (
            first_match_explanation_pattern
            or second_match_explanation_pattern
            or top_pattern
            or possessive_top_pattern
            or name_first_top_pattern
            or best_for_pattern
            or breeding_partner_pattern
            or explain_context_pair_pattern
            or explain_pair_pattern
            or great_match_pattern
            or broad_pair_match_pattern
            or pairwise_pattern
        )
    )
    profile_name = extract_name_from_question(message)
    asks_age = "how old" in lower or "age of" in lower
    asks_location = "where is" in lower or "location of" in lower
    asks_health = "health" in lower
    asks_status = any(marker in lower for marker in ("alive", "dead", "deceased", "status"))
    asks_personality = "personality" in lower
    asks_fun_fact = "fun fact" in lower
    asks_cubs = bool(
        re.search(r"\b(cubs|babies)\b", lower)
        and re.search(r"\b(have|has|how many|any)\b", lower)
    )

    unsupported_response = unsupported_chat_response(message)
    if unsupported_response is not None:
        return unsupported_response

    analytics_response = analytics_chat_response(session=session, message=message, memory=memory)
    if analytics_response is not None:
        return analytics_response

    if intent_decision.needs_clarification:
        return AgentTurn(
            intent="clarification_needed",
            response=(
                "Which panda should I use for that match question? You can ask for a "
                "specific panda or say 'best overall pair'."
            ),
            data={"intent": intent_decision.__dict__, "memory": memory},
        )

    if global_best_pattern:
        result = best_overall_match_data(session, k=1)
        if result["count"] == 0:
            return AgentTurn(
                intent="best_overall_empty",
                response="I could not find any ranked matches in the current scoring view.",
                data=result,
            )
        top = result["matches"][0]
        memory["last_panda_name"] = str(top.get("focal_panda_name") or "")
        memory["last_candidate_name"] = str(top.get("candidate_panda_name") or "")
        _remember_query_kind(memory, "best_overall")
        return AgentTurn(
            intent="best_overall",
            response=_deterministic_best_overall_response(top),
            data={"best_match": top, "ranking_meta": result},
        )

    if best_for_pattern:
        panda_name = best_for_pattern.group(1).strip(" ?.")
        resolved_name = _resolve_contextual_panda_name(session, panda_name, memory)
        result = top_matches_data(session, panda_name=resolved_name, k=1)
        memory["last_panda_name"] = resolved_name
        _remember_query_kind(memory, "top_matches")
        if result["count"] == 0:
            return AgentTurn(
                intent="top_matches_no_results",
                response=_deterministic_no_match_response(
                    diagnosis=diagnose_no_matches(session, panda_name=resolved_name),
                    curated=curated_override_for_name(session, panda_name=resolved_name),
                ),
                data=result,
            )
        if result["matches"]:
            memory["last_candidate_name"] = str(
                result["matches"][0].get("candidate_panda_name") or ""
            )
        return AgentTurn(
            intent="top_matches",
            response=_deterministic_top_matches_response(
                focal_name=resolved_name,
                matches=result["matches"],
                requested_k=1,
                detailed=_wants_numeric_detail(message),
            ),
            data=result,
        )

    if breeding_partner_pattern:
        panda_name = breeding_partner_pattern.group(1).strip(" ?.")
        resolved_name = _resolve_contextual_panda_name(session, panda_name, memory)
        result = top_matches_data(session, panda_name=resolved_name, k=1)
        memory["last_panda_name"] = resolved_name
        _remember_query_kind(memory, "top_matches")
        if result["count"] == 0:
            return AgentTurn(
                intent="top_matches_no_results",
                response=_deterministic_no_match_response(
                    diagnosis=diagnose_no_matches(session, panda_name=resolved_name),
                    curated=curated_override_for_name(session, panda_name=resolved_name),
                ),
                data=result,
            )
        if result["matches"]:
            memory["last_candidate_name"] = str(
                result["matches"][0].get("candidate_panda_name") or ""
            )
        return AgentTurn(
            intent="top_matches",
            response=_deterministic_top_matches_response(
                focal_name=resolved_name,
                matches=result["matches"],
                requested_k=1,
                detailed=_wants_numeric_detail(message),
            ),
            data=result,
        )

    if vague_best_match and "last_panda_name" in memory:
        panda_name = memory["last_panda_name"]
        result = top_matches_data(session, panda_name=panda_name, k=1)
        _remember_query_kind(memory, "top_matches")
        if result["count"] == 0:
            return AgentTurn(
                intent="top_matches_no_results",
                response=_deterministic_no_match_response(
                    diagnosis=diagnose_no_matches(session, panda_name=panda_name),
                    curated=curated_override_for_name(session, panda_name=panda_name),
                ),
                data=result,
            )
        if result["matches"]:
            memory["last_candidate_name"] = str(
                result["matches"][0].get("candidate_panda_name") or ""
            )
        return AgentTurn(
            intent="top_matches",
            response=_deterministic_top_matches_response(
                focal_name=panda_name,
                matches=result["matches"],
                requested_k=1,
                detailed=_wants_numeric_detail(message),
            ),
            data=result,
        )

    if vague_best_match:
        result = best_overall_match_data(session, k=1)
        if result["count"] == 0:
            return AgentTurn(
                intent="best_overall_empty",
                response="I could not find any ranked matches in the current scoring view.",
                data=result,
            )
        top = result["matches"][0]
        memory["last_panda_name"] = str(top.get("focal_panda_name") or "")
        memory["last_candidate_name"] = str(top.get("candidate_panda_name") or "")
        _remember_query_kind(memory, "best_overall")
        return AgentTurn(
            intent="best_overall",
            response=_deterministic_best_overall_response(top),
            data={"best_match": top, "ranking_meta": result},
        )

    if explain_context_pair_pattern and {
        "last_panda_name",
        "last_candidate_name",
    }.issubset(memory):
        return _ranked_pair_turn(
            session=session,
            message=message,
            memory=memory,
            focal_name=memory["last_panda_name"],
            candidate_name=memory["last_candidate_name"],
            explanation=True,
        )

    if explain_pair_pattern:
        return _ranked_pair_turn(
            session=session,
            message=message,
            memory=memory,
            focal_name=explain_pair_pattern.group(1).strip(" ?."),
            candidate_name=explain_pair_pattern.group(2).strip(" ?."),
            explanation=True,
        )

    explain_subject_pattern = re.match(
        r"^explain\s+(.+?)(?:\s+(?:match|compatibility|pairing))?\??$",
        query_text,
        flags=re.IGNORECASE,
    )
    if explain_subject_pattern:
        resolved_pair = _resolve_subject_pair(session, explain_subject_pattern.group(1))
        if resolved_pair is not None:
            candidate_name, resolved_name = resolved_pair
            return _ranked_pair_turn(
                session=session,
                message=message,
                memory=memory,
                focal_name=resolved_name,
                candidate_name=candidate_name,
                explanation=True,
            )

    if profile_name:
        resolved_name = _resolve_contextual_panda_name(session, profile_name, memory)
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
        elif asks_health or asks_status:
            status_text = str(profile.get("status") or "unknown").strip()
            health_text = str(profile.get("health_text") or "").strip()
            if asks_status and asks_health:
                response = f"{profile['name']} is listed as {status_text}."
                if health_text:
                    response += f" Health notes: {health_text}"
            elif asks_status:
                response = f"{profile['name']} is listed as {status_text}."
            else:
                response = (
                    f"Health notes for {profile['name']}: {health_text}"
                    if health_text
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
        elif asks_cubs:
            babies = profile.get("babies_had_count")
            if babies in (None, ""):
                response = f"I do not have cub records for {profile['name']}."
            elif int(babies) == 0:
                response = f"No, I do not see any cubs recorded for {profile['name']}."
            elif int(babies) == 1:
                response = f"Yes, {profile['name']} has 1 cub recorded."
            else:
                response = f"Yes, {profile['name']} has {babies} cubs recorded."
        else:
            response = _profile_overview_response(profile)

        return AgentTurn(intent="panda_info", response=response, data={"profile": profile})

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
                detailed=_wants_numeric_detail(message),
            ),
            data=result,
        )

    pair_opinion_pattern = great_match_pattern or broad_pair_match_pattern
    if pair_opinion_pattern:
        return _ranked_pair_turn(
            session=session,
            message=message,
            memory=memory,
            focal_name=pair_opinion_pattern.group(1).strip(" ?."),
            candidate_name=pair_opinion_pattern.group(2).strip(" ?."),
            explanation=False,
        )

    if what_about_pattern and memory.get("last_query_kind") == "top_matches":
        panda_name = what_about_pattern.group(1).strip(" ?.")
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        result = top_matches_data(session, panda_name=resolved_name, k=5)
        memory["last_panda_name"] = resolved_name
        _remember_query_kind(memory, "top_matches")
        if result["count"] == 0:
            diagnosis = diagnose_no_matches(session, panda_name=resolved_name)
            curated = curated_override_for_name(session, panda_name=resolved_name)
            return AgentTurn(
                intent="top_matches_no_results",
                response=_deterministic_no_match_response(
                    diagnosis=diagnosis,
                    curated=curated,
                ),
                data={
                    "requested_top_k": 5,
                    "panda_name": resolved_name,
                    "matches": [],
                    "diagnosis": diagnosis,
                    "curated_profile": curated,
                },
            )
        return AgentTurn(
            intent="top_matches",
            response=_deterministic_top_matches_response(
                focal_name=resolved_name,
                matches=result["matches"],
                requested_k=5,
                detailed=_wants_numeric_detail(message),
            ),
            data=result,
        )

    if top_pattern:
        k_raw = top_pattern.group(1)
        panda_name = _clean_panda_name_hint(top_pattern.group(2))
        k = int(k_raw) if k_raw else 5
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        result = top_matches_data(session, panda_name=resolved_name, k=k)
        memory["last_panda_name"] = resolved_name
        _remember_query_kind(memory, "top_matches")
        if result["count"] == 0:
            diagnosis = diagnose_no_matches(session, panda_name=resolved_name)
            curated = curated_override_for_name(session, panda_name=resolved_name)
            return AgentTurn(
                intent="top_matches_no_results",
                response=_deterministic_no_match_response(
                    diagnosis=diagnosis,
                    curated=curated,
                ),
                data={
                    "requested_top_k": k,
                    "panda_name": resolved_name,
                    "matches": [],
                    "diagnosis": diagnosis,
                    "curated_profile": curated,
                },
            )
        return AgentTurn(
            intent="top_matches",
            response=_deterministic_top_matches_response(
                focal_name=resolved_name,
                matches=result["matches"],
                requested_k=k,
                detailed=_wants_numeric_detail(message),
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
        _remember_query_kind(memory, "top_matches")
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
                detailed=_wants_numeric_detail(message),
            ),
            data=result,
        )

    if name_first_top_pattern:
        panda_name = _clean_panda_name_hint(name_first_top_pattern.group(1))
        k_raw = name_first_top_pattern.group(2)
        k = int(k_raw) if k_raw else 5
        resolved_name = find_panda_name_by_substring(session, panda_name) or panda_name
        result = top_matches_data(session, panda_name=resolved_name, k=k)
        memory["last_panda_name"] = resolved_name
        _remember_query_kind(memory, "top_matches")
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
                detailed=_wants_numeric_detail(message),
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
                detailed=_wants_numeric_detail(message),
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
                detailed=_wants_numeric_detail(message),
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
            lead_positive = _clean_factor(lead.get("top_positive_factors"))
            response = (
                f"In the current ranked view, {resolved_name} does have matches. "
                f"Right now {lead_name} is the top match"
            )
            if lead_positive:
                response += f", helped mainly by {lead_positive}"
            response += (
                ". If you were expecting no matches, the mismatch is probably coming "
                "from a different filter, a different table, or an earlier stage of "
                f"the pipeline rather than from {resolved_name} having no ranked options."
            )
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

    llm_response = llm_chat_response(session, message=message, memory=memory)
    if llm_response is not None:
        return llm_response

    if lower in {"help", "commands"}:
        return AgentTurn(
            intent="help",
            response=(
                "Commands: "
                "'top 5 matches for Bao Li' (or 'give me the top 5 matches for Bao Li'), "
                "'best overall panda match', "
                "'tell me about a panda match', "
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
