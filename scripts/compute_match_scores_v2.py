from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any

import psycopg


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return default


def _tag_set(raw_tags: str | None) -> set[str]:
    if not raw_tags:
        return set()
    return {token.strip().lower() for token in str(raw_tags).split(",") if token.strip()}


def _personality_compat_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.5
    overlap = len(left.intersection(right))
    union = len(left.union(right))
    jaccard = overlap / union if union else 0.0
    return _clamp(0.35 + 0.65 * jaccard)


def _logistics_score(focal: dict[str, Any], candidate: dict[str, Any]) -> float:
    focal_zoo = (focal.get("zoo_or_facility") or "").strip().lower()
    cand_zoo = (candidate.get("zoo_or_facility") or "").strip().lower()
    focal_country = (focal.get("country") or "").strip().lower()
    cand_country = (candidate.get("country") or "").strip().lower()

    if focal_zoo and cand_zoo and focal_zoo == cand_zoo:
        base = 0.95
    elif focal_country and cand_country and focal_country == cand_country:
        base = 0.8
    else:
        base = 0.55

    on_loan_focal = focal.get("on_loan")
    on_loan_cand = candidate.get("on_loan")
    if on_loan_focal is True or on_loan_cand is True:
        base -= 0.08
    return _clamp(base)


def _positive_factors(
    *,
    personality_compat: float,
    breeding_success: float,
    logistics: float,
    health_penalty: float,
) -> str:
    factors: list[str] = []
    if personality_compat >= 0.7:
        factors.append("strong personality compatibility")
    if breeding_success >= 0.65:
        factors.append("high past breeding success signal")
    if logistics >= 0.8:
        factors.append("high logistical feasibility")
    if health_penalty <= 0.2:
        factors.append("low combined health risk")
    return "; ".join(factors) if factors else "balanced profile"


def _negative_factors(
    *,
    personality_compat: float,
    aggression_penalty: float,
    health_penalty: float,
    age_gap_score: float,
) -> str:
    factors: list[str] = []
    if health_penalty >= 0.45:
        factors.append("elevated health risk")
    if aggression_penalty >= 0.45:
        factors.append("high aggression risk")
    if age_gap_score <= 0.35:
        factors.append("large age gap")
    if personality_compat <= 0.35:
        factors.append("weak personality overlap")
    return "; ".join(factors) if factors else "no major penalties"


def main() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    db_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT source_id, zoo_or_facility, country, on_loan, status
                FROM core.panda_profiles
                """
            )
            profiles = {str(row["source_id"]): dict(row) for row in cur.fetchall()}

            cur.execute(
                """
                SELECT
                    source_id, personality_tags_norm, health_risk_score,
                    fertility_status_score, aggression_level_score,
                    past_breeding_success_score, availability_window_score,
                    extraction_confidence
                FROM core.panda_text_features
                """
            )
            features = {str(row["source_id"]): dict(row) for row in cur.fetchall()}

            cur.execute("SELECT * FROM core.directional_recommended_matches")
            directional_rows = [dict(row) for row in cur.fetchall()]

        upsert_count = 0
        with conn.cursor() as cur:
            for row in directional_rows:
                focal_id = str(row["focal_panda_id"])
                candidate_id = str(row["candidate_panda_id"])
                base_score = _as_float(row.get("recommendation_score"), default=50.0)
                age_gap = abs(_as_float(row.get("age_gap_years"), default=8.0))
                age_gap_score = _clamp(1.0 - (min(age_gap, 20.0) / 20.0))

                focal_feat = features.get(focal_id, {})
                cand_feat = features.get(candidate_id, {})
                focal_tags = _tag_set(focal_feat.get("personality_tags_norm"))
                cand_tags = _tag_set(cand_feat.get("personality_tags_norm"))

                personality_compat = _personality_compat_score(focal_tags, cand_tags)
                health_penalty = _clamp(
                    (_as_float(focal_feat.get("health_risk_score"), 0.25) +
                     _as_float(cand_feat.get("health_risk_score"), 0.25)) / 2.0
                )
                aggression_penalty = _clamp(
                    (_as_float(focal_feat.get("aggression_level_score"), 0.2) +
                     _as_float(cand_feat.get("aggression_level_score"), 0.2)) / 2.0
                )
                breeding_success = _clamp(
                    (_as_float(focal_feat.get("past_breeding_success_score"), 0.5) +
                     _as_float(cand_feat.get("past_breeding_success_score"), 0.5)) / 2.0
                )
                logistics = _logistics_score(
                    profiles.get(focal_id, {}),
                    profiles.get(candidate_id, {}),
                )
                pair_history = 0.5

                bio_component = _clamp(0.45 * age_gap_score + 0.55 * (1.0 - health_penalty))
                behavior_component = _clamp(
                    0.5 * personality_compat
                    + 0.3 * breeding_success
                    + 0.2 * (1.0 - aggression_penalty)
                )
                logistics_component = logistics
                pair_history_component = pair_history

                composite = (
                    0.35 * bio_component
                    + 0.30 * behavior_component
                    + 0.20 * logistics_component
                    + 0.15 * pair_history_component
                )
                adjustment = (composite - 0.5) * 30.0
                final_score = _clamp((base_score + adjustment) / 100.0, 0.0, 1.0) * 100.0

                breakdown = {
                    "weights": {
                        "bio": 0.35,
                        "behavior": 0.30,
                        "logistics": 0.20,
                        "pair_history": 0.15,
                    },
                    "components": {
                        "bio": bio_component,
                        "behavior": behavior_component,
                        "logistics": logistics_component,
                        "pair_history": pair_history_component,
                    },
                    "sub_features": {
                        "age_gap_score": age_gap_score,
                        "personality_compat_score": personality_compat,
                        "health_penalty_score": health_penalty,
                        "aggression_penalty_score": aggression_penalty,
                        "breeding_success_score": breeding_success,
                    },
                }

                top_pos = _positive_factors(
                    personality_compat=personality_compat,
                    breeding_success=breeding_success,
                    logistics=logistics_component,
                    health_penalty=health_penalty,
                )
                top_neg = _negative_factors(
                    personality_compat=personality_compat,
                    aggression_penalty=aggression_penalty,
                    health_penalty=health_penalty,
                    age_gap_score=age_gap_score,
                )

                cur.execute(
                    """
                    INSERT INTO core.match_scores_v2 (
                        focal_panda_id,
                        candidate_panda_id,
                        base_score,
                        adjustment_score,
                        final_score_v2,
                        bio_component,
                        behavior_component,
                        logistics_component,
                        pair_history_component,
                        age_gap_score,
                        personality_compat_score,
                        health_penalty_score,
                        aggression_penalty_score,
                        breeding_success_score,
                        score_breakdown_json,
                        top_positive_factors,
                        top_negative_factors
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s::jsonb, %s, %s
                    )
                    ON CONFLICT (focal_panda_id, candidate_panda_id)
                    DO UPDATE SET
                        base_score = EXCLUDED.base_score,
                        adjustment_score = EXCLUDED.adjustment_score,
                        final_score_v2 = EXCLUDED.final_score_v2,
                        bio_component = EXCLUDED.bio_component,
                        behavior_component = EXCLUDED.behavior_component,
                        logistics_component = EXCLUDED.logistics_component,
                        pair_history_component = EXCLUDED.pair_history_component,
                        age_gap_score = EXCLUDED.age_gap_score,
                        personality_compat_score = EXCLUDED.personality_compat_score,
                        health_penalty_score = EXCLUDED.health_penalty_score,
                        aggression_penalty_score = EXCLUDED.aggression_penalty_score,
                        breeding_success_score = EXCLUDED.breeding_success_score,
                        score_breakdown_json = EXCLUDED.score_breakdown_json,
                        top_positive_factors = EXCLUDED.top_positive_factors,
                        top_negative_factors = EXCLUDED.top_negative_factors,
                        generated_at = NOW()
                    """,
                    (
                        focal_id,
                        candidate_id,
                        Decimal(str(base_score)),
                        Decimal(str(adjustment)),
                        Decimal(str(final_score)),
                        Decimal(str(bio_component)),
                        Decimal(str(behavior_component)),
                        Decimal(str(logistics_component)),
                        Decimal(str(pair_history_component)),
                        Decimal(str(age_gap_score)),
                        Decimal(str(personality_compat)),
                        Decimal(str(health_penalty)),
                        Decimal(str(aggression_penalty)),
                        Decimal(str(breeding_success)),
                        json.dumps(breakdown),
                        top_pos,
                        top_neg,
                    ),
                )
                upsert_count += 1

            cur.execute(
                """
                CREATE OR REPLACE VIEW core.ranked_directional_recommended_matches_v2 AS
                SELECT
                    d.*,
                    m.base_score,
                    m.adjustment_score,
                    m.final_score_v2,
                    m.bio_component,
                    m.behavior_component,
                    m.logistics_component,
                    m.pair_history_component,
                    m.age_gap_score,
                    m.personality_compat_score,
                    m.health_penalty_score,
                    m.aggression_penalty_score,
                    m.breeding_success_score,
                    m.score_breakdown_json,
                    m.top_positive_factors,
                    m.top_negative_factors,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.focal_panda_id
                        ORDER BY m.final_score_v2 DESC, d.recommendation_score DESC
                    ) AS recommendation_rank_v2
                FROM core.directional_recommended_matches d
                JOIN core.match_scores_v2 m
                  ON d.focal_panda_id = m.focal_panda_id
                 AND d.candidate_panda_id = m.candidate_panda_id
                """
            )

        conn.commit()

    print(f"Computed and upserted {upsert_count} match scores into core.match_scores_v2")


if __name__ == "__main__":
    main()
