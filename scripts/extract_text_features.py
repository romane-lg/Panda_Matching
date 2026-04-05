from __future__ import annotations

import json
import os
import re
from decimal import Decimal
from typing import Any

import psycopg


def _norm_text(*parts: str | None) -> str:
    text = " ".join((part or "").strip() for part in parts if part is not None)
    return re.sub(r"\s+", " ", text).strip().lower()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score_keywords(
    text: str,
    keywords: dict[str, float],
    baseline: float = 0.0,
) -> tuple[float, list[str]]:
    score = baseline
    matched: list[str] = []
    for key, delta in keywords.items():
        if key in text:
            score += delta
            matched.append(key)
    return _clamp(score), matched


def _normalize_personality_tags(raw_tags: str | None, text_blob: str) -> tuple[str, list[str]]:
    seed_tags: list[str] = []
    if raw_tags:
        seed_tags.extend([tag.strip().lower() for tag in raw_tags.split(",") if tag.strip()])

    keyword_to_tag = {
        "calm": "calm",
        "gentle": "gentle",
        "playful": "playful",
        "curious": "curious",
        "social": "social",
        "dominant": "dominant",
        "aggressive": "aggressive",
        "mischievous": "mischievous",
        "reserved": "reserved",
        "easygoing": "easygoing",
        "energetic": "energetic",
        "devoted mother": "devoted-mother",
        "protective mother": "protective-mother",
    }

    inferred: list[str] = []
    for key, tag in keyword_to_tag.items():
        if key in text_blob and tag not in inferred:
            inferred.append(tag)

    merged = []
    for tag in seed_tags + inferred:
        if tag and tag not in merged:
            merged.append(tag)
    return ",".join(merged), merged


def _derive_features(row: dict[str, Any]) -> dict[str, Any]:
    text_blob = _norm_text(
        row.get("description_agent"),
        row.get("life_journey"),
        row.get("breeding_notes"),
        row.get("health_notes_agent"),
    )

    personality_tags_norm, merged_tags = _normalize_personality_tags(
        row.get("personality_tags_agent"),
        text_blob,
    )

    health_keywords = {
        "kidney failure": 0.55,
        "brain tumor": 0.65,
        "seizure": 0.35,
        "ascites": 0.25,
        "hypertension": 0.2,
        "arthritis": 0.2,
        "obesity": 0.2,
        "aging joints": 0.15,
        "deceased": 1.0,
        "euthanized": 1.0,
    }
    health_risk_score, health_hits = _score_keywords(text_blob, health_keywords, baseline=0.05)
    if "healthy" in text_blob and health_risk_score < 0.4:
        health_risk_score = _clamp(health_risk_score - 0.08)

    aggression_keywords = {
        "aggressive": 0.55,
        "dominant": 0.35,
        "bossy": 0.35,
        "mischievous": 0.15,
    }
    aggression_level_score, aggression_hits = _score_keywords(
        text_blob, aggression_keywords, baseline=0.1
    )
    if "gentle" in text_blob or "calm" in text_blob:
        aggression_level_score = _clamp(aggression_level_score - 0.08)

    fertility_keywords = {
        "infertile": -0.45,
        "false pregnancy": -0.3,
        "no cubs": -0.35,
        "natural breeder": 0.35,
        "gave birth": 0.25,
        "breeding program": 0.15,
        "highly reproductive": 0.4,
    }
    fertility_status_score, fertility_hits = _score_keywords(
        text_blob, fertility_keywords, baseline=0.5
    )

    breeding_keywords = {
        "gave birth": 0.35,
        "mother of": 0.25,
        "father of": 0.2,
        "twins": 0.2,
        "natural breeder": 0.4,
        "highly reproductive": 0.45,
        "no cubs": -0.3,
    }
    past_breeding_success_score, breeding_hits = _score_keywords(
        text_blob, breeding_keywords, baseline=0.45
    )

    availability_keywords = {
        "deceased": -0.95,
        "euthanized": -0.95,
        "quarantine": -0.2,
        "returned to china": -0.1,
        "healthy": 0.05,
    }
    availability_window_score, availability_hits = _score_keywords(
        text_blob, availability_keywords, baseline=0.75
    )

    confidence = 0.45
    if row.get("curated_source_label"):
        confidence += 0.25
    if row.get("description_agent"):
        confidence += 0.15
    if row.get("health_notes_agent"):
        confidence += 0.15
    extraction_confidence = _clamp(confidence)

    evidence = {
        "health_hits": health_hits,
        "aggression_hits": aggression_hits,
        "fertility_hits": fertility_hits,
        "breeding_hits": breeding_hits,
        "availability_hits": availability_hits,
        "personality_tags": merged_tags,
    }

    return {
        "personality_tags_norm": personality_tags_norm or None,
        "health_risk_score": health_risk_score,
        "fertility_status_score": fertility_status_score,
        "aggression_level_score": aggression_level_score,
        "past_breeding_success_score": past_breeding_success_score,
        "availability_window_score": availability_window_score,
        "extraction_confidence": extraction_confidence,
        "evidence_json": evidence,
    }


def main() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    db_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    query = """
        SELECT
            source_id,
            name,
            description_agent,
            life_journey,
            breeding_notes,
            health_notes_agent,
            personality_tags_agent,
            curated_source_label
        FROM core.panda_profiles_agent
    """

    loaded = 0
    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query)
            rows = cur.fetchall()

        with conn.cursor() as cur:
            for row in rows:
                features = _derive_features(row)
                cur.execute(
                    """
                    INSERT INTO core.panda_text_features (
                        source_id,
                        personality_tags_norm,
                        health_risk_score,
                        fertility_status_score,
                        aggression_level_score,
                        past_breeding_success_score,
                        availability_window_score,
                        extraction_confidence,
                        evidence_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (source_id)
                    DO UPDATE SET
                        personality_tags_norm = EXCLUDED.personality_tags_norm,
                        health_risk_score = EXCLUDED.health_risk_score,
                        fertility_status_score = EXCLUDED.fertility_status_score,
                        aggression_level_score = EXCLUDED.aggression_level_score,
                        past_breeding_success_score = EXCLUDED.past_breeding_success_score,
                        availability_window_score = EXCLUDED.availability_window_score,
                        extraction_confidence = EXCLUDED.extraction_confidence,
                        evidence_json = EXCLUDED.evidence_json,
                        updated_at = NOW()
                    """,
                    (
                        str(row["source_id"]),
                        features["personality_tags_norm"],
                        Decimal(str(features["health_risk_score"])),
                        Decimal(str(features["fertility_status_score"])),
                        Decimal(str(features["aggression_level_score"])),
                        Decimal(str(features["past_breeding_success_score"])),
                        Decimal(str(features["availability_window_score"])),
                        Decimal(str(features["extraction_confidence"])),
                        json.dumps(features["evidence_json"]),
                    ),
                )
                loaded += 1
        conn.commit()

    print(f"Upserted text features for {loaded} pandas")


if __name__ == "__main__":
    main()
