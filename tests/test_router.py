from __future__ import annotations

from typing import Any

from panda_matching.agent import router
from panda_matching.agent.llm import LLMRateLimitError


def _top_match_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate_panda_name": "Xi Lan",
            "recommendation_rank_v2": 1,
            "top_positive_factors": "low combined health risk",
            "top_negative_factors": "weak personality overlap",
            "age_gap_years": 5,
            "candidate_panda_sex": "male",
            "focal_panda_age": 12,
            "candidate_panda_age": 17,
            "candidate_panda_babies": 0,
            "zero_babies_bonus": 8,
            "male_older_bonus": 10,
            "health_penalty_score": 0.05,
            "bio_component": 0.86,
            "behavior_component": 0.528,
            "logistics_component": 0.47,
        },
        {
            "candidate_panda_name": "Tai Shan",
            "recommendation_rank_v2": 2,
            "top_positive_factors": "low combined health risk",
            "top_negative_factors": "weak personality overlap",
            "age_gap_years": 8,
            "candidate_panda_sex": "male",
            "focal_panda_age": 12,
            "candidate_panda_age": 20,
            "candidate_panda_babies": 0,
            "zero_babies_bonus": 8,
            "male_older_bonus": 10,
            "health_penalty_score": 0.025,
            "bio_component": 0.806,
            "behavior_component": 0.528,
            "logistics_component": 0.47,
        },
        {
            "candidate_panda_name": "Po",
            "recommendation_rank_v2": 4,
            "top_positive_factors": "low combined health risk",
            "top_negative_factors": "no major penalties",
            "age_gap_years": 3,
            "candidate_panda_sex": "male",
            "focal_panda_age": 12,
            "candidate_panda_age": 15,
            "candidate_panda_babies": 0,
            "zero_babies_bonus": 8,
            "male_older_bonus": 10,
            "health_penalty_score": 0.02,
            "bio_component": 0.77,
            "behavior_component": 0.55,
            "logistics_component": 0.47,
        },
    ]


def test_top_matches_prompt_uses_deterministic_readable_summary(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "Who are Ai Bao's top 5 matches?", {})

    assert turn.intent == "top_matches"
    assert "Ai Bao's current top 3 matches are Xi Lan, Tai Shan, and Po." in turn.response
    assert "I can help with matches" not in turn.response
    assert not turn.response.strip().startswith("{")


def test_second_place_prompt_explains_gap_against_first(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "Why is Tai Shan only second for Ai Bao?", {})

    assert turn.intent == "match_explanation"
    assert "Ranking context:" in turn.response
    assert "Xi Lan has the tighter age gap" in turn.response
    assert "Xi Lan also looks a bit stronger biologically" in turn.response


def test_pair_opinion_prompt_includes_why_not_section(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(
        object(),
        "Do you think Ai Bao and Po would make a great match, why and why not?",
        {},
    )

    assert turn.intent == "pairwise_match_opinion"
    assert "Why it could work:" in turn.response
    assert "Why it may not be the best option:" in turn.response
    assert "Xi Lan also looks a bit stronger biologically" in turn.response


def test_most_eligible_panda_routes_to_best_overall_before_profile_lookup(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: None)
    monkeypatch.setattr(
        router,
        "best_overall_match_data",
        lambda *_args, **_kwargs: {
            "count": 1,
            "matches": [
                {
                    "focal_panda_name": "Ai Bao",
                    "candidate_panda_name": "Xi Lan",
                    "final_score_v2": 0.91,
                    "top_positive_factors": "low combined health risk",
                    "top_negative_factors": "weak personality overlap",
                    "age_gap_years": 5,
                    "bio_component": 0.86,
                    "behavior_component": 0.528,
                    "logistics_component": 0.47,
                }
            ],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "who is the most eligible panda", {})

    assert turn.intent == "best_overall"
    assert "Ai Bao with Xi Lan" in turn.response
    assert "I could not find a panda matching" not in turn.response


def test_broad_match_prompt_routes_to_best_overall(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "best_overall_match_data",
        lambda *_args, **_kwargs: {
            "count": 1,
            "matches": [
                {
                    "focal_panda_name": "Ai Bao",
                    "candidate_panda_name": "Xi Lan",
                    "final_score_v2": 0.91,
                    "top_positive_factors": "low combined health risk",
                }
            ],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "tell me about a panda match", {})

    assert turn.intent == "best_overall"
    assert "strongest overall directional match" in turn.response


def test_rate_limited_llm_falls_back_to_help(monkeypatch) -> None:
    fallback_events: list[tuple[str, str]] = []
    monkeypatch.setattr(router, "llm_enabled", lambda: True)
    monkeypatch.setattr(
        router,
        "llm_plan_message",
        lambda **_kwargs: (_ for _ in ()).throw(LLMRateLimitError("429")),
    )
    monkeypatch.setattr(
        router,
        "_record_llm_fallback",
        lambda reason, phase: fallback_events.append((reason, phase)),
    )

    turn = router.route_chat_message(object(), "help", {})

    assert turn.intent == "help"
    assert fallback_events == [("429", "planning")]
