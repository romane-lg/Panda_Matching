from __future__ import annotations

from typing import Any

from panda_matching.agent import router
from panda_matching.agent.llm import LLMRateLimitError


def _top_match_rows() -> list[dict[str, Any]]:
    return [
        {
            "focal_panda_name": "Ai Bao",
            "candidate_panda_name": "Xi Lan",
            "recommendation_rank_v2": 1,
            "top_positive_factors": "low combined health risk",
            "top_negative_factors": "weak personality overlap",
            "age_gap_years": 5,
            "focal_country": "South Korea",
            "focal_city_region": "Yongin",
            "candidate_country": "USA",
            "candidate_city_region": "Atlanta",
            "focal_personality_text": "calm and affectionate",
            "candidate_personality_text": "playful and energetic",
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
            "focal_panda_name": "Ai Bao",
            "candidate_panda_name": "Tai Shan",
            "recommendation_rank_v2": 2,
            "top_positive_factors": "low combined health risk",
            "top_negative_factors": "weak personality overlap",
            "age_gap_years": 8,
            "focal_country": "South Korea",
            "focal_city_region": "Yongin",
            "candidate_country": "USA",
            "candidate_city_region": "Washington",
            "focal_personality_text": "calm and affectionate",
            "candidate_personality_text": "confident and energetic",
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
            "focal_panda_name": "Ai Bao",
            "candidate_panda_name": "Po",
            "recommendation_rank_v2": 4,
            "top_positive_factors": "low combined health risk",
            "top_negative_factors": "no major penalties",
            "age_gap_years": 3,
            "focal_country": "South Korea",
            "focal_city_region": "Yongin",
            "candidate_country": "Belgium",
            "candidate_city_region": "Pairi Daiza",
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


class _FakeMappingsResult:
    def __init__(self, row: dict[str, Any] | list[dict[str, Any]] | None) -> None:
        self._row = row

    def mappings(self) -> _FakeMappingsResult:
        return self

    def first(self) -> dict[str, Any] | None:
        if isinstance(self._row, list):
            return self._row[0] if self._row else None
        return self._row

    def all(self) -> list[dict[str, Any]]:
        if self._row is None:
            return []
        if isinstance(self._row, list):
            return self._row
        return [self._row]


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
    assert "Ai Bao's current top 3 matches are:" in turn.response
    assert "1. Xi Lan" in turn.response
    assert "2. Tai Shan" in turn.response
    assert "Xi Lan:" in turn.response
    assert "Tai Shan:" in turn.response
    assert "Po:" in turn.response
    assert "Ai Bao and Xi Lan both look relatively healthy" in turn.response
    assert (
        "compared with Xi Lan, this one is weaker because Xi Lan has the tighter age gap"
        in turn.response
    )
    assert "strong health penalties" not in turn.response
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
    assert "the personality side looks less convincing" in turn.response


def test_first_place_prompt_routes_to_match_explanation(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Bao": "Ai Bao", "Xi Lan": "Xi Lan"}.get(name),
    )
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "Why is Xi Lan Ai Bao's first top match?", {})

    assert turn.intent == "match_explanation"
    assert "Xi Lan comes out as Ai Bao's top match" in turn.response


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
    assert "the personality fit looks weaker" not in turn.response


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
                    "top_positive_factors": "high logistical feasibility; low combined health risk",
                    "top_negative_factors": "weak personality overlap",
                    "age_gap_years": 5,
                    "focal_city_region": "Yongin",
                    "focal_country": "South Korea",
                    "candidate_city_region": "Atlanta",
                    "candidate_country": "USA",
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
                    "top_positive_factors": "high logistical feasibility; low combined health risk",
                    "focal_city_region": "Yongin",
                    "focal_country": "South Korea",
                    "candidate_city_region": "Atlanta",
                    "candidate_country": "USA",
                }
            ],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "tell me about a panda match", {})

    assert turn.intent == "best_overall"
    assert "strongest overall directional match" in turn.response
    assert "score" not in turn.response
    assert "Ai Bao and Xi Lan" in turn.response
    assert (
        "Ai Bao is in Yongin and Xi Lan is in Atlanta, so geography does not look like "
        "the main obstacle for a transfer"
    ) in turn.response


def test_top_matches_explanation_prompt_routes_before_profile_lookup(monkeypatch) -> None:
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

    turn = router.route_chat_message(object(), "Why are Ai Bao's top matches scored highly?", {})

    assert turn.intent == "top_matches"
    assert "Ai Bao's current top 3 matches are:" in turn.response
    assert "I could not find a panda matching" not in turn.response


def test_vague_best_match_uses_last_panda_memory(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda _session, panda_name, k: seen.update({"panda_name": panda_name, "k": k})
        or {
            "count": 1,
            "matches": _top_match_rows()[:1],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )
    monkeypatch.setattr(
        router,
        "best_overall_match_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected global route")),
    )

    turn = router.route_chat_message(
        object(),
        "Who is the best match?",
        {"last_panda_name": "Ai Bao"},
    )

    assert turn.intent == "top_matches"
    assert seen == {"panda_name": "Ai Bao", "k": 1}
    assert "Ai Bao's current top match is:" in turn.response
    assert "1. Xi Lan" in turn.response


def test_profile_overview_uses_richer_profile_fields(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {
            "name": "Ai Bao",
            "sex": "female",
            "status": "alive",
            "age_years": 12,
            "zoo_or_facility": "Everland Resort",
            "city_region": "Yongin",
            "country": "South Korea",
            "babies_had_count": 3,
            "personality_text": "calm, attentive",
            "health_text": "no major recorded issues",
            "description_text": "Ai Bao is part of the Korean breeding program.",
        },
    )

    turn = router.route_chat_message(object(), "Tell me more about Ai Bao", {})

    assert turn.intent == "panda_info"
    assert "Everland Resort" in turn.response
    assert "Ai Bao is part of the Korean breeding program." in turn.response
    assert "personality notes include calm, attentive" in turn.response
    assert "health context says no major recorded issues" in turn.response


def test_broad_pair_question_routes_to_pair_opinion(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Bao": "Ai Bao", "Ai Li": "Ai Li"}.get(name),
    )
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 2,
            "matches": [
                {
                    "candidate_panda_name": "Ai Li",
                    "recommendation_rank_v2": 2,
                    "top_positive_factors": "low combined health risk",
                    "top_negative_factors": "no major penalties",
                    "age_gap_years": 4,
                    "candidate_panda_sex": "male",
                    "focal_panda_age": 12,
                    "candidate_panda_age": 16,
                    "candidate_panda_babies": 0,
                    "zero_babies_bonus": 8,
                    "male_older_bonus": 10,
                    "health_penalty_score": 0.04,
                    "bio_component": 0.82,
                    "behavior_component": 0.58,
                    "logistics_component": 0.55,
                }
            ],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "Would Ai Bao and Ai Li make a good match", {})

    assert turn.intent == "pairwise_match_opinion"
    assert "Ai Li does look like a strong match for Ai Bao" in turn.response


def test_unranked_pair_explains_profile_blocker(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Bao": "Ai Bao", "Ai Li": "Ai Li"}.get(name),
    )
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 1,
            "matches": _top_match_rows()[:1],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda _session, panda_name: {
            "Ai Bao": {"name": "Ai Bao", "sex": "female", "status": "alive"},
            "Ai Li": {"name": "Ai Li", "sex": "female", "status": "alive"},
        }.get(panda_name),
    )

    turn = router.route_chat_message(object(), "Would Ai Bao and Ai Li make a good match", {})

    assert turn.intent == "pairwise_match_not_found"
    assert "both pandas are recorded as female" in turn.response


def test_intent_classifier_core_categories() -> None:
    pair_intent = router.classify_chat_intent("Would Ai Bao and Bao Li make a good match", {})
    assert pair_intent.category == "match"
    assert router.classify_chat_intent("Top 5 matches for Ai Bao", {}).category == "ranking"
    assert router.classify_chat_intent("Explain Ai Bao Bao Li", {}).category == "explanation"
    assert router.classify_chat_intent("Tell me about Ai Bao", {}).category == "profile"
    assert router.classify_chat_intent("Is Ai Bao alive", {}).category == "health_status"
    assert router.classify_chat_intent("How many eligible pandas", {}).category == "eligibility"


def test_vague_match_without_context_asks_clarification(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "best_overall_match_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should clarify first")),
    )

    turn = router.route_chat_message(object(), "Best match?", {})

    assert turn.intent == "clarification_needed"
    assert "Which panda" in turn.response


def test_best_match_for_specific_panda_does_not_ask_for_clarification(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "An An")
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 0,
            "matches": [],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )
    monkeypatch.setattr(
        router,
        "diagnose_no_matches",
        lambda *_args, **_kwargs: {
            "summary": "An An has no ranked matches.",
            "waterfall": {},
            "profile": {"status": "deceased"},
        },
    )
    monkeypatch.setattr(router, "curated_override_for_name", lambda *_args, **_kwargs: None)

    turn = router.route_chat_message(object(), "who is the best match for An An", {})

    assert turn.intent == "top_matches_no_results"
    assert "Which panda should I use" not in turn.response
    assert "An An has no ranked matches." in turn.response


def test_compatible_pair_wording_routes_to_pair_opinion(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Bao": "Ai Bao", "Bao Li": "Bao Li"}.get(name),
    )
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 1,
            "matches": [
                {
                    "candidate_panda_name": "Bao Li",
                    "recommendation_rank_v2": 1,
                    "top_positive_factors": "low combined health risk",
                    "top_negative_factors": "no major penalties",
                    "age_gap_years": 4,
                    "candidate_panda_sex": "male",
                    "focal_panda_age": 12,
                    "candidate_panda_age": 16,
                    "candidate_panda_babies": 0,
                    "zero_babies_bonus": 8,
                    "male_older_bonus": 10,
                    "health_penalty_score": 0.04,
                    "bio_component": 0.82,
                    "behavior_component": 0.58,
                    "logistics_component": 0.55,
                }
            ],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "Are Ai Bao and Bao Li compatible", {})

    assert turn.intent == "pairwise_match_opinion"
    assert "Bao Li looks like an excellent match for Ai Bao" in turn.response


def test_candidate_comparison_uses_bulleted_sections(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {
            "Ai Bao": "Ai Bao",
            "Xi Lan": "Xi Lan",
            "Tai Shan": "Tai Shan",
        }.get(name),
    )
    monkeypatch.setattr(
        router,
        "compare_candidates_for_focal_data",
        lambda *_args, **_kwargs: {
            "candidate_a_name": "Xi Lan",
            "candidate_b_name": "Tai Shan",
            "candidate_a": _top_match_rows()[0],
            "candidate_b": _top_match_rows()[1],
            "better_match": _top_match_rows()[0],
            "missing_candidates": [],
        },
    )

    turn = router.route_chat_message(
        object(), "Is Xi Lan or Tai Shan better for Ai Bao, and why?", {}
    )

    assert turn.intent == "compare_candidates"
    assert "Why Xi Lan is ahead:" in turn.response
    assert "- Xi Lan is currently ranked #1." in turn.response
    assert "How Tai Shan compares:" in turn.response
    assert "the personality fit looks weaker" not in turn.response


def test_odd_age_gap_prompt_routes_to_match_explanation(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Bao": "Ai Bao", "Xiao Liwu": "Xiao Liwu"}.get(name),
    )
    rows = _top_match_rows() + [
        {
            "focal_panda_name": "Ai Bao",
            "candidate_panda_name": "Xiao Liwu",
            "recommendation_rank_v2": 3,
            "top_positive_factors": "low combined health risk",
            "top_negative_factors": "weak personality overlap",
            "age_gap_years": 11,
            "focal_country": "South Korea",
            "focal_city_region": "Yongin",
            "candidate_country": "China",
            "candidate_city_region": "Chengdu",
            "focal_personality_text": "beloved",
            "candidate_personality_text": "playful",
            "candidate_panda_sex": "male",
            "focal_panda_age": 12,
            "candidate_panda_age": 23,
            "candidate_panda_babies": 0,
            "zero_babies_bonus": 8,
            "male_older_bonus": 10,
            "health_penalty_score": 0.03,
            "bio_component": 0.74,
            "behavior_component": 0.42,
            "logistics_component": 0.41,
        }
    ]
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 4,
            "matches": rows,
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(
        object(),
        (
            "It seems odd to me that Ai Bao and Xiao Liwu make a good match considering "
            "the age gap. Tell me why."
        ),
        {},
    )

    assert turn.intent == "match_explanation"
    assert "Xiao Liwu" in turn.response
    assert "Xi Lan" not in turn.response.splitlines()[0]


def test_explanation_context_uses_last_pair(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: None)
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 1,
            "matches": _top_match_rows()[:1],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(
        object(),
        "Why are they a good match?",
        {"last_panda_name": "Ai Bao", "last_candidate_name": "Xi Lan"},
    )

    assert turn.intent == "match_explanation"
    assert "Xi Lan comes out as Ai Bao's top match" in turn.response


def test_weak_personality_overlap_follow_up_uses_last_pair(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: None)
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 1,
            "matches": _top_match_rows()[:1],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(
        object(),
        "It seems odd that weak personality overlap still leads to a strong match. Explain that.",
        {"last_panda_name": "Ai Bao", "last_candidate_name": "Xi Lan"},
    )

    assert turn.intent == "match_explanation"
    assert "Xi Lan comes out as Ai Bao's top match" in turn.response


def test_status_question_returns_status(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {
            "name": "Ai Bao",
            "sex": "female",
            "status": "alive",
            "age_years": 12,
            "health_text": "no major recorded issues",
        },
    )

    turn = router.route_chat_message(object(), "Is Ai Bao alive", {})

    assert turn.intent == "panda_info"
    assert "Ai Bao is listed as alive" in turn.response


def test_pronoun_status_question_uses_last_panda_memory(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {
            "name": "Ai Bao",
            "status": "alive",
        },
    )

    turn = router.route_chat_message(
        object(),
        "is he dead or alive?",
        {"last_panda_name": "Ai Bao"},
    )

    assert turn.intent == "panda_info"
    assert turn.response == "Ai Bao is listed as alive."


def test_location_count_alias_for_america(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_scalar_int(_session, _sql, params=None):
        seen["params"] = params
        return 4

    monkeypatch.setattr(router, "scalar_int", fake_scalar_int)

    turn = router.route_chat_message(object(), "how many pandas in america", {})

    assert turn.intent == "analytics_count_location"
    assert turn.data == {
        "requested_location": "america",
        "normalized_location": "USA",
        "display_location": "USA",
        "count": 4,
    }
    assert seen["params"] == {
        "location": "USA",
        "location_like": "%USA%",
    }


def test_location_any_count_for_canada(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_scalar_int(_session, _sql, params=None):
        seen["params"] = params
        return 0

    monkeypatch.setattr(router, "scalar_int", fake_scalar_int)

    turn = router.route_chat_message(object(), "are there any pandas in canada", {})

    assert turn.intent == "analytics_count_location"
    assert turn.response == "No, I do not see any pandas listed in Canada."
    assert turn.data == {
        "requested_location": "canada",
        "normalized_location": "Canada",
        "display_location": "Canada",
        "count": 0,
    }
    assert seen["params"] == {
        "location": "Canada",
        "location_like": "%Canada%",
    }


def test_count_male_pandas_does_not_leak_relation(monkeypatch) -> None:
    monkeypatch.setattr(router, "scalar_int", lambda *_args, **_kwargs: 68)

    turn = router.route_chat_message(object(), "How many male pandas are there in my data?", {})

    assert turn.intent == "analytics_count_sex"
    assert turn.response == "There are 68 male pandas."


def test_eligible_count_explanation_follow_up(monkeypatch) -> None:
    turn = router.route_chat_message(
        object(),
        "how are you calculating that",
        {
            "last_query_kind": "analytics_count_eligible",
            "last_eligible_relation": "breedeable_pandas",
        },
    )

    assert turn.intent == "analytics_count_eligible_explained"
    assert "eligible breeding list" in turn.response
    assert "core.breedeable_pandas" in turn.response


def test_health_notes_inventory_query(monkeypatch) -> None:
    monkeypatch.setattr(router, "relation_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(router, "relation_columns", lambda *_args, **_kwargs: {"health_notes"})
    monkeypatch.setattr(router, "scalar_int", lambda *_args, **_kwargs: 3)
    session = type(
        "FakeSession",
        (),
        {
            "execute": lambda self, _query: _FakeMappingsResult(
                [{"name": "Ai Bao"}, {"name": "Ai Jiu"}, {"name": "Er Shun"}]
            )
        },
    )()

    turn = router.route_chat_message(session, "which pandas do you have health notes for", {})

    assert turn.intent == "analytics_health_notes_available"
    assert "I currently have health notes for 3 pandas." in turn.response
    assert "Ai Bao" in turn.response
    assert "Ai Jiu" in turn.response


def test_no_matches_example_query(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "pick_relation",
        lambda *_args, **_kwargs: "ranked_directional_recommended_matches_v2",
    )
    session = type(
        "FakeSession",
        (),
        {
            "execute": lambda self, _query: _FakeMappingsResult(
                {"name": "An An", "status": "deceased"}
            )
        },
    )()

    turn = router.route_chat_message(session, "Who is the panda with no matches", {})

    assert turn.intent == "analytics_no_matches_example"
    assert "One panda with no ranked matches right now is An An." in turn.response
    assert "listed as deceased" in turn.response


def test_no_match_diagnosis_uses_plain_language(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_args, **_kwargs: "Ai Bao")
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda *_args, **_kwargs: {
            "count": 1,
            "matches": _top_match_rows()[:1],
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "Why does Ai Bao have no matches?", {})

    assert turn.intent == "no_match_diagnosis"
    assert "different view of the data" in turn.response
    assert "different table" not in turn.response


def test_profile_cubs_question_returns_cub_count(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Er Shun")
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {
            "name": "Er Shun",
            "babies_had_count": 2,
        },
    )
    monkeypatch.setattr(
        router,
        "_offspring_summary",
        lambda *_args, **_kwargs: (
            "Recorded cubs include Jia Panpan (alive) and Jia Yueyue (alive)."
        ),
    )

    turn = router.route_chat_message(object(), "Does Er Shun have any cubs?", {})

    assert turn.intent == "panda_info"
    assert (
        turn.response
        == "Yes, Er Shun has 2 cubs recorded. Recorded cubs include Jia Panpan (alive) "
        "and Jia Yueyue (alive)."
    )


def test_health_question_without_notes_mentions_status(monkeypatch) -> None:
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {"name": "Ai Bao", "status": "alive", "health_text": ""},
    )

    turn = router.route_chat_message(object(), "Is Ai Bao in good health conditions?", {})

    assert turn.intent == "panda_info"
    assert "do not have detailed health notes for Ai Bao" in turn.response
    assert "lists Ai Bao as alive" in turn.response


def test_follow_up_what_about_reuses_health_question(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Bao": "Ai Bao", "Ai Jiu": "Ai Jiu"}.get(name),
    )
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda _session, panda_name: {"name": panda_name, "status": "alive", "health_text": ""},
    )

    first = router.route_chat_message(object(), "Is Ai Bao in good health conditions?", {})
    second = router.route_chat_message(
        object(),
        "what about Ai Jiu",
        {"last_panda_name": "Ai Bao", "last_query_kind": "health"},
    )

    assert first.intent == "panda_info"
    assert second.intent == "panda_info"
    assert "Ai Jiu" in second.response
    assert "top 5 matches" not in second.response


def test_follow_up_and_name_reuses_fun_fact(monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Bao": "Ai Bao", "Ai Jiu": "Ai Jiu"}.get(name),
    )
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda _session, panda_name: {"name": panda_name, "babies_had_count": 3},
    )

    turn = router.route_chat_message(
        object(),
        "and Ai Jiu",
        {"last_panda_name": "Ai Bao", "last_query_kind": "fun_fact"},
    )

    assert turn.intent == "panda_info"
    assert "Fun fact:" in turn.response
    assert "Ai Jiu" in turn.response


def test_youngest_panda_query_uses_analytics_path(monkeypatch) -> None:
    session = type(
        "FakeSession",
        (),
        {"execute": lambda self, _query: _FakeMappingsResult({"name": "Yuan Zai", "age_years": 2})},
    )()
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {
            "name": "Yuan Zai",
            "zoo_or_facility": "Taipei Zoo",
            "city_region": "Taipei",
            "country": "Taiwan",
            "babies_had_count": 0,
        },
    )

    turn = router.route_chat_message(session, "Who is the youngest panda in the dataset?", {})

    assert turn.intent == "analytics_youngest_panda"
    assert "Yuan Zai" in turn.response
    assert "2 years old" in turn.response
    assert "Taipei Zoo" in turn.response
    assert "core.panda_profiles" not in turn.response


def test_oldest_panda_query_uses_analytics_path(monkeypatch) -> None:
    session = type(
        "FakeSession",
        (),
        {"execute": lambda self, _query: _FakeMappingsResult({"name": "Jia Jia", "age_years": 38})},
    )()
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {
            "name": "Jia Jia",
            "zoo_or_facility": "Ocean Park",
            "city_region": "Hong Kong",
            "country": "China",
            "babies_had_count": 6,
        },
    )

    turn = router.route_chat_message(session, "who is the oldest panda", {})

    assert turn.intent == "analytics_oldest_panda"
    assert "Jia Jia" in turn.response
    assert "38 years old" in turn.response
    assert "Ocean Park" in turn.response
    assert "core.panda_profiles" not in turn.response


def test_most_cubs_query_uses_analytics_path(monkeypatch) -> None:
    session = type(
        "FakeSession",
        (),
        {
            "execute": lambda self, _query: _FakeMappingsResult(
                {"name": "Er Shun", "babies_had_count": 7}
            )
        },
    )()
    monkeypatch.setattr(
        router,
        "panda_profile_data",
        lambda *_args, **_kwargs: {
            "name": "Er Shun",
            "zoo_or_facility": "Toronto Zoo",
            "city_region": "Toronto",
            "country": "Canada",
            "babies_had_count": 7,
        },
    )

    turn = router.route_chat_message(session, "Who is the panda with the most cubs?", {})

    assert turn.intent == "analytics_most_cubs"
    assert "The panda with the most recorded cubs is Er Shun, with 7 cubs." in turn.response
    assert "Toronto Zoo" in turn.response


def test_top_matches_strips_trailing_why(monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda _session, panda_name, k: seen.update({"panda_name": panda_name, "k": k})
        or {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(
        object(),
        "tell me the best 5 matches for Ai Bao and why",
        {},
    )

    assert turn.intent == "top_matches"
    assert seen == {"panda_name": "Ai Bao", "k": 5}
    assert "Ai Bao's current top 3 matches are:" in turn.response


def test_who_is_best_two_matches_routes_to_top_matches(monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(router, "find_panda_name_by_substring", lambda *_: "Ai Bao")
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda _session, panda_name, k: seen.update({"panda_name": panda_name, "k": k})
        or {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn = router.route_chat_message(object(), "Who is the best 2 matches for Ai Bao", {})

    assert turn.intent == "top_matches"
    assert seen == {"panda_name": "Ai Bao", "k": 2}
    assert "Ai Bao's current top 2 matches are:" in turn.response


def test_what_about_follow_up_reuses_top_match_context(monkeypatch) -> None:
    seen: list[tuple[str, int]] = []

    def fake_top_matches(_session, panda_name, k):
        seen.append((panda_name, k))
        if panda_name == "An An":
            return {
                "count": 0,
                "matches": [],
                "source_view": "core.ranked_directional_recommended_matches_v2",
            }
        return {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        }

    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"An An": "An An", "Ai Li": "Ai Li"}.get(name),
    )
    monkeypatch.setattr(router, "top_matches_data", fake_top_matches)
    monkeypatch.setattr(
        router,
        "diagnose_no_matches",
        lambda *_args, **_kwargs: {
            "summary": "An An has no ranked matches.",
            "waterfall": {},
            "profile": {},
        },
    )
    monkeypatch.setattr(router, "curated_override_for_name", lambda *_args, **_kwargs: None)

    memory: dict[str, str] = {}
    first_turn = router.route_chat_message(object(), "Who are An An's top 5 matches?", memory)
    second_turn = router.route_chat_message(object(), "what about Ai Li", memory)

    assert first_turn.intent == "top_matches_no_results"
    assert second_turn.intent == "top_matches"
    assert "Ai Li's current top 3 matches are:" in second_turn.response
    assert seen == [("An An", 5), ("Ai Li", 5)]


def test_name_first_top_pattern_handles_missing_apostrophe_and_typos(monkeypatch) -> None:
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        router,
        "find_panda_name_by_substring",
        lambda _session, name: {"Ai Li": "Ai Li", "Bao Bao": "Bao Bao"}.get(name),
    )
    monkeypatch.setattr(
        router,
        "top_matches_data",
        lambda _session, panda_name, k: seen.append((panda_name, k))
        or {
            "count": 3,
            "matches": _top_match_rows(),
            "source_view": "core.ranked_directional_recommended_matches_v2",
        },
    )

    turn_one = router.route_chat_message(object(), "Whoa re Ai Li Top 5 matches", {})
    turn_two = router.route_chat_message(object(), "Who are Bao Bao top 5 matches", {})

    assert turn_one.intent == "top_matches"
    assert turn_two.intent == "top_matches"
    assert seen == [("Ai Li", 5), ("Bao Bao", 5)]


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
