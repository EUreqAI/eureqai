"""Tests for ProhibitedPracticesEvaluator (Article 5)."""

import pytest

from eureqai import ProhibitedPracticesEvaluator, screen_blockers


def _evaluator():
    return ProhibitedPracticesEvaluator(model_name="test", model_version="1.0")


def _result(results, req_id):
    matches = [r for r in results if r.requirement.id == req_id]
    assert matches, f"missing result for {req_id}"
    return matches[0]


CLEAN_DECLARATIONS = {
    "subliminal_manipulation": "no",
    "vulnerability_exploitation": "no",
    "social_scoring": "no",
    "predictive_policing_profiling": "no",
    "untargeted_facial_scraping": "no",
    "emotion_recognition_workplace_education": "no",
    "biometric_categorisation_protected": "no",
    "realtime_remote_biometric_id_le": "no",
}


def test_all_no_declarations_pass_clean():
    results = _evaluator().evaluate(declarations=CLEAN_DECLARATIONS)
    assert len(results) == 8
    assert all(r.score == pytest.approx(1.0) for r in results)
    assert not screen_blockers(results)


def test_every_practice_is_critical_priority():
    evaluator = _evaluator()
    assert all(r.priority == "critical" for r in evaluator.requirements)
    assert len(evaluator.requirements) == 8


def test_practice_keys_is_canonical_list():
    keys = ProhibitedPracticesEvaluator.practice_keys()
    assert sorted(keys) == sorted(CLEAN_DECLARATIONS)


def test_unknown_practice_key_raises():
    evaluator = _evaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(declarations={"laser_eyes": "no"})


def test_invalid_declaration_raises():
    evaluator = _evaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            declarations={"subliminal_manipulation": "maybe"}
        )


def test_missing_key_is_treated_as_unclear():
    # Empty declarations means every practice is unanswered.
    results = _evaluator().evaluate(declarations={})
    assert all(r.score == pytest.approx(0.5) for r in results)
    assert all(
        "unclear" in r.metadata["declaration"] for r in results
    )


def test_absolute_prohibition_yes_is_hard_blocker():
    # Article 5(1)(a) has no carve-out.
    results = _evaluator().evaluate(
        declarations={**CLEAN_DECLARATIONS, "subliminal_manipulation": "yes"}
    )
    blocker = _result(results, "PROHIB-1")
    assert blocker.score == 0.0
    assert blocker.metadata.get("blocker") is True
    assert any("absolute prohibition" in r.lower() for r in blocker.recommendations)


def test_untargeted_facial_scraping_has_no_carve_out():
    # Even with a "carve-out" passed in, 5(1)(e) admits none.
    results = _evaluator().evaluate(
        declarations={
            **CLEAN_DECLARATIONS,
            "untargeted_facial_scraping": "yes",
        },
        carve_outs={"untargeted_facial_scraping": {"some_excuse": True}},
    )
    assert _result(results, "PROHIB-5").score == 0.0


def test_emotion_recognition_with_medical_carveout_is_amber():
    results = _evaluator().evaluate(
        declarations={
            **CLEAN_DECLARATIONS,
            "emotion_recognition_workplace_education": "yes",
        },
        carve_outs={
            "emotion_recognition_workplace_education": {
                "medical_or_safety_documented": True,
            }
        },
    )
    result = _result(results, "PROHIB-6")
    assert result.score == pytest.approx(0.5)
    assert "narrow" in " ".join(result.recommendations).lower()


def test_emotion_recognition_without_carveout_is_blocker():
    results = _evaluator().evaluate(
        declarations={
            **CLEAN_DECLARATIONS,
            "emotion_recognition_workplace_education": "yes",
        },
    )
    result = _result(results, "PROHIB-6")
    assert result.score == 0.0
    assert "medical_or_safety_documented" in str(
        result.metadata.get("missing_carve_outs")
    )


def test_realtime_rbi_requires_two_carveouts():
    # 5(1)(h) requires BOTH authorisation and a purpose in the listed
    # exceptions. Supplying only one is still a blocker.
    results = _evaluator().evaluate(
        declarations={
            **CLEAN_DECLARATIONS,
            "realtime_remote_biometric_id_le": "yes",
        },
        carve_outs={
            "realtime_remote_biometric_id_le": {
                "authorisation_documented": True,
                # purpose_in_listed_exceptions intentionally missing
            }
        },
    )
    result = _result(results, "PROHIB-8")
    assert result.score == 0.0
    assert "purpose_in_listed_exceptions" in str(
        result.metadata.get("missing_carve_outs")
    )


def test_realtime_rbi_with_both_carveouts_documented_is_amber():
    results = _evaluator().evaluate(
        declarations={
            **CLEAN_DECLARATIONS,
            "realtime_remote_biometric_id_le": "yes",
        },
        carve_outs={
            "realtime_remote_biometric_id_le": {
                "authorisation_documented": True,
                "purpose_in_listed_exceptions": "victim search",
            }
        },
    )
    assert _result(results, "PROHIB-8").score == pytest.approx(0.5)


def test_unclear_declaration_is_amber_with_legal_review_recommendation():
    results = _evaluator().evaluate(
        declarations={**CLEAN_DECLARATIONS, "social_scoring": "unclear"}
    )
    result = _result(results, "PROHIB-3")
    assert result.score == pytest.approx(0.5)
    assert any("counsel" in r.lower() for r in result.recommendations)


def test_screen_blockers_helper_returns_only_problematic_results():
    results = _evaluator().evaluate(
        declarations={
            **CLEAN_DECLARATIONS,
            "social_scoring": "yes",
            "untargeted_facial_scraping": "yes",
        }
    )
    blockers = screen_blockers(results)
    assert len(blockers) == 2
    assert {r.requirement.id for r in blockers} == {"PROHIB-3", "PROHIB-5"}


def test_generate_report_marks_overall_non_compliant_on_blocker():
    evaluator = _evaluator()
    evaluator.evaluate(
        declarations={**CLEAN_DECLARATIONS, "social_scoring": "yes"}
    )
    report = evaluator.generate_report()
    assert report["summary"]["compliance_level"] == "non_compliant"
    assert len(report["summary"]["critical_issues"]) == 1
