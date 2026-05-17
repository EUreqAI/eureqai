"""Tests for the GPAI obligations evaluator (Articles 51, 53, 55)."""

import pytest

from eureqai import GPAIEvaluator
from eureqai.evaluators.gpai import SYSTEMIC_RISK_FLOPS_THRESHOLD


def _evaluator():
    return GPAIEvaluator(model_name="test-gpai", model_version="1.0")


def _result(results, req_id):
    matches = [r for r in results if r.requirement.id == req_id]
    assert matches, f"missing result for {req_id}"
    return matches[0]


def test_classification_below_threshold_is_not_systemic():
    results = _evaluator().evaluate(
        training_compute_flops=1e24,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": "https://example.com/summary",
        },
    )
    classification = _result(results, "GPAI-CLASS")
    assert classification.metadata["has_systemic_risk"] is False
    # Article 55 obligations should be marked not applicable, score 1.0.
    eval_result = _result(results, "GPAI-EVAL")
    assert eval_result.metadata["applicable"] is False
    assert eval_result.score == 1.0


def test_classification_above_threshold_triggers_article_55():
    results = _evaluator().evaluate(
        training_compute_flops=SYSTEMIC_RISK_FLOPS_THRESHOLD,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": "https://example.com/summary",
        },
        systemic_risk_measures={
            "model_evaluation": True,
            "adversarial_testing": True,
            "risk_assessment": True,
            "incident_reporting_process": True,
            "cybersecurity_measures": True,
        },
    )
    classification = _result(results, "GPAI-CLASS")
    assert classification.metadata["has_systemic_risk"] is True
    eval_result = _result(results, "GPAI-EVAL")
    assert eval_result.metadata["applicable"] is True
    assert eval_result.score == pytest.approx(1.0)


def test_commission_designation_alone_triggers_systemic_risk():
    results = _evaluator().evaluate(
        designated_systemic_risk=True,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": "https://example.com/s",
        },
    )
    assert _result(results, "GPAI-CLASS").metadata["has_systemic_risk"] is True
    incident = _result(results, "GPAI-INCIDENT")
    assert incident.metadata["applicable"] is True
    assert incident.score == 0.0  # no measures declared


def test_missing_classification_input_flags_undocumented():
    results = _evaluator().evaluate(documentation={})
    classification = _result(results, "GPAI-CLASS")
    assert classification.score == 0.0
    assert classification.recommendations


def test_open_source_exempts_only_article_53_a_b_when_not_systemic():
    results = _evaluator().evaluate(
        training_compute_flops=1e23,
        is_open_source=True,
        documentation={
            # 53(1)(a) and (b) intentionally absent to test the exemption.
            "copyright_policy": True,
            "training_data_summary": "https://example.com/s",
        },
    )
    assert _result(results, "GPAI-DOC").score == 1.0  # exempt
    assert _result(results, "GPAI-DOWN").score == 1.0  # exempt
    assert _result(results, "GPAI-COPYRIGHT").score == 1.0
    assert _result(results, "GPAI-SUMMARY").score == 1.0


def test_open_source_does_not_exempt_systemic_risk_models():
    results = _evaluator().evaluate(
        training_compute_flops=SYSTEMIC_RISK_FLOPS_THRESHOLD * 2,
        is_open_source=True,
        documentation={
            "copyright_policy": True,
            "training_data_summary": "https://example.com/s",
        },
    )
    # Annex XI/XII obligations are NOT exempt for systemic-risk models.
    assert _result(results, "GPAI-DOC").score == 0.0
    assert _result(results, "GPAI-DOWN").score == 0.0


def test_open_source_never_exempts_copyright_or_summary():
    results = _evaluator().evaluate(
        training_compute_flops=1e23,
        is_open_source=True,
        documentation={},  # nothing declared
    )
    assert _result(results, "GPAI-COPYRIGHT").score == 0.0
    assert _result(results, "GPAI-SUMMARY").score == 0.0


def test_training_summary_must_look_public():
    results = _evaluator().evaluate(
        training_compute_flops=1e23,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": True,  # bool, not a URL
        },
    )
    summary = _result(results, "GPAI-SUMMARY")
    assert summary.score == pytest.approx(0.5)
    assert summary.recommendations


def test_partial_systemic_risk_measures_score_partially():
    results = _evaluator().evaluate(
        designated_systemic_risk=True,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": "https://example.com/s",
        },
        systemic_risk_measures={
            "model_evaluation": True,
            "adversarial_testing": False,  # half-credit for GPAI-EVAL
            "risk_assessment": True,
            "incident_reporting_process": False,
            "cybersecurity_measures": True,
        },
    )
    assert _result(results, "GPAI-EVAL").score == pytest.approx(0.5)
    assert _result(results, "GPAI-RISK").score == pytest.approx(1.0)
    assert _result(results, "GPAI-INCIDENT").score == pytest.approx(0.0)
    assert _result(results, "GPAI-CYBER").score == pytest.approx(1.0)


def test_code_of_practice_boosts_confidence():
    base = _evaluator().evaluate(
        training_compute_flops=1e23,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": "https://example.com/s",
        },
    )
    boosted = _evaluator().evaluate(
        training_compute_flops=1e23,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": "https://example.com/s",
        },
        relies_on_code_of_practice=True,
    )
    base_doc = _result(base, "GPAI-DOC")
    boosted_doc = _result(boosted, "GPAI-DOC")
    assert boosted_doc.confidence > base_doc.confidence


def test_generate_report_summary_includes_all_requirements():
    evaluator = _evaluator()
    evaluator.evaluate(
        training_compute_flops=SYSTEMIC_RISK_FLOPS_THRESHOLD,
        documentation={
            "technical_documentation_annex_xi": True,
            "downstream_information_annex_xii": True,
            "copyright_policy": True,
            "training_data_summary": "https://example.com/s",
        },
        systemic_risk_measures={
            "model_evaluation": True,
            "adversarial_testing": True,
            "risk_assessment": True,
            "incident_reporting_process": True,
            "cybersecurity_measures": True,
        },
    )
    report = evaluator.generate_report()
    assert report["summary"]["evaluated_requirements"] == 9
    assert report["summary"]["compliance_level"] == "compliant"
