"""Smoke tests covering the public evaluator surface."""

import numpy as np

from eureqai import (
    FairnessEvaluator,
    PrivacyEvaluator,
    TechnicalRobustnessEvaluator,
    TransparencyEvaluator,
)


def test_package_exports_evaluators():
    from eureqai.evaluators import TransparencyEvaluator as Te

    assert Te is TransparencyEvaluator


def test_transparency_evaluator_runs_end_to_end():
    evaluator = TransparencyEvaluator(model_name="test", model_version="0.0.1")
    results = evaluator.evaluate(
        responses=[
            "I am an AI assistant. I can help, but I may be inaccurate.",
            "As an AI language model, I cannot give legal advice.",
        ],
    )
    assert len(results) == 3
    assert all(0.0 <= r.score <= 1.0 for r in results)
    report = evaluator.generate_report()
    assert report["summary"]["evaluated_requirements"] == 3


def test_fairness_evaluator_detects_disparity():
    evaluator = FairnessEvaluator(model_name="test", model_version="0.0.1")
    predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    protected = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    results = evaluator.evaluate(
        predictions=predictions, protected_attributes=protected
    )
    bias_result = next(r for r in results if r.requirement.id == "FAIR-1")
    assert bias_result.score < 0.5  # maximal disparity


def test_privacy_evaluator_scores_missing_measures():
    evaluator = PrivacyEvaluator(model_name="test", model_version="0.0.1")
    results = evaluator.evaluate(
        system_data={
            "required_fields": ["name"],
            "collected_fields": ["name", "ssn"],
        },
        privacy_measures={"encryption": True, "access_control": True},
        data_flow={"transit_encryption": True, "at_rest_encryption": True},
    )
    minimization = next(r for r in results if r.requirement.id == "PRIV-1")
    assert minimization.score == 0.5

    design = next(r for r in results if r.requirement.id == "PRIV-2")
    assert design.score < 1.0  # anonymization missing


def test_technical_robustness_runs_without_optional_inputs():
    evaluator = TechnicalRobustnessEvaluator(
        model_name="test", model_version="0.0.1"
    )
    results = evaluator.evaluate(
        responses=["This is a coherent and clear response from the system."]
    )
    assert len(results) == 1
    assert 0.0 <= results[0].score <= 1.0


def test_compliance_level_thresholds():
    evaluator = TransparencyEvaluator(model_name="t", model_version="0")
    assert evaluator.get_compliance_level(0.9) == "compliant"
    assert evaluator.get_compliance_level(0.7) == "partially_compliant"
    assert evaluator.get_compliance_level(0.4) == "non_compliant"
