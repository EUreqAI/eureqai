"""Privacy and data protection evaluator.

Maps to Regulation (EU) 2024/1689 Article 10 (Data and data governance) read
together with the GDPR (Reg. (EU) 2016/679). The AI Act itself does not
restate GDPR principles, but data quality, minimisation and protection
measures are part of an Article 10 quality management system.
"""

from typing import Any, Dict, List

from eureqai.evaluators.base import (
    BaseEvaluator,
    EvaluationResult,
    Requirement,
)


REQUIRED_PRIVACY_MEASURES: Dict[str, Dict[str, Any]] = {
    "encryption": {"weight": 0.3, "required": True},
    "anonymization": {"weight": 0.3, "required": True},
    "access_control": {"weight": 0.2, "required": True},
    "data_retention": {"weight": 0.1, "required": False},
    "audit_logging": {"weight": 0.1, "required": False},
}


class PrivacyEvaluator(BaseEvaluator):
    """Evaluates privacy and data protection requirements (Article 10)."""

    def _initialize_requirements(self) -> None:
        self.requirements = [
            Requirement(
                id="PRIV-1",
                name="Data Minimisation",
                description=(
                    "System collects and processes only data necessary for its "
                    "intended purpose."
                ),
                article="Article 10(2)",
                priority="high",
                category="Privacy",
                metrics=["data_necessity_score"],
            ),
            Requirement(
                id="PRIV-2",
                name="Privacy by Design",
                description=(
                    "Privacy controls are integrated into the system design."
                ),
                article="Article 10(5)",
                priority="critical",
                category="Privacy",
                metrics=["privacy_design_score"],
            ),
            Requirement(
                id="PRIV-3",
                name="Data Protection Measures",
                description=(
                    "Appropriate technical and organisational measures protect "
                    "personal data in the AI lifecycle."
                ),
                article="Article 10(5); GDPR Article 32",
                priority="critical",
                category="Privacy",
                metrics=["protection_measure_score"],
            ),
        ]

    def evaluate(
        self,
        system_data: Dict[str, Any],
        privacy_measures: Dict[str, Any],
        data_flow: Dict[str, Any],
        **kwargs,
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        for result in (
            self._evaluate_data_minimization(system_data),
            self._evaluate_privacy_design(privacy_measures),
            self._evaluate_data_protection(data_flow),
        ):
            results.append(result)
            self.results.append(result)

        return results

    def _evaluate_data_minimization(
        self, system_data: Dict[str, Any]
    ) -> EvaluationResult:
        required_fields = set(system_data.get("required_fields", []))
        collected_fields = list(system_data.get("collected_fields", []))
        if not collected_fields:
            return EvaluationResult(
                requirement=self.requirements[0],
                score=1.0,
                confidence=0.6,
                evidence=["No fields declared as collected."],
                recommendations=[],
            )
        unnecessary = [f for f in collected_fields if f not in required_fields]
        score = 1.0 - len(unnecessary) / len(collected_fields)
        return EvaluationResult(
            requirement=self.requirements[0],
            score=score,
            confidence=0.9,
            evidence=[
                f"Collected fields: {collected_fields}",
                f"Unnecessary fields: {unnecessary}",
            ],
            recommendations=self._minimization_recommendations(unnecessary),
        )

    def _evaluate_privacy_design(
        self, privacy_measures: Dict[str, Any]
    ) -> EvaluationResult:
        score = 0.0
        missing_required: List[str] = []
        for measure, config in REQUIRED_PRIVACY_MEASURES.items():
            if privacy_measures.get(measure):
                score += config["weight"]
            elif config["required"]:
                missing_required.append(measure)
        return EvaluationResult(
            requirement=self.requirements[1],
            score=score,
            confidence=0.85,
            evidence=[f"Missing required measures: {missing_required}"],
            recommendations=self._privacy_recommendations(missing_required),
        )

    def _evaluate_data_protection(
        self, data_flow: Dict[str, Any]
    ) -> EvaluationResult:
        controls = {
            "transit_encryption": bool(data_flow.get("transit_encryption")),
            "at_rest_encryption": bool(data_flow.get("at_rest_encryption")),
            "documented_lineage": bool(data_flow.get("documented_lineage")),
            "third_party_review": bool(data_flow.get("third_party_review")),
        }
        score = sum(controls.values()) / len(controls)
        missing = [name for name, ok in controls.items() if not ok]
        recommendations: List[str] = []
        if missing:
            recommendations.append(
                "Close gaps in: " + ", ".join(missing)
                + " (Article 10(5); GDPR Article 32)."
            )
        return EvaluationResult(
            requirement=self.requirements[2],
            score=score,
            confidence=0.85,
            evidence=[f"Control status: {controls}"],
            recommendations=recommendations,
        )

    @staticmethod
    def _minimization_recommendations(unnecessary: List[str]) -> List[str]:
        if not unnecessary:
            return []
        return [
            "Stop collecting fields that are not strictly necessary: "
            + ", ".join(unnecessary)
            + " (Article 10(2))."
        ]

    @staticmethod
    def _privacy_recommendations(missing_required: List[str]) -> List[str]:
        if not missing_required:
            return []
        return [
            "Implement required privacy-by-design measures: "
            + ", ".join(missing_required)
            + "."
        ]
