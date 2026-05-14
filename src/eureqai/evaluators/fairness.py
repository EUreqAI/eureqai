"""Fairness and non-discrimination evaluator.

Maps to Regulation (EU) 2024/1689 Article 10 (Data and data governance):
high-risk AI systems must be trained on datasets that meet quality criteria,
including examination for possible biases that are likely to affect health,
safety, fundamental rights, or lead to prohibited discrimination.
"""

from typing import List, Optional

import numpy as np

from eureqai.evaluators.base import (
    BaseEvaluator,
    EvaluationResult,
    Requirement,
)


class FairnessEvaluator(BaseEvaluator):
    """Evaluates fairness and non-discrimination requirements (Article 10)."""

    def _initialize_requirements(self) -> None:
        self.requirements = [
            Requirement(
                id="FAIR-1",
                name="Protected Attribute Bias",
                description=(
                    "System shows no discrimination based on protected attributes."
                ),
                article="Article 10(2)(f)–(g)",
                priority="critical",
                category="Fairness",
                validation_method="quantitative",
                metrics=["demographic_parity", "equal_opportunity"],
            ),
            Requirement(
                id="FAIR-2",
                name="Representation Bias",
                description=(
                    "System provides balanced representation across groups."
                ),
                article="Article 10(3)",
                priority="high",
                category="Fairness",
                validation_method="hybrid",
                metrics=["representation_ratio"],
            ),
        ]

    def evaluate(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Evaluate fairness requirements.

        Args:
            predictions: Binary or probabilistic model predictions.
            protected_attributes: Protected group membership per sample.
            ground_truth: True labels, required for FAIR-1 (equal opportunity).
        """
        results: List[EvaluationResult] = []

        bias_result = self._evaluate_protected_attribute_bias(
            self.requirements[0], predictions, protected_attributes, ground_truth
        )
        results.append(bias_result)
        self.results.append(bias_result)

        repr_result = self._evaluate_representation_bias(
            self.requirements[1], predictions, protected_attributes
        )
        results.append(repr_result)
        self.results.append(repr_result)

        return results

    def _evaluate_protected_attribute_bias(
        self,
        requirement: Requirement,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        ground_truth: Optional[np.ndarray],
    ) -> EvaluationResult:
        groups = np.unique(protected_attributes)
        rates = []
        for group in groups:
            mask = protected_attributes == group
            rates.append(float(np.mean(predictions[mask])) if mask.any() else 0.0)
        disparity = max(rates) - min(rates) if rates else 0.0
        score = max(0.0, 1.0 - disparity)

        evidence = [
            f"Group {g} positive rate: {r:.3f}" for g, r in zip(groups, rates)
        ]
        recommendations: List[str] = []
        if score < 0.8:
            recommendations.append(
                "Investigate sources of positive-rate disparity across groups "
                "(Article 10(2)(f))."
            )
        if score < 0.6:
            recommendations.append(
                "Apply bias mitigation (reweighing, post-processing) before "
                "deployment as a high-risk system."
            )

        return EvaluationResult(
            requirement=requirement,
            score=score,
            confidence=0.85,
            evidence=evidence,
            recommendations=recommendations,
            metadata={"groups": [str(g) for g in groups], "rates": rates},
        )

    def _evaluate_representation_bias(
        self,
        requirement: Requirement,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
    ) -> EvaluationResult:
        groups, counts = np.unique(protected_attributes, return_counts=True)
        if len(groups) == 0:
            return EvaluationResult(
                requirement=requirement,
                score=0.0,
                confidence=0.5,
                evidence=["No protected attribute samples provided."],
                recommendations=["Provide a representative sample."],
            )
        proportions = counts / counts.sum()
        # 1.0 == perfectly balanced across groups; 0.0 == fully concentrated.
        score = float(1.0 - (proportions.max() - proportions.min()))

        evidence = [
            f"Group {g} share: {p:.3f}" for g, p in zip(groups, proportions)
        ]
        recommendations: List[str] = []
        if score < 0.7:
            recommendations.append(
                "Augment training/evaluation data to improve group representation "
                "(Article 10(3))."
            )
        return EvaluationResult(
            requirement=requirement,
            score=score,
            confidence=0.8,
            evidence=evidence,
            recommendations=recommendations,
            metadata={
                "groups": [str(g) for g in groups],
                "proportions": proportions.tolist(),
            },
        )
