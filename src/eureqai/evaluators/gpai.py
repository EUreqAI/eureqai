"""General-purpose AI (GPAI) model obligations evaluator.

Maps to Regulation (EU) 2024/1689 Articles 51–55:

* Article 51 — Classification of GPAI models as having systemic risk.
  Presumption when cumulative training compute exceeds 10^25 FLOPs.
* Article 53 — Obligations for *all* providers of GPAI models:
    (a) technical documentation per Annex XI
    (b) information made available to downstream providers per Annex XII
    (c) copyright policy aligned with Directive (EU) 2019/790
    (d) sufficiently detailed public summary of training content
  Open-source models that meet Article 53(2) conditions are exempt from
  (a) and (b) but never from (c) and (d), and never if they have
  systemic risk.
* Article 55 — Additional obligations for GPAI with systemic risk:
    (a) state-of-the-art model evaluation incl. adversarial testing
    (b) systemic risk assessment and mitigation at Union level
    (c) tracking, documentation and reporting of serious incidents
    (d) adequate cybersecurity protection
* Article 56 — Providers can rely on codes of practice (e.g. the GPAI
  Code of Practice published in 2025) to demonstrate compliance.

These obligations have been applicable since 2 August 2025.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from eureqai.evaluators.base import (
    BaseEvaluator,
    EvaluationResult,
    Requirement,
)

# Article 51(2) presumption threshold for systemic risk.
SYSTEMIC_RISK_FLOPS_THRESHOLD = 1e25


class GPAIEvaluator(BaseEvaluator):
    """Evaluates GPAI provider obligations (Articles 51, 53, 55)."""

    def _initialize_requirements(self) -> None:
        self.requirements = [
            Requirement(
                id="GPAI-CLASS",
                name="Systemic-risk classification",
                description=(
                    "Provider has assessed whether the model meets the "
                    "Article 51 systemic-risk presumption (>10^25 FLOPs "
                    "training compute) or is otherwise designated."
                ),
                article="Article 51",
                priority="critical",
                category="GPAI classification",
            ),
            Requirement(
                id="GPAI-DOC",
                name="Technical documentation (Annex XI)",
                description=(
                    "Up-to-date technical documentation drawn up before "
                    "placing on the market and made available to the AI "
                    "Office and national competent authorities on request."
                ),
                article="Article 53(1)(a); Annex XI",
                priority="critical",
                category="GPAI documentation",
            ),
            Requirement(
                id="GPAI-DOWN",
                name="Information for downstream providers (Annex XII)",
                description=(
                    "Information enabling downstream providers to "
                    "integrate the model and comply with their own "
                    "obligations is drawn up, kept current and made "
                    "available."
                ),
                article="Article 53(1)(b); Annex XII",
                priority="high",
                category="GPAI documentation",
            ),
            Requirement(
                id="GPAI-COPYRIGHT",
                name="Copyright policy",
                description=(
                    "Provider has a policy to comply with EU copyright "
                    "law, in particular to identify and respect "
                    "Article 4(3) Directive (EU) 2019/790 opt-outs."
                ),
                article="Article 53(1)(c)",
                priority="critical",
                category="GPAI obligations",
            ),
            Requirement(
                id="GPAI-SUMMARY",
                name="Public training-data summary",
                description=(
                    "Sufficiently detailed summary of the content used to "
                    "train the model is drawn up and made publicly "
                    "available, following the AI Office's template."
                ),
                article="Article 53(1)(d)",
                priority="critical",
                category="GPAI obligations",
            ),
            Requirement(
                id="GPAI-EVAL",
                name="Model evaluation and adversarial testing",
                description=(
                    "For systemic-risk models: state-of-the-art "
                    "evaluations including standardised adversarial "
                    "testing, with results documented."
                ),
                article="Article 55(1)(a)",
                priority="critical",
                category="GPAI systemic risk",
            ),
            Requirement(
                id="GPAI-RISK",
                name="Systemic-risk assessment and mitigation",
                description=(
                    "For systemic-risk models: assessment and mitigation "
                    "of possible systemic risks at Union level, "
                    "including the sources of those risks."
                ),
                article="Article 55(1)(b)",
                priority="critical",
                category="GPAI systemic risk",
            ),
            Requirement(
                id="GPAI-INCIDENT",
                name="Serious-incident reporting",
                description=(
                    "For systemic-risk models: process to track, "
                    "document and report serious incidents and possible "
                    "corrective measures to the AI Office and, where "
                    "relevant, national competent authorities."
                ),
                article="Article 55(1)(c)",
                priority="critical",
                category="GPAI systemic risk",
            ),
            Requirement(
                id="GPAI-CYBER",
                name="Cybersecurity protection",
                description=(
                    "For systemic-risk models: adequate level of "
                    "cybersecurity protection for the model and its "
                    "physical infrastructure."
                ),
                article="Article 55(1)(d)",
                priority="high",
                category="GPAI systemic risk",
            ),
        ]

    def evaluate(
        self,
        training_compute_flops: Optional[float] = None,
        designated_systemic_risk: bool = False,
        is_open_source: bool = False,
        documentation: Optional[Dict[str, Any]] = None,
        systemic_risk_measures: Optional[Dict[str, Any]] = None,
        relies_on_code_of_practice: bool = False,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Evaluate GPAI provider obligations.

        Args:
            training_compute_flops: Cumulative training compute in FLOPs.
                Triggers the Article 51 systemic-risk presumption above
                ``SYSTEMIC_RISK_FLOPS_THRESHOLD``.
            designated_systemic_risk: True if the Commission has
                designated the model as systemic-risk regardless of
                compute.
            is_open_source: True if the model meets the Article 53(2)
                free-and-open-source release conditions. Only exempts
                from 53(1)(a)–(b), and only when the model is NOT
                systemic-risk.
            documentation: dict with boolean / string fields:
                ``technical_documentation_annex_xi``,
                ``downstream_information_annex_xii``,
                ``copyright_policy``,
                ``training_data_summary`` (URL/path is best).
            systemic_risk_measures: dict applicable to systemic-risk
                providers: ``model_evaluation``, ``adversarial_testing``,
                ``risk_assessment``, ``incident_reporting_process``,
                ``cybersecurity_measures``.
            relies_on_code_of_practice: True if the provider relies on
                the GPAI Code of Practice (Article 56). Adds a small
                confidence boost.
        """
        documentation = documentation or {}
        systemic_risk_measures = systemic_risk_measures or {}

        has_systemic_risk = bool(designated_systemic_risk) or (
            training_compute_flops is not None
            and training_compute_flops >= SYSTEMIC_RISK_FLOPS_THRESHOLD
        )
        oss_exempts_documentation = is_open_source and not has_systemic_risk
        confidence_boost = 0.05 if relies_on_code_of_practice else 0.0

        results: List[EvaluationResult] = []
        classification_result = self._evaluate_classification(
            training_compute_flops, designated_systemic_risk, has_systemic_risk
        )
        results.append(classification_result)
        self.results.append(classification_result)

        # Article 53 obligations.
        for result in (
            self._evaluate_documentation(
                req_id="GPAI-DOC",
                key="technical_documentation_annex_xi",
                documentation=documentation,
                exempt=oss_exempts_documentation,
                exemption_note=(
                    "Exempted by Article 53(2) (open-source release "
                    "without systemic risk)."
                ),
            ),
            self._evaluate_documentation(
                req_id="GPAI-DOWN",
                key="downstream_information_annex_xii",
                documentation=documentation,
                exempt=oss_exempts_documentation,
                exemption_note=(
                    "Exempted by Article 53(2) (open-source release "
                    "without systemic risk)."
                ),
            ),
            self._evaluate_documentation(
                req_id="GPAI-COPYRIGHT",
                key="copyright_policy",
                documentation=documentation,
                exempt=False,
            ),
            self._evaluate_training_summary(documentation),
        ):
            result.confidence = min(1.0, result.confidence + confidence_boost)
            results.append(result)
            self.results.append(result)

        # Article 55 obligations only apply to systemic-risk providers.
        systemic_results = self._evaluate_systemic_risk_measures(
            has_systemic_risk=has_systemic_risk,
            measures=systemic_risk_measures,
        )
        for result in systemic_results:
            result.confidence = min(1.0, result.confidence + confidence_boost)
            results.append(result)
            self.results.append(result)

        return results

    def _evaluate_classification(
        self,
        training_compute_flops: Optional[float],
        designated_systemic_risk: bool,
        has_systemic_risk: bool,
    ) -> EvaluationResult:
        req = self._requirement("GPAI-CLASS")
        evidence: List[str] = []
        if training_compute_flops is None and not designated_systemic_risk:
            return EvaluationResult(
                requirement=req,
                score=0.0,
                confidence=0.6,
                evidence=[
                    "Training compute not declared and no Commission "
                    "designation provided."
                ],
                recommendations=[
                    "Declare cumulative training compute in FLOPs and "
                    "whether the Commission has designated the model "
                    "(Article 52). Without this, classification is "
                    "undocumented."
                ],
            )
        if training_compute_flops is not None:
            evidence.append(
                f"Declared training compute: {training_compute_flops:.2e} "
                "FLOPs"
            )
        if designated_systemic_risk:
            evidence.append("Commission has designated the model (Article 52).")
        if has_systemic_risk:
            evidence.append(
                "Model is treated as a GPAI model with systemic risk; "
                "Article 55 obligations apply."
            )
        else:
            evidence.append(
                "Model is below the Article 51 systemic-risk presumption."
            )
        return EvaluationResult(
            requirement=req,
            score=1.0,
            confidence=0.9,
            evidence=evidence,
            recommendations=[],
            metadata={
                "has_systemic_risk": has_systemic_risk,
                "compute_threshold_flops": SYSTEMIC_RISK_FLOPS_THRESHOLD,
            },
        )

    def _evaluate_documentation(
        self,
        req_id: str,
        key: str,
        documentation: Dict[str, Any],
        exempt: bool,
        exemption_note: str = "",
    ) -> EvaluationResult:
        req = self._requirement(req_id)
        value = documentation.get(key)
        if exempt:
            return EvaluationResult(
                requirement=req,
                score=1.0,
                confidence=0.85,
                evidence=[exemption_note],
                recommendations=[],
                metadata={"exempt": True},
            )
        present = self._is_present(value)
        recommendations: List[str] = []
        if not present:
            recommendations.append(
                f"Prepare {req.name.lower()} before placing the model on the "
                f"market ({req.article})."
            )
        return EvaluationResult(
            requirement=req,
            score=1.0 if present else 0.0,
            confidence=0.85,
            evidence=[
                f"{key} declared: {value!r}" if value is not None
                else f"{key} not declared"
            ],
            recommendations=recommendations,
        )

    def _evaluate_training_summary(
        self, documentation: Dict[str, Any]
    ) -> EvaluationResult:
        req = self._requirement("GPAI-SUMMARY")
        value = documentation.get("training_data_summary")
        present = self._is_present(value)
        public = isinstance(value, str) and (
            value.startswith("http://")
            or value.startswith("https://")
            or value.startswith("doi:")
        )
        if not present:
            return EvaluationResult(
                requirement=req,
                score=0.0,
                confidence=0.85,
                evidence=["No training-data summary declared."],
                recommendations=[
                    "Publish a sufficiently detailed training-data summary "
                    "following the AI Office template (Article 53(1)(d))."
                ],
            )
        score = 1.0 if public else 0.5
        recommendations: List[str] = []
        if not public:
            recommendations.append(
                "Summary is declared but does not look publicly hosted "
                "(no URL/DOI). Article 53(1)(d) requires it to be public."
            )
        return EvaluationResult(
            requirement=req,
            score=score,
            confidence=0.8,
            evidence=[f"training_data_summary declared: {value!r}"],
            recommendations=recommendations,
        )

    def _evaluate_systemic_risk_measures(
        self,
        has_systemic_risk: bool,
        measures: Dict[str, Any],
    ) -> List[EvaluationResult]:
        ids_and_keys = [
            ("GPAI-EVAL", ("model_evaluation", "adversarial_testing")),
            ("GPAI-RISK", ("risk_assessment",)),
            ("GPAI-INCIDENT", ("incident_reporting_process",)),
            ("GPAI-CYBER", ("cybersecurity_measures",)),
        ]
        out: List[EvaluationResult] = []
        for req_id, keys in ids_and_keys:
            req = self._requirement(req_id)
            if not has_systemic_risk:
                out.append(
                    EvaluationResult(
                        requirement=req,
                        score=1.0,
                        confidence=0.85,
                        evidence=[
                            "Article 55 obligation not triggered: model is "
                            "not classified as systemic-risk."
                        ],
                        recommendations=[],
                        metadata={"applicable": False},
                    )
                )
                continue
            satisfied = sum(
                1 for key in keys if self._is_present(measures.get(key))
            )
            score = satisfied / len(keys)
            missing = [key for key in keys if not self._is_present(measures.get(key))]
            recommendations: List[str] = []
            if missing:
                recommendations.append(
                    f"Implement {', '.join(missing)} before placing the "
                    f"systemic-risk model on the market ({req.article})."
                )
            out.append(
                EvaluationResult(
                    requirement=req,
                    score=score,
                    confidence=0.85,
                    evidence=[f"Declared keys: {sorted(measures)}"],
                    recommendations=recommendations,
                    metadata={"applicable": True, "missing": missing},
                )
            )
        return out

    def _requirement(self, req_id: str) -> Requirement:
        for req in self.requirements:
            if req.id == req_id:
                return req
        raise KeyError(req_id)

    @staticmethod
    def _is_present(value: Any) -> bool:
        if value is True:
            return True
        if value is False or value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True
