"""Prohibited practices screening (Article 5).

Article 5(1) of Regulation (EU) 2024/1689 lists eight practices that may
not be placed on the Union market, put into service, or used. The
prohibitions have been applicable since **2 February 2025**.

Unlike the other evaluators in this package, this is a **hard gate**: any
practice that is present without a documented carve-out makes the entire
system unlawful, regardless of how well the rest of the AI Act
obligations are met. Scores here are therefore binary in spirit (1.0 =
clean, 0.0 = blocker) with a 0.5 middle band reserved for two cases:

* the developer has declared the practice as present but documented a
  carve-out — must be justified rigorously and reviewed by counsel;
* the developer is "unclear" — equivalent to a blocker pending review.

This module is *not legal advice*. Carve-outs in Article 5 are narrow,
heavily conditioned (e.g. prior judicial authorisation for real-time
remote biometric identification under Article 5(3)–(5)), and benefit from
expert review.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from eureqai.evaluators.base import (
    BaseEvaluator,
    EvaluationResult,
    Requirement,
)


# Allowed declarations.
_VALID_DECLARATIONS = {"no", "yes", "unclear"}


# (id, name, article, default_guidance, allowed_carve_outs_keys)
_PRACTICES: List[Tuple[str, str, str, str, Tuple[str, ...]]] = [
    (
        "subliminal_manipulation",
        "Subliminal / manipulative / deceptive techniques",
        "Article 5(1)(a)",
        (
            "Banned where the system materially distorts behaviour by "
            "circumventing awareness and causes or is reasonably likely "
            "to cause significant harm. No carve-out."
        ),
        (),
    ),
    (
        "vulnerability_exploitation",
        "Exploitation of vulnerabilities",
        "Article 5(1)(b)",
        (
            "Banned where the system exploits vulnerabilities of a person "
            "or group due to age, disability, or a specific social or "
            "economic situation, and causes or is reasonably likely to "
            "cause significant harm. No carve-out."
        ),
        (),
    ),
    (
        "social_scoring",
        "Social scoring leading to detrimental treatment",
        "Article 5(1)(c)",
        (
            "Banned where natural persons or groups are evaluated or "
            "classified over time based on social behaviour or known / "
            "predicted personal characteristics, and the resulting score "
            "leads to detrimental or unfavourable treatment in unrelated "
            "contexts or treatment that is unjustified or disproportionate. "
            "No carve-out."
        ),
        (),
    ),
    (
        "predictive_policing_profiling",
        "Predictive policing based solely on profiling",
        "Article 5(1)(d)",
        (
            "Banned where the system is used to make risk assessments of "
            "natural persons in order to predict criminal offences based "
            "solely on profiling or assessing personality traits. Permitted "
            "only when supporting a human assessment already based on "
            "objective and verifiable facts directly linked to a criminal "
            "activity."
        ),
        ("human_assessment_with_objective_facts",),
    ),
    (
        "untargeted_facial_scraping",
        "Untargeted scraping of facial images",
        "Article 5(1)(e)",
        (
            "Banned where the system creates or expands facial recognition "
            "databases through untargeted scraping of facial images from "
            "the internet or CCTV footage. No carve-out."
        ),
        (),
    ),
    (
        "emotion_recognition_workplace_education",
        "Emotion recognition in workplace or education",
        "Article 5(1)(f)",
        (
            "Banned where the system infers emotions of a natural person "
            "in the area of workplace or education. Permitted only when "
            "intended to be put in place for medical or safety reasons."
        ),
        ("medical_or_safety_documented",),
    ),
    (
        "biometric_categorisation_protected",
        "Biometric categorisation by protected attribute",
        "Article 5(1)(g)",
        (
            "Banned where the system categorises natural persons based on "
            "biometric data to deduce or infer race, political opinions, "
            "trade union membership, religious or philosophical beliefs, "
            "sex life or sexual orientation. Permitted only for labelling "
            "or filtering of lawfully acquired biometric datasets in the "
            "area of law enforcement."
        ),
        ("law_enforcement_filtering",),
    ),
    (
        "realtime_remote_biometric_id_le",
        "Real-time remote biometric identification by law enforcement",
        "Article 5(1)(h)",
        (
            "Banned in publicly accessible spaces for the purpose of law "
            "enforcement, except for narrow situations (targeted search "
            "for victims; prevention of substantial threat to life or "
            "terrorist attack; localising suspects of listed serious "
            "offences) and only with prior authorisation under "
            "Article 5(3)–(5)."
        ),
        ("authorisation_documented", "purpose_in_listed_exceptions"),
    ),
]


class ProhibitedPracticesEvaluator(BaseEvaluator):
    """Hard-gate screener for Article 5 prohibited practices."""

    def _initialize_requirements(self) -> None:
        self.requirements = [
            Requirement(
                id=f"PROHIB-{i + 1}",
                name=name,
                description=guidance,
                article=article,
                priority="critical",
                category="Prohibited practices",
            )
            for i, (_key, name, article, guidance, _carve_outs) in enumerate(
                _PRACTICES
            )
        ]
        # Map practice key → requirement so evaluate() can look up cleanly.
        self._req_by_key: Dict[str, Requirement] = {
            key: self.requirements[i]
            for i, (key, *_rest) in enumerate(_PRACTICES)
        }
        self._carve_outs_by_key: Dict[str, Tuple[str, ...]] = {
            key: carve_outs
            for key, _name, _article, _guidance, carve_outs in _PRACTICES
        }

    @classmethod
    def practice_keys(cls) -> List[str]:
        """The canonical list of practice keys accepted by ``evaluate``."""
        return [key for key, *_rest in _PRACTICES]

    def evaluate(
        self,
        declarations: Optional[Dict[str, str]] = None,
        carve_outs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Screen the system against the eight Article 5 prohibitions.

        Args:
            declarations: Mapping from practice key (see
                :meth:`practice_keys`) to one of ``"no"`` (practice is
                not present), ``"yes"`` (practice is present), or
                ``"unclear"`` (developer cannot yet say). Missing keys
                are treated as ``"unclear"`` and flagged.
            carve_outs: Mapping from practice key to a dict of carve-out
                flags relevant to that practice. Only Article 5(1)(d),
                (f), (g) and (h) have any lawful carve-outs; everything
                else is prohibited absolutely.

        Returns:
            One :class:`EvaluationResult` per practice, in catalogue
            order. Score is ``1.0`` for a clean "no", ``0.0`` for a
            blocker, ``0.5`` for a documented carve-out (must still be
            reviewed) or an "unclear" answer.
        """
        declarations = {
            k: v.strip().lower() if isinstance(v, str) else v
            for k, v in (declarations or {}).items()
        }
        carve_outs = carve_outs or {}
        self._validate_declarations(declarations)

        results: List[EvaluationResult] = []
        for key, _name, article, _guidance, allowed_carve_outs in _PRACTICES:
            declaration = declarations.get(key, "unclear")
            practice_carve_outs = carve_outs.get(key, {}) or {}
            result = self._evaluate_practice(
                req=self._req_by_key[key],
                declaration=declaration,
                article=article,
                allowed_carve_outs=allowed_carve_outs,
                carve_outs_supplied=practice_carve_outs,
            )
            results.append(result)
            self.results.append(result)
        return results

    @property
    def blockers(self) -> List[EvaluationResult]:
        """Subset of ``self.results`` that constitute hard blockers."""
        return [r for r in self.results if r.score < 1.0]

    @staticmethod
    def _validate_declarations(declarations: Dict[str, Any]) -> None:
        known: Set[str] = {key for key, *_rest in _PRACTICES}
        for key, value in declarations.items():
            if key not in known:
                raise ValueError(
                    f"unknown practice key {key!r}; expected one of "
                    f"{sorted(known)}"
                )
            if value not in _VALID_DECLARATIONS:
                raise ValueError(
                    f"declaration for {key!r} must be one of "
                    f"{sorted(_VALID_DECLARATIONS)}, got {value!r}"
                )

    def _evaluate_practice(
        self,
        req: Requirement,
        declaration: str,
        article: str,
        allowed_carve_outs: Tuple[str, ...],
        carve_outs_supplied: Dict[str, Any],
    ) -> EvaluationResult:
        if declaration == "no":
            return EvaluationResult(
                requirement=req,
                score=1.0,
                confidence=0.85,
                evidence=[
                    f"Declared as not present ({article})."
                ],
                recommendations=[],
                metadata={"declaration": "no"},
            )

        if declaration == "unclear":
            return EvaluationResult(
                requirement=req,
                score=0.5,
                confidence=0.7,
                evidence=[
                    "Developer marked this prohibition as 'unclear'."
                ],
                recommendations=[
                    f"Resolve {article} status with qualified counsel "
                    "before placing the system on the market. Until "
                    "resolved, treat as a blocker."
                ],
                metadata={"declaration": "unclear"},
            )

        # declaration == "yes"
        if not allowed_carve_outs:
            return EvaluationResult(
                requirement=req,
                score=0.0,
                confidence=0.95,
                evidence=[
                    f"Practice declared as present but {article} admits "
                    "no carve-out."
                ],
                recommendations=[
                    f"This is an absolute prohibition under {article}. "
                    "The system cannot be placed on the EU market in its "
                    "current form."
                ],
                metadata={"declaration": "yes", "blocker": True},
            )

        # The practice is present and the article allows narrow carve-outs.
        documented = self._supplied_carve_outs(carve_outs_supplied)
        required = set(allowed_carve_outs)
        if not required.issubset(documented):
            missing = sorted(required - documented)
            return EvaluationResult(
                requirement=req,
                score=0.0,
                confidence=0.9,
                evidence=[
                    "Practice declared as present without complete "
                    f"carve-out evidence. Missing: {missing}."
                ],
                recommendations=[
                    f"Document {', '.join(missing)} or remove the practice "
                    f"from the system ({article})."
                ],
                metadata={
                    "declaration": "yes",
                    "blocker": True,
                    "missing_carve_outs": missing,
                },
            )

        # All required carve-out keys are present — still recommend review.
        return EvaluationResult(
            requirement=req,
            score=0.5,
            confidence=0.75,
            evidence=[
                f"Practice declared as present with carve-outs documented: "
                f"{sorted(documented)}."
            ],
            recommendations=[
                f"Carve-outs under {article} are narrow. Have qualified "
                "counsel confirm each documented element holds in "
                "practice before relying on the exemption."
            ],
            metadata={
                "declaration": "yes",
                "carve_outs_documented": sorted(documented),
            },
        )

    @staticmethod
    def _supplied_carve_outs(carve_outs_supplied: Dict[str, Any]) -> Set[str]:
        present: Set[str] = set()
        for key, value in carve_outs_supplied.items():
            if value is True or (isinstance(value, str) and value.strip()):
                present.add(key)
        return present


def screen_blockers(results: Iterable[EvaluationResult]) -> List[EvaluationResult]:
    """Convenience: extract hard blockers (score < 1.0) from any evaluator output.

    Useful for callers that want a single boolean gate, e.g. in CI:

        if screen_blockers(evaluator.results):
            sys.exit(1)
    """
    return [r for r in results if r.score < 1.0]
