"""Declarative readiness checklist.

Each item maps a yes/no/partial answer about a developer's AI system to a
specific obligation in Regulation (EU) 2024/1689. Items are grouped by
checklist area and weighted by priority so the report can produce a
per-area and overall readiness score.

This is the data model behind ``eureqai assess --config myproject.yml`` —
developers describe their system once and get a readiness report back,
rather than feeding model responses into the evaluator API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Priority = str  # "critical" | "high" | "medium" | "low"
Answer = str  # "yes" | "no" | "partial" | "na"


PRIORITY_WEIGHT: Dict[Priority, float] = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}

ANSWER_SCORE: Dict[Answer, float] = {
    "yes": 1.0,
    "partial": 0.5,
    "no": 0.0,
    "na": 1.0,  # treat documented-not-applicable as fully addressed
}


@dataclass(frozen=True)
class ChecklistItem:
    """A single readiness question."""

    id: str
    area: str
    question: str
    article: str
    priority: Priority
    guidance: str = ""
    # Optional path under the YAML to surface as evidence (e.g. "documents.dpia").
    evidence_field: str = ""


@dataclass
class AreaScore:
    """Per-area readiness rollup."""

    area: str
    score: float
    items: List[Tuple[ChecklistItem, Answer, str]] = field(default_factory=list)

    @property
    def compliance_level(self) -> str:
        if self.score >= 0.8:
            return "ready"
        if self.score >= 0.6:
            return "gaps"
        return "not_ready"


CATALOGUE: List[ChecklistItem] = [
    # --- Prohibited practices (Art. 5) — applicable since 2 Feb 2025
    ChecklistItem(
        id="PROHIB-1",
        area="Prohibited practices",
        question=(
            "Have you confirmed the system does NOT engage in any practice "
            "prohibited by Article 5 (subliminal manipulation, exploitation "
            "of vulnerabilities, social scoring, untargeted scraping of "
            "facial images, emotion recognition in workplace/education, "
            "biometric categorisation for protected attributes, real-time "
            "remote biometric ID in public spaces, predictive policing "
            "based solely on profiling)?"
        ),
        article="Article 5",
        priority="critical",
        guidance=(
            "If any of the listed practices is even arguably present, the "
            "system cannot be placed on the EU market — full stop. Document "
            "the analysis that rules each prohibition out."
        ),
    ),
    # --- AI literacy (Art. 4) — applicable since 2 Feb 2025
    ChecklistItem(
        id="LIT-1",
        area="AI literacy",
        question=(
            "Do staff who operate or are affected by the AI system receive "
            "AI-literacy training appropriate to their role (Article 4)?"
        ),
        article="Article 4",
        priority="high",
        evidence_field="documents.ai_literacy_programme",
    ),
    # --- Risk classification — Annex III gateway
    ChecklistItem(
        id="RISK-1",
        area="Risk classification",
        question=(
            "Have you documented whether the system is high-risk under "
            "Article 6 / Annex III, GPAI under Article 51, limited-risk "
            "under Article 50, or minimal-risk?"
        ),
        article="Articles 6, 50, 51; Annex III",
        priority="critical",
        evidence_field="documents.risk_classification",
    ),
    # --- Data governance (Art. 10)
    ChecklistItem(
        id="DATA-1",
        area="Data governance",
        question=(
            "Do training, validation and testing datasets meet the quality "
            "criteria in Article 10(2)–(3): relevance, representativeness, "
            "freedom from errors, and bias examination?"
        ),
        article="Article 10(2)–(3)",
        priority="high",
        evidence_field="documents.data_governance_policy",
    ),
    ChecklistItem(
        id="DATA-2",
        area="Data governance",
        question=(
            "Is data lineage documented and is collection minimised to what "
            "is strictly necessary for the intended purpose?"
        ),
        article="Article 10(2)(b); GDPR Article 5(1)(c)",
        priority="high",
    ),
    # --- Technical documentation (Art. 11 + Annex IV)
    ChecklistItem(
        id="DOC-1",
        area="Technical documentation",
        question=(
            "For a high-risk system, is technical documentation prepared in "
            "line with Annex IV (general description, design choices, "
            "datasets, validation, monitoring) before placing on the market?"
        ),
        article="Article 11; Annex IV",
        priority="critical",
        evidence_field="documents.technical_documentation",
    ),
    # --- Record-keeping / logging (Art. 12)
    ChecklistItem(
        id="LOG-1",
        area="Record-keeping",
        question=(
            "Does the system automatically log events sufficient to ensure "
            "traceability over its lifecycle (Article 12)?"
        ),
        article="Article 12",
        priority="high",
    ),
    # --- Transparency to deployers / users (Art. 13, 50)
    ChecklistItem(
        id="TRANS-1",
        area="Transparency",
        question=(
            "Are end users informed when interacting with an AI system, "
            "and is AI-generated/manipulated content marked as such where "
            "Article 50 requires it?"
        ),
        article="Article 50",
        priority="critical",
    ),
    ChecklistItem(
        id="TRANS-2",
        area="Transparency",
        question=(
            "Do deployers receive instructions for use covering intended "
            "purpose, accuracy/robustness levels, foreseeable misuse and "
            "limitations (Article 13)?"
        ),
        article="Article 13",
        priority="high",
        evidence_field="documents.instructions_for_use",
    ),
    # --- Human oversight (Art. 14)
    ChecklistItem(
        id="OVER-1",
        area="Human oversight",
        question=(
            "Are human-oversight measures designed in: ability to monitor, "
            "interpret outputs, intervene and stop the system (Article 14)?"
        ),
        article="Article 14",
        priority="critical",
    ),
    # --- Accuracy, robustness and cybersecurity (Art. 15)
    ChecklistItem(
        id="TECH-1",
        area="Technical robustness",
        question=(
            "Are accuracy, robustness and cybersecurity levels declared and "
            "tested, including resilience to adversarial inputs (Article 15)?"
        ),
        article="Article 15",
        priority="critical",
    ),
    # --- Quality management & post-market monitoring (Art. 17, 72)
    ChecklistItem(
        id="QMS-1",
        area="Governance",
        question=(
            "For high-risk: is a quality management system in place "
            "(Article 17), and a post-market monitoring plan (Article 72)?"
        ),
        article="Articles 17, 72",
        priority="high",
    ),
    # --- GPAI (Art. 53–55)
    ChecklistItem(
        id="GPAI-1",
        area="GPAI obligations",
        question=(
            "If providing a GPAI model: technical documentation, copyright "
            "policy and a sufficiently detailed summary of training content "
            "(Article 53) — and systemic-risk evaluations under Article 55 "
            "where the model meets the Article 51 threshold?"
        ),
        article="Articles 51, 53, 55",
        priority="critical",
    ),
    # --- Fundamental rights impact assessment (Art. 27) — for certain deployers
    ChecklistItem(
        id="FRIA-1",
        area="Fundamental rights",
        question=(
            "If you are a deployer covered by Article 27 (public bodies and "
            "certain banking/insurance use cases), have you carried out a "
            "fundamental rights impact assessment?"
        ),
        article="Article 27",
        priority="high",
        evidence_field="documents.fria",
    ),
]


def items_for_role(role: str) -> List[ChecklistItem]:
    """Filter the catalogue by the user's role under the AI Act.

    Providers carry the heavy obligations; deployers, importers and
    distributors carry a subset. We keep the model simple: everyone sees
    the cross-cutting items, providers see everything, deployers also see
    FRIA and instructions-for-use receipt.
    """
    role = role.lower()
    if role == "provider":
        return list(CATALOGUE)
    if role == "deployer":
        deployer_areas = {
            "Prohibited practices",
            "AI literacy",
            "Risk classification",
            "Transparency",
            "Human oversight",
            "Fundamental rights",
            "Governance",
        }
        return [item for item in CATALOGUE if item.area in deployer_areas]
    if role in {"importer", "distributor"}:
        return [
            item
            for item in CATALOGUE
            if item.area
            in {"Prohibited practices", "Technical documentation", "Transparency"}
        ]
    if role == "gpai_provider":
        return [item for item in CATALOGUE if item.area != "Fundamental rights"]
    return list(CATALOGUE)


def score_area(items: List[Tuple[ChecklistItem, Answer, str]]) -> float:
    """Weighted score for a single area's answers."""
    if not items:
        return 0.0
    weighted_sum = 0.0
    total_weight = 0.0
    for item, answer, _ in items:
        weight = PRIORITY_WEIGHT[item.priority]
        weighted_sum += ANSWER_SCORE[answer] * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0
