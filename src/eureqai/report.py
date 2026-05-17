"""Render a readiness report from a project config + checklist answers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from eureqai.checklist import (
    AreaScore,
    ChecklistItem,
    items_for_role,
    score_area,
)
from eureqai.config import ProjectConfig


@dataclass
class Assessment:
    """Result of running a project config against the checklist catalogue."""

    config: ProjectConfig
    areas: List[AreaScore]
    unanswered: List[ChecklistItem]

    @property
    def overall_score(self) -> float:
        if not self.areas:
            return 0.0
        return sum(a.score for a in self.areas) / len(self.areas)

    @property
    def overall_level(self) -> str:
        if any(
            answer == "no" and item.priority == "critical"
            for area in self.areas
            for item, answer, _ in area.items
        ):
            return "not_ready"
        score = self.overall_score
        if score >= 0.8:
            return "ready"
        if score >= 0.6:
            return "gaps"
        return "not_ready"


def assess(config: ProjectConfig) -> Assessment:
    """Run the checklist for a project config."""
    items = items_for_role(config.classification.role)
    grouped: Dict[str, List[Tuple[ChecklistItem, str, str]]] = {}
    unanswered: List[ChecklistItem] = []

    for item in items:
        answer = config.answers.get(item.id)
        if answer is None:
            unanswered.append(item)
            continue
        note = config.notes.get(item.id, "")
        grouped.setdefault(item.area, []).append((item, answer, note))

    areas = [
        AreaScore(area=area, score=score_area(answers), items=answers)
        for area, answers in grouped.items()
    ]
    return Assessment(config=config, areas=areas, unanswered=unanswered)


_LEVEL_BADGE = {
    "ready": "🟢 Ready",
    "gaps": "🟡 Gaps to close",
    "not_ready": "🔴 Not ready",
}


def render_markdown(assessment: Assessment) -> str:
    """Render the assessment as a Markdown report."""
    cfg = assessment.config
    lines: List[str] = []
    lines.append(f"# AI Act readiness report — {cfg.project.name}")
    lines.append("")
    lines.append(
        f"_Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} "
        f"by EUreqAI._"
    )
    lines.append("")
    lines.append("## Project")
    lines.append("")
    lines.append(f"- **Version**: {cfg.project.version}")
    if cfg.project.organisation:
        lines.append(f"- **Organisation**: {cfg.project.organisation}")
    lines.append(f"- **Intended purpose**: {cfg.project.intended_purpose}")
    if cfg.project.deployment_context:
        lines.append(f"- **Deployment context**: {cfg.project.deployment_context}")
    lines.append(f"- **Role under the AI Act**: `{cfg.classification.role}`")
    lines.append(f"- **Risk classification**: `{cfg.classification.risk_class}`")
    if cfg.classification.annex_iii_categories:
        cats = ", ".join(cfg.classification.annex_iii_categories)
        lines.append(f"- **Annex III categories**: {cats}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    badge = _LEVEL_BADGE[assessment.overall_level]
    lines.append(
        f"**Overall readiness**: {badge} "
        f"(score {assessment.overall_score:.2f} / 1.00)"
    )
    lines.append("")
    if assessment.areas:
        lines.append("| Area | Score | Status |")
        lines.append("| ---- | ----- | ------ |")
        for area in assessment.areas:
            lines.append(
                f"| {area.area} | {area.score:.2f} | "
                f"{_LEVEL_BADGE[area.compliance_level]} |"
            )
        lines.append("")

    critical_blockers = [
        (area, item, note)
        for area in assessment.areas
        for item, answer, note in area.items
        if answer == "no" and item.priority == "critical"
    ]
    if critical_blockers:
        lines.append("## ⛔ Critical blockers")
        lines.append("")
        for _, item, note in critical_blockers:
            lines.append(f"- **[{item.id}] {item.area}** — {item.article}")
            lines.append(f"  - {item.question}")
            if item.guidance:
                lines.append(f"  - _Guidance_: {item.guidance}")
            if note:
                lines.append(f"  - _Note_: {note}")
        lines.append("")

    lines.append("## Detailed findings")
    lines.append("")
    for area in assessment.areas:
        lines.append(f"### {area.area} — {_LEVEL_BADGE[area.compliance_level]}")
        lines.append("")
        for item, answer, note in area.items:
            emoji = _answer_emoji(answer)
            lines.append(
                f"- {emoji} **[{item.id}]** ({item.article}, "
                f"{item.priority}) — {item.question}"
            )
            evidence = cfg.documents.get(item.evidence_field.split(".")[-1]) if (
                item.evidence_field
            ) else None
            if evidence:
                lines.append(f"  - _Evidence_: `{evidence}`")
            if note:
                lines.append(f"  - _Note_: {note}")
        lines.append("")

    if assessment.unanswered:
        lines.append("## ❓ Unanswered checklist items")
        lines.append("")
        lines.append(
            "These items apply to your role under the AI Act but were not "
            "addressed in the config. Each should resolve to `yes`, `no`, "
            "`partial`, or `na` before publishing the report."
        )
        lines.append("")
        for item in assessment.unanswered:
            lines.append(
                f"- **[{item.id}] {item.area}** — {item.article}: "
                f"{item.question}"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_This report is an engineering aid, not legal advice. Validate with "
        "qualified counsel before relying on it for conformity decisions._"
    )
    lines.append("")
    return "\n".join(lines)


def _answer_emoji(answer: str) -> str:
    return {
        "yes": "✅",
        "partial": "🟡",
        "no": "❌",
        "na": "⚪",
    }.get(answer, "❓")
