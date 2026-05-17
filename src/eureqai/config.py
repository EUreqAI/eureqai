"""Pydantic schema for the ``eureqai assess`` project YAML.

The schema is intentionally permissive: developers should be able to
describe a system in a few minutes. Unknown fields under ``documents``
and ``answers`` are accepted as free-form evidence pointers and
checklist responses respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from eureqai.checklist import ANSWER_SCORE

Role = Literal["provider", "deployer", "importer", "distributor", "gpai_provider"]
RiskClass = Literal[
    "prohibited", "high_risk", "gpai", "limited_risk", "minimal_risk", "unknown"
]


class ProjectInfo(BaseModel):
    """Top-level identifying information for the AI system."""

    name: str
    version: str = "0.0.1"
    organisation: Optional[str] = None
    intended_purpose: str
    deployment_context: Optional[str] = None


class Classification(BaseModel):
    """How the system is classified under the AI Act."""

    role: Role
    risk_class: RiskClass = "unknown"
    annex_iii_categories: List[str] = Field(default_factory=list)


class ProjectConfig(BaseModel):
    """Root config object loaded from the project YAML."""

    project: ProjectInfo
    classification: Classification
    answers: Dict[str, Any] = Field(default_factory=dict)
    notes: Dict[str, str] = Field(default_factory=dict)
    documents: Dict[str, str] = Field(default_factory=dict)

    @field_validator("answers", mode="before")
    @classmethod
    def _validate_answers(cls, value: Dict[str, Any]) -> Dict[str, str]:
        # PyYAML parses bare `yes`/`no`/`on`/`off` as booleans under YAML 1.1.
        # Accept those so developers don't need to quote everything.
        normalised: Dict[str, str] = {}
        for item_id, answer in value.items():
            if isinstance(answer, bool):
                answer = "yes" if answer else "no"
            else:
                answer = str(answer).strip().lower()
            if answer not in ANSWER_SCORE:
                raise ValueError(
                    f"answer for {item_id!r} must be one of "
                    f"{sorted(ANSWER_SCORE)}, got {answer!r}"
                )
            normalised[item_id] = answer
        return normalised


def load_config(path: Path | str) -> ProjectConfig:
    """Load and validate a project YAML."""
    raw: Any
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return ProjectConfig(**raw)
