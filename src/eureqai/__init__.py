"""EUreqAI — EU AI Act readiness assessment framework.

References the final Regulation (EU) 2024/1689 of the European Parliament and
of the Council of 13 June 2024 (the "AI Act").
"""

from eureqai.checklist import CATALOGUE, ChecklistItem, items_for_role
from eureqai.config import ProjectConfig, load_config
from eureqai.evaluators.base import (
    BaseEvaluator,
    EvaluationResult,
    Requirement,
)
from eureqai.evaluators.fairness import FairnessEvaluator
from eureqai.evaluators.gpai import GPAIEvaluator
from eureqai.evaluators.privacy import PrivacyEvaluator
from eureqai.evaluators.prohibited_practices import (
    ProhibitedPracticesEvaluator,
    screen_blockers,
)
from eureqai.evaluators.technical_robustness import TechnicalRobustnessEvaluator
from eureqai.evaluators.transparency import TransparencyEvaluator
from eureqai.report import Assessment, assess, render_markdown

__version__ = "0.2.0"

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "Requirement",
    "FairnessEvaluator",
    "GPAIEvaluator",
    "PrivacyEvaluator",
    "ProhibitedPracticesEvaluator",
    "TechnicalRobustnessEvaluator",
    "TransparencyEvaluator",
    "screen_blockers",
    "Assessment",
    "CATALOGUE",
    "ChecklistItem",
    "ProjectConfig",
    "assess",
    "items_for_role",
    "load_config",
    "render_markdown",
    "__version__",
]
