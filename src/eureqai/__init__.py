"""EUreqAI — EU AI Act readiness assessment framework.

References the final Regulation (EU) 2024/1689 of the European Parliament and
of the Council of 13 June 2024 (the "AI Act").
"""

from eureqai.evaluators.base import (
    BaseEvaluator,
    EvaluationResult,
    Requirement,
)
from eureqai.evaluators.fairness import FairnessEvaluator
from eureqai.evaluators.privacy import PrivacyEvaluator
from eureqai.evaluators.technical_robustness import TechnicalRobustnessEvaluator
from eureqai.evaluators.transparency import TransparencyEvaluator

__version__ = "0.1.0"

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "Requirement",
    "FairnessEvaluator",
    "PrivacyEvaluator",
    "TechnicalRobustnessEvaluator",
    "TransparencyEvaluator",
    "__version__",
]
