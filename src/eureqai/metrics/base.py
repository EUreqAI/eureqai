"""Base classes for metric computation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class MetricResult:
    """Stores the result of a metric computation."""

    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    submetrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric."""

    def compute_confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute confidence interval for an array of values."""
        mean = np.mean(values)
        ci = stats.t.interval(
            confidence,
            len(values) - 1,
            loc=mean,
            scale=stats.sem(values),
        )
        return (float(ci[0]), float(ci[1]))
