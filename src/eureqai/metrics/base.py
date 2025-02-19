from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import numpy as np
from scipy import stats

@dataclass
class MetricResult:
    """Stores the result of a metric computation."""
    name: str
    value: float
    confidence_interval: Optional[tuple] = None
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
        pass
    
    def compute_confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95
    ) -> tuple:
        """Compute confidence interval."""
        mean = np.mean(values)
        ci = stats.t.interval(
            confidence,
            len(values)-1,
            loc=mean,
            scale=stats.sem(values)
        )
        return (ci[0], ci[1])