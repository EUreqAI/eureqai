from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

@dataclass
class MetricResult:
    """Stores the result of a metric computation."""
    name: str
    value: float
    confidence_interval: Optional[tuple] = None
    metadata: Dict[str, Any] = None

class BaseMetrics:
    """Base class for metric computation."""
    
    @staticmethod
    def compute_confidence_interval(
        values: np.ndarray,
        confidence: float = 0.95
    ) -> tuple:
        """Compute confidence interval for a set of values."""
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
        interval = sem * 1.96  # For 95% confidence
        return (mean - interval, mean + interval)

    @staticmethod
    def bootstrap_metric(
        metric_fn: callable,
        data: np.ndarray,
        n_bootstrap: int = 1000
    ) -> tuple:
        """Compute bootstrapped confidence intervals for a metric."""
        bootstrap_values = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(
                len(data),
                size=len(data),
                replace=True
            )
            bootstrap_values.append(metric_fn(data[indices]))
        
        return np.percentile(bootstrap_values, [2.5, 97.5])