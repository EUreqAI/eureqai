from .base import BaseMetric, MetricResult
import numpy as np
from sklearn.metrics import confusion_matrix

class DemographicParityMetric(BaseMetric):
    """Measures demographic parity across protected groups."""
    
    def compute(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray
    ) -> MetricResult:
        groups = np.unique(protected_attributes)
        group_rates = []
        
        for group in groups:
            group_mask = (protected_attributes == group)
            group_rate = predictions[group_mask].mean()
            group_rates.append(group_rate)
        
        max_disparity = max(group_rates) - min(group_rates)
        value = 1 - max_disparity  # Convert to [0,1] where 1 is perfect parity
        
        return MetricResult(
            name="demographic_parity",
            value=value,
            submetrics={
                f"group_{g}_rate": r
                for g, r in zip(groups, group_rates)
            }
        )

class EqualOpportunityMetric(BaseMetric):
    """Measures equal opportunity across protected groups."""
    
    def compute(
        self,
        predictions: np.ndarray,
        protected_attributes: np.ndarray,
        ground_truth: np.ndarray
    ) -> MetricResult:
        groups = np.unique(protected_attributes)
        true_positive_rates = []
        
        for group in groups:
            group_mask = (protected_attributes == group)
            group_pred = predictions[group_mask]
            group_truth = ground_truth[group_mask]
            
            tn, fp, fn, tp = confusion_matrix(
                group_truth,
                group_pred
            ).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            true_positive_rates.append(tpr)
        
        max_disparity = max(true_positive_rates) - min(true_positive_rates)
        value = 1 - max_disparity
        
        return MetricResult(
            name="equal_opportunity",
            value=value,
            submetrics={
                f"group_{g}_tpr": r
                for g, r in zip(groups, true_positive_rates)
            }
        )