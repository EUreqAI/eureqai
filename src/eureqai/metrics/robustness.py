from .base import BaseMetric, MetricResult
from typing import List, Callable
import numpy as np
from scipy.stats import entropy

class ConsistencyMetric(BaseMetric):
    """Measures consistency of model responses."""
    
    def compute(
        self,
        responses: List[str],
        similar_prompts: List[List[str]]
    ) -> MetricResult:
        """
        Evaluate consistency across similar prompts.
        
        Args:
            responses: List of base responses
            similar_prompts: List of lists of responses to similar prompts
        """
        consistency_scores = []
        
        for base_resp, similar_resps in zip(responses, similar_prompts):
            variations = [base_resp] + similar_resps
            
            # Compute pairwise similarities
            sim_scores = []
            for i in range(len(variations)):
                for j in range(i+1, len(variations)):
                    sim = self._compute_similarity(variations[i], variations[j])
                    sim_scores.append(sim)
            
            consistency_scores.append(np.mean(sim_scores))
        
        value = np.mean(consistency_scores)
        ci = self.compute_confidence_interval(np.array(consistency_scores))
        
        return MetricResult(
            name="consistency",
            value=value,
            confidence_interval=ci,
            submetrics={
                "min_consistency": np.min(consistency_scores),
                "max_consistency": np.max(consistency_scores),
                "std_consistency": np.std(consistency_scores)
            }
        )
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Implement your similarity metric here
        # This is a placeholder implementation
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class StabilityMetric(BaseMetric):
    """Measures stability of model outputs under perturbations."""
    
    def compute(
        self,
        original_outputs: List[Any],
        perturbed_outputs: List[List[Any]],
        perturbation_types: List[str]
    ) -> MetricResult:
        stability_scores = []
        per_type_scores = {ptype: [] for ptype in perturbation_types}
        
        for orig, perturbed, ptype in zip(
            original_outputs,
            perturbed_outputs,
            perturbation_types
        ):
            stability = self._compute_stability(orig, perturbed)
            stability_scores.append(stability)
            per_type_scores[ptype].append(stability)
        
        value = np.mean(stability_scores)
        ci = self.compute_confidence_interval(np.array(stability_scores))
        
        return MetricResult(
            name="stability",
            value=value,
            confidence_interval=ci,
            submetrics={
                f"{ptype}_stability": np.mean(scores)
                for ptype, scores in per_type_scores.items()
            }
        )