from typing import List, Dict, Any, Optional
from .base import BaseEvaluator, Requirement, EvaluationResult
import numpy as np

class PrivacyEvaluator(BaseEvaluator):
    """Evaluates privacy and data protection requirements."""
    
    def _initialize_requirements(self):
        self.requirements = [
            Requirement(
                id="PRIV-1",
                name="Data Minimization",
                description="System collects and processes only necessary data",
                article="Article 10(5)",
                priority="high",
                category="Privacy",
                metrics=["data_necessity_score"]
            ),
            Requirement(
                id="PRIV-2",
                name="Privacy by Design",
                description="Privacy considerations integrated into system design",
                article="Article 10(6)",
                priority="critical",
                category="Privacy",
                metrics=["privacy_design_score"]
            ),
            Requirement(
                id="PRIV-3",
                name="Data Protection",
                description="Appropriate measures for data protection implemented",
                article="Article 10(7)",
                priority="critical",
                category="Privacy",
                metrics=["protection_measure_score"]
            )
        ]

    def evaluate(self, 
                system_data: Dict[str, Any],
                privacy_measures: Dict[str, Any],
                data_flow: Dict[str, Any]) -> List[EvaluationResult]:
        results = []
        
        # Evaluate data minimization
        minimization_result = self._evaluate_data_minimization(system_data)
        results.append(minimization_result)
        
        # Evaluate privacy by design
        privacy_design_result = self._evaluate_privacy_design(privacy_measures)
        results.append(privacy_design_result)
        
        # Evaluate data protection measures
        protection_result = self._evaluate_data_protection(data_flow)
        results.append(protection_result)
        
        return results

    def _evaluate_data_minimization(self, system_data: Dict[str, Any]) -> EvaluationResult:
        # Implementation for data minimization evaluation
        required_fields = system_data.get('required_fields', [])
        collected_fields = system_data.get('collected_fields', [])
        
        unnecessary_fields = [f for f in collected_fields if f not in required_fields]
        score = 1.0 - (len(unnecessary_fields) / len(collected_fields) if collected_fields else 0)
        
        return EvaluationResult(
            requirement=self.requirements[0],
            score=score,
            confidence=0.9,
            evidence=[f"Unnecessary fields: {unnecessary_fields}"],
            recommendations=self._get_minimization_recommendations(unnecessary_fields)
        )

    def _evaluate_privacy_design(self, privacy_measures: Dict[str, Any]) -> EvaluationResult:
        required_measures = {
            'encryption': {'weight': 0.3, 'required': True},
            'anonymization': {'weight': 0.3, 'required': True},
            'access_control': {'weight': 0.2, 'required': True},
            'data_retention': {'weight': 0.1, 'required': False},
            'audit_logging': {'weight': 0.1, 'required': False}
        }
        
        score = 0.0
        missing_required = []
        
        for measure, config in required_measures.items():
            if measure in privacy_measures:
                score += config['weight']
            elif config['required']:
                missing_required.append(measure)
        
        return EvaluationResult(
            requirement=self.requirements[1],
            score=score,
            confidence=0.85,
            evidence=[f"Missing required measures: {missing_required}"],
            recommendations=self._get_privacy_recommendations(missing_required)
        )