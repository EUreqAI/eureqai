from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import logging

@dataclass
class Requirement:
    """Represents a specific requirement from the EU AI Act."""
    id: str
    name: str
    description: str
    article: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    category: str
    subcategory: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    validation_method: str = "qualitative"  # 'qualitative', 'quantitative', 'hybrid'

@dataclass
class EvaluationResult:
    """Stores the result of evaluating a specific requirement."""
    requirement: Requirement
    score: float
    confidence: float
    evidence: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.results: List[EvaluationResult] = []
        self.requirements: List[Requirement] = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._initialize_requirements()

    @abstractmethod
    def _initialize_requirements(self):
        """Initialize specific requirements for this evaluator."""
        pass

    @abstractmethod
    def evaluate(self, **kwargs) -> List[EvaluationResult]:
        """Perform the evaluation."""
        pass

    def get_compliance_level(self, score: float) -> str:
        """Determine compliance level based on score."""
        if score >= 0.8:
            return "compliant"
        elif score >= 0.6:
            return "partially_compliant"
        return "non_compliant"

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        return {
            "model_info": {
                "name": self.model_name,
                "version": self.model_version,
                "evaluation_date": datetime.now().isoformat()
            },
            "summary": {
                "overall_score": self.get_overall_score(),
                "compliance_level": self.get_overall_compliance_level(),
                "critical_issues": self.get_critical_issues(),
                "total_requirements": len(self.requirements),
                "evaluated_requirements": len(self.results)
            },
            "detailed_results": [
                {
                    "requirement": {
                        "id": result.requirement.id,
                        "name": result.requirement.name,
                        "article": result.requirement.article,
                        "priority": result.requirement.priority
                    },
                    "score": result.score,
                    "confidence": result.confidence,
                    "compliance_level": self.get_compliance_level(result.score),
                    "evidence": result.evidence,
                    "recommendations": result.recommendations,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata
                }
                for result in self.results
            ]
        }

    def get_overall_score(self) -> float:
        """Calculate weighted overall score."""
        if not self.results:
            return 0.0
        
        weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        weighted_sum = sum(
            result.score * weights[result.requirement.priority]
            for result in self.results
        )
        total_weight = sum(
            weights[result.requirement.priority]
            for result in self.results
        )
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """Identify critical non-compliance issues."""
        critical_issues = []
        for result in self.results:
            if (result.requirement.priority == 'critical' and 
                result.score < 0.6):
                critical_issues.append({
                    "requirement": result.requirement.name,
                    "article": result.requirement.article,
                    "score": result.score,
                    "recommendations": result.recommendations
                })
        return critical_issues

    def get_overall_compliance_level(self) -> str:
        """Determine overall compliance level."""
        overall_score = self.get_overall_score()
        critical_issues = self.get_critical_issues()
        
        if critical_issues:
            return "non_compliant"
        return self.get_compliance_level(overall_score)