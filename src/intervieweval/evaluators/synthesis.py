"""
Synthesis evaluator for final hiring recommendations.
"""

import json
import logging
from typing import Any, Dict, List

from intervieweval.evaluators.base import BaseEvaluator
from intervieweval.models.evaluation import QuestionEvaluation, SynthesisResult

logger = logging.getLogger(__name__)


class SynthesisEvaluator(BaseEvaluator):
    """
    Synthesizes all evaluation results into a final hiring recommendation.
    """

    def __init__(self, settings, prompt_manager, cache=None) -> None:
        """
        Initializes the SynthesisEvaluator with settings, prompt manager, and optional cache.

        :param settings: Configuration settings.
        :param prompt_manager: Prompt template manager.
        :param cache: Optional persistent cache.
        :return: None.
        """
        super().__init__(settings, prompt_manager, cache, chain_name="SYNTHESIS")

    @staticmethod
    def get_prompt_key() -> str:
        """
        Get the prompt key for synthesis evaluation.

        :return: Prompt key string.
        """
        return "synthesis"

    async def evaluate(self, evaluations: List[QuestionEvaluation], job_description: str) -> SynthesisResult:
        """
        Synthesize all evaluations into a final recommendation.

        :param evaluations: List of individual question evaluations.
        :param job_description: Job requirements.
        :return: SynthesisResult with final recommendation.
        """
        # Format evaluations for the prompt
        evaluation_summary = []
        for eval in evaluations:
            summary = {
                "question": eval.question,
                "scores": {
                    "plausibility": eval.plausibility.plausibility_score if eval.plausibility else 0,
                    "technical": eval.technical.technical_score if eval.technical else 0,
                    "communication": eval.communication.communication_score if eval.communication else 0,
                },
                "key_issues": {
                    "impossible_claims": (
                        eval.plausibility.technical_feasibility.impossible_claims if eval.plausibility else []
                    ),
                    "technical_errors": eval.technical.accuracy_assessment.errors if eval.technical else [],
                    "red_flags": eval.plausibility.verifiability.red_flags if eval.plausibility else [],
                },
            }
            evaluation_summary.append(summary)

        # Prepare inputs
        inputs = {"evaluations": json.dumps(evaluation_summary, indent=2), "job_description": job_description}

        # Invoke chain
        result = await self.chain.ainvoke(inputs)

        # Validate and parse result
        if not self.validate_result(result):
            logger.error(f"Invalid synthesis result: {result}")
            raise ValueError("Synthesis evaluation failed validation")

        # Convert to a Pydantic model
        return SynthesisResult(**result)

    @staticmethod
    def validate_result(result: Dict[str, Any]) -> bool:
        """
        Validate synthesis result structure.

        :param result: Result dictionary from the synthesis chain.
        :return: True if valid, False otherwise.
        """
        required_fields = [
            "recommendation_level",
            "confidence",
            "key_strengths",
            "critical_concerns",
            "deal_breakers",
            "hiring_risk",
            "comparison_to_typical",
            "specific_recommendations",
            "detailed_rationale",
        ]

        if not all(field in result for field in required_fields):
            logger.warning(f"Missing required fields in synthesis result")
            return False

        # Validate recommendation level
        valid_levels = ["Strong Yes", "Weak Yes", "Weak No", "Strong No"]
        if result.get("recommendation_level") not in valid_levels:
            logger.warning(f"Invalid recommendation level: {result.get('recommendation_level')}")
            return False

        # Validate confidence
        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            logger.warning(f"Invalid confidence score: {confidence}")
            return False

        # Validate hiring risk
        if result.get("hiring_risk") not in ["low", "medium", "high"]:
            logger.warning(f"Invalid hiring risk: {result.get('hiring_risk')}")
            return False

        # Validate specific recommendations
        recs = result.get("specific_recommendations", {})
        if not isinstance(recs, dict) or "if_hire" not in recs or "if_reject" not in recs:
            return False

        return True
