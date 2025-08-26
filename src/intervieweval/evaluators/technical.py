"""
Technical proficiency evaluator
Location: src/intervieweval/evaluators/technical.py
"""

import logging
from typing import Dict, Any, Optional

from intervieweval.evaluators.base import BaseEvaluator
from intervieweval.models.evaluation import TechnicalResult

logger = logging.getLogger(__name__)


class TechnicalEvaluator(BaseEvaluator):
    """
    Evaluates technical proficiency and knowledge depth of candidate responses.
    """

    def __init__(self, settings, prompt_manager, cache=None):
        super().__init__(settings, prompt_manager, cache, chain_name="TECHNICAL")

    def get_prompt_key(self) -> str:
        return "technical"

    async def evaluate(
        self, job_description: str, question: str, response: str, technical_context: Optional[str] = None
    ) -> TechnicalResult:
        """
        Evaluate the technical proficiency of a candidate's response.

        Args:
            job_description: Job requirements
            question: Interview question
            response: Candidate's response
            technical_context: Optional additional technical context

        Returns:
            TechnicalResult with scores and analysis
        """
        # Prepare inputs
        inputs = {
            "job_description": job_description,
            "question": question,
            "response": response,
            "technical_context": technical_context or "No additional context",
        }

        # Invoke chain
        result = await self.chain.ainvoke(inputs)

        # Validate and parse result
        if not self.validate_result(result):
            logger.error(f"Invalid technical evaluation result: {result}")
            raise ValueError("Technical evaluation failed validation")

        # Convert to Pydantic model
        return TechnicalResult(**result)

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate technical evaluation result structure.
        """
        required_fields = [
            "technical_score",
            "accuracy_assessment",
            "depth_analysis",
            "problem_solving",
            "role_fit",
            "overall_assessment",
        ]

        if not all(field in result for field in required_fields):
            logger.warning(f"Missing required fields in technical result")
            return False

        # Validate score range
        score = result.get("technical_score")
        if not isinstance(score, (int, float)) or not 0 <= score <= 100:
            logger.warning(f"Invalid technical score: {score}")
            return False

        # Validate nested structures
        accuracy = result.get("accuracy_assessment", {})
        if not isinstance(accuracy, dict):
            return False

        depth = result.get("depth_analysis", {})
        if not isinstance(depth, dict) or "level" not in depth:
            return False

        # Validate level value
        if depth.get("level") not in ["shallow", "moderate", "deep"]:
            logger.warning(f"Invalid depth level: {depth.get('level')}")
            return False

        role_fit = result.get("role_fit", {})
        if not isinstance(role_fit, dict) or "alignment_score" not in role_fit:
            return False

        return True
