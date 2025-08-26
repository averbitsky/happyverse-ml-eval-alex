"""
Communication skills evaluator
Location: src/intervieweval/evaluators/communication.py
"""

import logging
from typing import Dict, Any

from intervieweval.evaluators.base import BaseEvaluator
from intervieweval.models.evaluation import CommunicationResult

logger = logging.getLogger(__name__)


class CommunicationEvaluator(BaseEvaluator):
    """
    Evaluates communication effectiveness and professionalism of candidate responses.
    """

    def __init__(self, settings, prompt_manager, cache=None, cache_namespace_suffix=""):
        super().__init__(
            settings, prompt_manager, cache, chain_name="COMMUNICATION", cache_namespace_suffix=cache_namespace_suffix
        )

    def get_prompt_key(self) -> str:
        return "communication"

    async def evaluate(self, question: str, response: str) -> CommunicationResult:
        """
        Evaluate the communication effectiveness of a candidate's response.

        Args:
            question: Interview question
            response: Candidate's response

        Returns:
            CommunicationResult with scores and analysis
        """
        # Prepare inputs
        inputs = {"question": question, "response": response}

        # Invoke chain
        result = await self.chain.ainvoke(inputs)

        # Validate and parse result
        if not self.validate_result(result):
            logger.error(f"Invalid communication evaluation result: {result}")
            raise ValueError("Communication evaluation failed validation")

        # Convert to Pydantic model
        return CommunicationResult(**result)

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate communication evaluation result structure.
        """
        required_fields = [
            "communication_score",
            "clarity",
            "relevance",
            "persuasiveness",
            "professionalism",
            "overall_assessment",
        ]

        if not all(field in result for field in required_fields):
            logger.warning(f"Missing required fields in communication result")
            return False

        # Validate score range
        score = result.get("communication_score")
        if not isinstance(score, (int, float)) or not 0 <= score <= 100:
            logger.warning(f"Invalid communication score: {score}")
            return False

        # Validate nested structures
        clarity = result.get("clarity", {})
        if not isinstance(clarity, dict) or "score" not in clarity:
            return False

        relevance = result.get("relevance", {})
        if not isinstance(relevance, dict) or "answers_question" not in relevance:
            return False

        persuasiveness = result.get("persuasiveness", {})
        if not isinstance(persuasiveness, dict) or "score" not in persuasiveness:
            return False

        professionalism = result.get("professionalism", {})
        if not isinstance(professionalism, dict) or "appropriate_tone" not in professionalism:
            return False

        return True
