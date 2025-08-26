"""
Communication skills evaluator.
"""

import logging
from typing import Any, Dict

from intervieweval.evaluators.base import BaseEvaluator
from intervieweval.models.evaluation import CommunicationResult

logger = logging.getLogger(__name__)


class CommunicationEvaluator(BaseEvaluator):
    """
    Evaluates communication effectiveness and professionalism of candidate responses.
    """

    def __init__(self, settings, prompt_manager, cache=None, cache_namespace_suffix="") -> None:
        """
        Initialize communication evaluator.

        :param settings: Configuration settings.
        :param prompt_manager: Prompt template manager.
        :param cache: Optional persistent cache.
        :param cache_namespace_suffix: Suffix for cache namespace to prevent cross-contamination.
        :return: None.
        """
        super().__init__(
            settings, prompt_manager, cache, chain_name="COMMUNICATION", cache_namespace_suffix=cache_namespace_suffix
        )

    @staticmethod
    def get_prompt_key() -> str:
        """
        Gets the prompt key for communication evaluation.

        :return: Prompt key string.
        """
        return "communication"

    async def evaluate(self, question: str, response: str) -> CommunicationResult:
        """
        Evaluates the communication effectiveness of a candidate's response.

        :param question: Interview question.
        :param response: Candidate's response.
        :return: CommunicationResult with scores and analysis
        """
        # Prepare inputs
        inputs = {"question": question, "response": response}

        # Invoke chain
        result = await self.chain.ainvoke(inputs)

        # Validate and parse result
        if not self.validate_result(result):
            logger.error(f"Invalid communication evaluation result: {result}")
            raise ValueError("Communication evaluation failed validation")

        # Convert to a Pydantic model
        return CommunicationResult(**result)

    @staticmethod
    def validate_result(result: Dict[str, Any]) -> bool:
        """
        Validates communication evaluation result structure.

        :param result: Evaluation result.
        :return: True if valid, False otherwise.
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
