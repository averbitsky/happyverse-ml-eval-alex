"""
Plausibility evaluator for assessing truthfulness and feasibility.
"""

import logging
from typing import Any, Dict, Optional

from intervieweval.evaluators.base import BaseEvaluator
from intervieweval.models.evaluation import PlausibilityResult
from intervieweval.tools.verification import EntityVerifier

logger = logging.getLogger(__name__)


class PlausibilityEvaluator(BaseEvaluator):
    """
    Evaluates the plausibility and truthfulness of candidate responses.
    """

    def __init__(self, settings, prompt_manager, cache=None, cache_namespace_suffix="") -> None:
        """
        Initialize PlausibilityEvaluator.

        :param settings: Configuration settings.
        :param prompt_manager: Prompt template manager.
        :param cache: Optional persistent cache.
        :param cache_namespace_suffix: Suffix for cache namespace to prevent cross-contamination.
        :return: None.
        """
        super().__init__(
            settings, prompt_manager, cache, chain_name="PLAUSIBILITY", cache_namespace_suffix=cache_namespace_suffix
        )
        # Verifier is passed in from the orchestrator to ensure a new instance

    @staticmethod
    def get_prompt_key() -> str:
        """
        Get the prompt key for plausibility evaluation.

        :return: Prompt key string.
        """
        return "plausibility"

    async def evaluate(
        self, job_description: str, question: str, response: str, search_results: Optional[str] = None
    ) -> PlausibilityResult:
        """
        Evaluates the plausibility of a candidate's response.

        :param job_description: Job requirements.
        :param question: Interview question.
        :param response: Candidate's response.
        :param search_results: Optional web search results for verification.
        :return: PlausibilityResult with scores and analysis.
        """
        # Prepare inputs
        inputs = {
            "job_description": job_description,
            "question": question,
            "response": response,
            "search_results": search_results or "No search performed",
        }

        # Invoke chain
        result = await self.chain.ainvoke(inputs)

        # Validate and parse result
        if not self.validate_result(result):
            logger.error(f"Invalid plausibility evaluation result: {result}")
            raise ValueError("Plausibility evaluation failed validation")

        # Convert to a Pydantic model
        return PlausibilityResult(**result)

    @staticmethod
    def validate_result(result: Dict[str, Any]) -> bool:
        """
        Validates plausibility evaluation result structure.

        :param result: Result dictionary from the evaluation chain.
        :return: True if valid, False otherwise.
        """
        required_fields = [
            "plausibility_score",
            "technical_feasibility",
            "verifiability",
            "authenticity_indicators",
            "overall_assessment",
        ]

        if not all(field in result for field in required_fields):
            return False

        # Validate score range
        score = result.get("plausibility_score")
        if not isinstance(score, (int, float)) or not 0 <= score <= 100:
            return False

        # Validate nested structures
        tech_feasibility = result.get("technical_feasibility", {})
        if not isinstance(tech_feasibility, dict) or "assessment" not in tech_feasibility:
            return False

        return True

    async def evaluate_with_verification(
        self, job_description: str, question: str, response: str
    ) -> tuple[PlausibilityResult, Dict[str, Any]]:
        """
        Evaluates plausibility with automatic entity verification.

        :param job_description: Job requirements.
        :param question: Interview question.
        :param response: Candidate's response.
        :return: Tuple of (PlausibilityResult, verification_results).
        """
        # First, extract and verify entities
        entities = await self.verifier.extract_entities(response)
        verification_results = await self.verifier.verify_entities(entities)

        # Format search results for the prompt
        search_results = self._format_verification_results(verification_results)

        # Evaluate with verification results
        plausibility_result = await self.evaluate(
            job_description=job_description, question=question, response=response, search_results=search_results
        )

        return plausibility_result, verification_results

    @staticmethod
    def _format_verification_results(verification_results: Dict[str, Any]) -> str:
        """
        Formats verification results for inclusion in the prompt.

        :param verification_results: Dictionary of verification results.
        :return: Formatted string of verification results.
        """
        if not verification_results:
            return "No verification performed"

        formatted_parts = []

        # Format company verifications
        if "companies" in verification_results:
            formatted_parts.append("Company Verifications:")
            for company, result in verification_results["companies"].items():
                formatted_parts.append(f"  - {company}: {result[:200]}")

        # Format technology verifications
        if "technologies" in verification_results:
            formatted_parts.append("\nTechnology Verifications:")
            for tech, result in verification_results["technologies"].items():
                formatted_parts.append(f"  - {tech}: {result[:200]}")

        # Format claim verifications
        if "implementations" in verification_results:
            formatted_parts.append("\nClaim Verifications:")
            for claim, result in verification_results["implementations"].items():
                formatted_parts.append(f"  - {claim[:100]}: {result[:200]}")

        return "\n".join(formatted_parts) if formatted_parts else "No verification results"
