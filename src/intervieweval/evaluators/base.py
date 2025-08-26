"""
Base evaluator class with common functionality.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import openai
from intervieweval.cache.manager import PersistentCache
from intervieweval.config.settings import Settings
from intervieweval.prompts.manager import PromptManager
from intervieweval.utils.logging import ColoredLogger
from intervieweval.utils.metrics import llm_calls, rate_limit_errors
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators. Provides common functionality for LLM chains, caching, and monitoring.
    """

    def __init__(
        self,
        settings: Settings,
        prompt_manager: PromptManager,
        cache: Optional[PersistentCache] = None,
        chain_name: str = "BASE",
        cache_namespace_suffix: str = "",
    ) -> None:
        """
        Initializes base evaluator.

        :param settings: Configuration settings.
        :param prompt_manager: Prompt template manager.
        :param cache: Optional persistent cache.
        :param chain_name: Name of this evaluation chain for metrics.
        :param cache_namespace_suffix: Suffix for cache namespace to prevent cross-contamination.
        :return: None.
        """
        self.settings = settings
        self.prompt_manager = prompt_manager
        self.cache = cache
        self.chain_name = chain_name
        self.cache_namespace_suffix = cache_namespace_suffix

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model, temperature=settings.openai_temperature, openai_api_key=settings.openai_api_key
        )

        # Initialize chain
        self.chain = self._create_chain()

        logger.info(f"Initialized {chain_name} evaluator")

    @abstractmethod
    def get_prompt_key(self) -> str:
        """
        Gets the key for this evaluator's prompt in the prompt manager.

        :return: Prompt key (e.g., 'plausibility', 'technical').
        """
        pass

    def _get_cache_namespace(self) -> str:
        """
        Gets the cache namespace for this evaluator. Includes suffix to prevent cross-contamination between transcripts.

        :return: Cache namespace string.
        """
        base_namespace = f"llm_{self.chain_name.lower()}"
        if self.cache_namespace_suffix:
            return f"{base_namespace}_{self.cache_namespace_suffix}"
        return base_namespace

    def _create_chain(self) -> "MonitoredChain":
        """
        Creates the evaluator's LLM chain with prompt, LLM, and output parser.

        :return: Configured LangChain chain object.
        """
        # Get prompt template
        prompt_template = self.prompt_manager.get_prompt(self.get_prompt_key())
        if not prompt_template:
            raise ValueError(f"Prompt not found for key: {self.get_prompt_key()}")

        # Create chain components
        prompt = ChatPromptTemplate.from_template(prompt_template)
        parser = JsonOutputParser()
        base_chain = prompt | self.llm | parser

        # Wrap with monitoring and retry logic
        return self._create_monitored_chain(base_chain)

    def _create_monitored_chain(self, base_chain) -> "MonitoredChain":
        """
        Wraps a chain with monitoring, caching, and retry logic.

        :param base_chain: Base LangChain chain to wrap.
        :return: Wrapped chain with monitoring and caching.
        """

        class MonitoredChain:
            """
            Wrapper for a chain with monitoring, caching, and retry.
            """

            def __init__(self, chain, evaluator) -> None:
                """
                Initializes monitored chain wrapper.

                :param chain: Base chain to wrap.
                :param evaluator: Evaluator instance for context.
                :return: None.
                """
                self.chain = chain
                self.evaluator = evaluator

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=60),
                retry=retry_if_exception_type(openai.RateLimitError),
            )
            async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """
                Async invoke with monitoring and retry.

                :param inputs: Chain inputs.
                :return: Chain outputs.
                """
                llm_calls.labels(chain_type=self.evaluator.chain_name).inc()

                # Check the cache if available
                cache_key = self.evaluator._get_cache_key(inputs)
                if self.evaluator.cache and cache_key:
                    cached_result = self.evaluator.cache.get(
                        namespace=self.evaluator._get_cache_namespace(), key=cache_key
                    )
                    if cached_result is not None:
                        ColoredLogger.log_cache_hit(self.evaluator.chain_name, cache_key)
                        return cached_result

                # Log input
                ColoredLogger.log_llm_input(self.evaluator.chain_name, inputs)

                try:
                    # Invoke chain
                    result = await self.chain.ainvoke(inputs)

                    # Log output
                    ColoredLogger.log_llm_output(self.evaluator.chain_name, result)

                    # Cache result if available
                    if self.evaluator.cache and cache_key:
                        self.evaluator.cache.set(
                            namespace=self.evaluator._get_cache_namespace(),
                            key=cache_key,
                            value=result,
                            ttl=self.evaluator.settings.cache_ttl_seconds,
                        )

                    return result

                except openai.RateLimitError as e:
                    rate_limit_errors.inc()
                    logger.warning(f"Rate limit hit for {self.evaluator.chain_name}: {str(e)}")
                    raise

            def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """
                Synchronous invoke wrapper.

                :param inputs: Chain inputs.
                :return: Chain outputs.
                """
                return asyncio.run(self.ainvoke(inputs))

        return MonitoredChain(base_chain, self)

    def _get_cache_key(self, inputs: Dict[str, Any]) -> Optional[str]:
        """
        Generates a cache key for the given inputs. Override in subclasses for custom caching strategies.

        :param inputs: Chain inputs,
        :return: Cache key or None to skip caching.
        """
        # Create a deterministic hash of the inputs
        key_parts = []
        for key in sorted(inputs.keys()):
            if key in ["question", "response"]:  # Cache based on Q&A
                # Hash the content for a consistent key length
                content_hash = hashlib.md5(str(inputs[key]).encode()).hexdigest()[:16]
                key_parts.append(f"{key}:{content_hash}")

        return "|".join(key_parts) if key_parts else None

    @abstractmethod
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Performs evaluation with the given inputs.

        :return: Evaluation results dictionary
        """
        pass

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validates the structure of the evaluation result. Override in subclasses for specific validation.

        :param result: Evaluation result.
        :return: True if valid.
        """
        return isinstance(result, dict)


class EvaluatorFactory:
    """
    Factory for creating evaluator instances.
    """

    @staticmethod
    def create_evaluators(
        settings: Settings, prompt_manager: PromptManager, cache: Optional[PersistentCache] = None
    ) -> Dict[str, BaseEvaluator]:
        """
        Creates all evaluator instances.

        :param settings: Configuration settings.
        :param prompt_manager: Prompt template manager.
        :param cache: Optional persistent cache.
        :return: Dictionary of evaluator name to instance.
        """
        from intervieweval.evaluators.plausibility import PlausibilityEvaluator
        from intervieweval.evaluators.technical import TechnicalEvaluator
        from intervieweval.evaluators.communication import CommunicationEvaluator
        from intervieweval.evaluators.synthesis import SynthesisEvaluator

        evaluators = {
            "plausibility": PlausibilityEvaluator(settings, prompt_manager, cache),
            "technical": TechnicalEvaluator(settings, prompt_manager, cache),
            "communication": CommunicationEvaluator(settings, prompt_manager, cache),
            "synthesis": SynthesisEvaluator(settings, prompt_manager, cache),
        }

        logger.info(f"Created {len(evaluators)} evaluators")
        return evaluators
