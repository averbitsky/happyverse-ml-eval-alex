"""
Test import patterns.
"""

import pytest
from intervieweval.config.settings import Settings
from intervieweval.evaluators.orchestrator import EvaluationOrchestrator
from intervieweval.evaluators.plausibility import PlausibilityEvaluator
from intervieweval.models.evaluation import FinalEvaluation, RecommendationLevel
from intervieweval.models.evaluation import PlausibilityResult, TechnicalResult
from intervieweval.prompts.manager import PromptManager
from intervieweval.utils.logging import ColoredLogger, setup_logging


class TestImports:
    """
    Tests to ensure that all key package parts can be imported.
    """

    def test_settings_import(self) -> None:
        """
        Tests that Settings can be imported and instantiated.

        :return: None.
        """
        settings = Settings(openai_api_key="test_key")
        assert settings.openai_api_key == "test_key"
        assert settings.openai_model == "gpt-4"

    def test_prompt_manager_import(self) -> None:
        """
        Tests that PromptManager can be imported.

        :return: None.
        """
        assert PromptManager is not None

    def test_model_imports(self) -> None:
        """
        Tests that all models can be imported.

        :return: None.
        """
        assert PlausibilityResult is not None
        assert TechnicalResult is not None
        assert FinalEvaluation is not None
        assert RecommendationLevel.STRONG_YES.value == "Strong Yes"

    def test_evaluator_imports(self) -> None:
        """
        Tests that evaluator classes can be imported.

        :return: None.
        """
        assert PlausibilityEvaluator is not None

    def test_utils_imports(self) -> None:
        """
        Tests that utility functions can be imported.

        :return: None.
        """
        assert ColoredLogger is not None
        assert setup_logging is not None

        # Test ColoredLogger methods exist
        assert hasattr(ColoredLogger, "log_agent_thought")
        assert hasattr(ColoredLogger, "log_llm_input")
        assert hasattr(ColoredLogger, "log_success")


class TestEvaluatorIntegration:
    """
    Integration tests for the evaluation system.
    """

    @pytest.fixture
    def settings(self) -> Settings:
        """
        Creates test settings.

        :return: Settings object with test configuration.
        """
        return Settings(
            openai_api_key="test_key",
            openai_model="gpt-3.5-turbo",
            enable_metrics=False,
            enable_search_cache=False,
            enable_prompt_cache=False,
        )

    @pytest.fixture
    def prompt_manager(self, tmp_path) -> PromptManager:
        """
        Creates a test prompt manager with minimal prompts.

        :param tmp_path: Temporary path for creating test files.
        :return: PromptManager object with test prompts.
        """
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
prompts:
  plausibility:
    template: "Test plausibility prompt {job_description} {question} {response} {search_results}"
  technical:
    template: "Test technical prompt {job_description} {question} {response} {technical_context}"
  communication:
    template: "Test communication prompt {question} {response}"
  synthesis:
    template: "Test synthesis prompt {evaluations} {job_description}"
  entity_extraction:
    template: "Test entity extraction {response}"
  claim_verification:
    template: "Test claim verification {claim}"
"""
        )
        return PromptManager(str(prompts_file))

    def test_orchestrator_creation(self, settings, prompt_manager) -> None:
        """
        Tests that orchestrator can be created with proper imports.

        :param settings: Settings fixture.
        :param prompt_manager: PromptManager fixture.
        :return: None.
        """
        # This would normally require mocking the LLM calls
        orchestrator = EvaluationOrchestrator(settings=settings, prompt_manager=prompt_manager, cache=None)
        assert orchestrator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
