"""
Example test file showing proper import patterns
Location: tests/test_imports.py
"""

import pytest
from pathlib import Path

# All imports use the package name 'intervieweval'
from intervieweval import (
    Settings,
    PromptManager,
    PersistentCache,
    EvaluationOrchestrator,
    FinalEvaluation,
    RecommendationLevel,
    __version__
)
from intervieweval.evaluators.plausibility import PlausibilityEvaluator
from intervieweval.models.evaluation import PlausibilityResult, TechnicalResult
from intervieweval.utils.logging import ColoredLogger, setup_logging


class TestImports:
    """Test that all imports work correctly"""

    def test_version(self):
        """Test package version is accessible"""
        assert __version__ == "0.0.1"

    def test_settings_import(self):
        """Test Settings can be imported and instantiated"""
        settings = Settings(openai_api_key="test_key")
        assert settings.openai_api_key == "test_key"
        assert settings.openai_model == "gpt-4"

    def test_prompt_manager_import(self):
        """Test PromptManager can be imported"""
        # Would need actual prompts.yaml file to instantiate
        assert PromptManager is not None

    def test_model_imports(self):
        """Test all models can be imported"""
        assert PlausibilityResult is not None
        assert TechnicalResult is not None
        assert FinalEvaluation is not None
        assert RecommendationLevel.STRONG_YES.value == "Strong Yes"

    def test_evaluator_imports(self):
        """Test evaluator classes can be imported"""
        assert PlausibilityEvaluator is not None

    def test_utils_imports(self):
        """Test utility functions can be imported"""
        assert ColoredLogger is not None
        assert setup_logging is not None

        # Test ColoredLogger methods exist
        assert hasattr(ColoredLogger, 'log_agent_thought')
        assert hasattr(ColoredLogger, 'log_llm_input')
        assert hasattr(ColoredLogger, 'log_success')


class TestEvaluatorIntegration:
    """Integration tests for the evaluation system"""

    @pytest.fixture
    def settings(self):
        """Create test settings"""
        return Settings(
            openai_api_key="test_key",
            openai_model="gpt-3.5-turbo",
            enable_metrics=False,
            enable_search_cache=False,
            enable_prompt_cache=False
        )

    @pytest.fixture
    def prompt_manager(self, tmp_path):
        """Create test prompt manager with minimal prompts"""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text("""
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
""")
        return PromptManager(str(prompts_file))

    def test_orchestrator_creation(self, settings, prompt_manager):
        """Test that orchestrator can be created with proper imports"""
        # This would normally require mocking the LLM calls
        orchestrator = EvaluationOrchestrator(
            settings=settings,
            prompt_manager=prompt_manager,
            cache=None
        )
        assert orchestrator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])