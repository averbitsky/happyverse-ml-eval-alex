"""
Configuration settings management using Pydantic.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with Pydantic validation. Loads from environment variables or .env file.
    """

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.2, env="OPENAI_TEMPERATURE", ge=0.0, le=2.0)

    # Rate Limiting
    max_retries: int = Field(default=3, env="MAX_RETRIES", ge=1, le=10)
    retry_min_wait: int = Field(default=4, env="RETRY_MIN_WAIT", ge=1)
    retry_max_wait: int = Field(default=60, env="RETRY_MAX_WAIT", le=300)

    # Parallelization
    max_parallel_evaluations: int = Field(default=3, env="MAX_PARALLEL_EVALUATIONS", ge=1, le=10)
    max_parallel_verifications: int = Field(default=3, env="MAX_PARALLEL_VERIFICATIONS", ge=1, le=10)
    batch_size: int = Field(default=10, env="BATCH_SIZE", ge=1, le=50)

    # Caching
    enable_prompt_cache: bool = Field(default=True, env="ENABLE_PROMPT_CACHE")
    enable_search_cache: bool = Field(default=True, env="ENABLE_SEARCH_CACHE")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS", ge=60)
    cache_db_name: str = Field(default="evaluator_cache.db", env="CACHE_DB_NAME")
    max_cache_size_mb: int = Field(default=100, env="MAX_CACHE_SIZE_MB", ge=10)

    # Search Configuration
    search_max_results: int = Field(default=5, env="SEARCH_MAX_RESULTS", ge=1, le=20)

    # File names (not paths)
    technologies_filename: Optional[str] = Field(default="Tools and Technology.xlsx", env="TECHNOLOGIES_FILE")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8000, env="METRICS_PORT", ge=1024, le=65535)

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="colored", env="LOG_FORMAT")  # "colored" or "json"

    class Config:
        """
        Pydantic configuration for environment variable loading.
        """

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def project_root(self) -> Path:
        """
        Gets the root directory of the project (parent of src/).

        :return: Path to the project root directory.
        """
        # Go up from settings.py -> config -> intervieweval -> src -> project_root
        return Path(__file__).parent.parent.parent.parent

    def get_cache_dir(self) -> Path:
        """
        Gets the cache directory path.

        :return: Path to the cache directory.
        """
        return self.project_root / "cache"

    def get_cache_path(self) -> Path:
        """
        Gets the full cache database path.

        :return: Path to the cache database file.
        """
        return self.get_cache_dir() / self.cache_db_name

    def get_prompts_path(self) -> Path:
        """
        Gets the path to the prompts.yaml file.

        :return: Path to the prompts.yaml file located alongside settings.py.
        """
        return Path(__file__).parent / "prompts.yaml"

    def get_output_dir(self) -> Path:
        """
        Gets the output directory path.

        :return: Path to the output directory.
        """
        return self.project_root / "output"

    def get_data_dir(self) -> Path:
        """
        Gets the data directory path.

        :return: Path to the data directory.
        """
        return self.project_root / "data"

    def get_job_description_path(self) -> Path:
        """
        Gets the path to the job description file.

        :return: Path to the job_description.txt file.
        """
        return self.get_data_dir() / "job_description.txt"

    def get_questions_path(self) -> Path:
        """
        Gets the path to the interview questions file.

        :return: Path to the questions.txt file.
        """
        return self.get_data_dir() / "questions.txt"

    def get_transcripts_dir(self) -> Path:
        """
        Gets the transcripts directory path.

        :return: Path to the transcripts directory.
        """
        return self.get_data_dir() / "transcripts"

    def get_technologies_file_path(self) -> Optional[Path]:
        """
        Gets the path to the technologies Excel file if it exists.

        :return: Path to the technologies Excel file, or None if not set or does not exist.
        """
        if not self.technologies_filename:
            return None

        tech_path = self.get_data_dir() / self.technologies_filename
        return tech_path if tech_path.exists() else None
