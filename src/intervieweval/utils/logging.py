"""
Logging utilities with colored output for LLM interactions and system events.
"""

import logging
from typing import Any, Dict

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColoredLogger:
    """
    Helper class for colored logging of LLM interactions and system events.
    """

    @staticmethod
    def log_llm_input(prompt_type: str, content: Dict[str, Any]) -> None:
        """
        Logs LLM input with color for headers only.

        :param prompt_type: Type of the prompt (e.g., "initial", "follow-up").
        :param content: Dictionary containing the prompt content.
        """
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f"{Fore.CYAN}LLM INPUT [{prompt_type}]:")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        for key, value in content.items():
            # Only trim search results and long strings
            if key == "search_results" and isinstance(value, str) and len(value) > 500:
                print(f"{key}: {value[:500]}...")
            elif isinstance(value, str) and len(value) > 1000:
                print(f"{key}: {value[:1000]}...")
            else:
                print(f"{key}: {value}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}`\n")

    @staticmethod
    def log_llm_output(prompt_type: str, result: Dict[str, Any]) -> None:
        """
        Logs LLM output with selective coloring.

        :param prompt_type: Type of the prompt (e.g., "initial", "follow-up").
        :param result: The output result from the LLM, typically a dictionary.
        :return: None.
        """
        print(f"\n{Fore.GREEN}{'=' * 60}")
        print(f"{Fore.GREEN}LLM OUTPUT [{prompt_type}]:")
        print(f"{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
        if isinstance(result, dict):
            for key, value in result.items():
                if key in ["score", "plausibility_score", "technical_score", "communication_score"]:
                    print(f"{Fore.GREEN}{key}: {value}{Style.RESET_ALL}")
                elif key in ["impossible_claims", "technical_errors", "red_flags"]:
                    if value:
                        print(f"{Fore.RED}{key}: {value}{Style.RESET_ALL}")
                    else:
                        print(f"{key}: {value}")
                elif isinstance(value, dict):
                    print(f"{key}: <dict with {len(value)} items>")
                else:
                    print(f"{key}: {value}")
        else:
            print(result)
        print(f"{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}\n")

    @staticmethod
    def log_agent_thought(thought: str) -> None:
        """
        Logs agent reasoning steps in yellow.

        :param thought: The agent's thought process to log.
        :return: None.
        """
        print(f"{Fore.YELLOW}[AGENT THOUGHT]: {thought}{Style.RESET_ALL}")

    @staticmethod
    def log_agent_action(action: str, input_str: str = None) -> None:
        """
        Logs agent actions in cyan.

        :param action: The action taken by the agent.
        :param input_str: Optional input associated with the action.
        :return: None.
        """
        print(f"{Fore.CYAN}[AGENT ACTION]: {action}{Style.RESET_ALL}")
        if input_str:
            print(f"   Input: {input_str}")

    @staticmethod
    def log_agent_observation(observation: str) -> None:
        """
        Logs agent observations in magenta.

        :param observation: The observation made by the agent.
        :return: None.
        """
        # Truncate very long observations
        if len(observation) > 200:
            observation = observation[:200] + "..."
        print(f"{Fore.MAGENTA}[OBSERVATION]: {observation}{Style.RESET_ALL}")

    @staticmethod
    def log_cache_hit(cache_type: str, key: str) -> None:
        """
        Logs cache hits.

        :param cache_type: Type of cache (e.g., "prompt", "search").
        :param key: The cache key that was hit.
        :return: None.
        """
        print(f"{Fore.YELLOW}[CACHE HIT]: Found cached {cache_type} for key: {key[:50]}...{Style.RESET_ALL}")

    @staticmethod
    def log_error(message: str) -> None:
        """
        Logs errors in red.

        :param message: The error message to log.
        :return: None.
        """
        print(f"{Fore.RED}[ERROR]: {message}{Style.RESET_ALL}")

    @staticmethod
    def log_success(message: str) -> None:
        """
        Logs success messages in green.

        :param message: The success message to log.
        :return: None.
        """
        print(f"{Fore.GREEN}[SUCCESS]: {message}{Style.RESET_ALL}")

    @staticmethod
    def log_info(message: str) -> None:
        """
        Logs info messages in blue.

        :param message: The info message to log.
        :return: None.
        """
        print(f"{Fore.BLUE}[INFO]: {message}{Style.RESET_ALL}")


def setup_logging(log_level: str = "INFO", log_format: str = "colored") -> None:
    """
    Sets up the logging configuration.

    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    :param log_format: Format type ("colored" or "json").
    :return: None.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    if log_format == "colored":
        # Colored console output
        format_string = (
            f"{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.YELLOW}%(levelname)s{Style.RESET_ALL} - %(message)s"
        )
    else:
        # Standard format for JSON logging
        format_string = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    logging.basicConfig(level=level, format=format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
