"""
Prompt Manager for loading and managing evaluation prompts from YAML
Location: src/intervieweval/prompts/manager.py
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates loaded from YAML configuration.
    Supports versioning, hot-reloading, and validation.
    """

    def __init__(self, prompt_file: str = "prompts.yaml"):
        """
        Initialize prompt manager.

        Args:
            prompt_file: Path to YAML file containing prompts
        """
        self.prompt_file = Path(prompt_file)
        self.prompts: Dict[str, Dict] = {}
        self.file_hash: Optional[str] = None
        self.last_loaded: Optional[datetime] = None

        # Load prompts on initialization
        self.load_prompts()

    def load_prompts(self) -> bool:
        """
        Load prompts from YAML file.

        Returns:
            True if successfully loaded
        """
        try:
            if not self.prompt_file.exists():
                logger.error(f"Prompt file not found: {self.prompt_file}")
                return False

            with open(self.prompt_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Check if file has changed
                new_hash = hashlib.md5(content.encode()).hexdigest()
                if new_hash == self.file_hash:
                    logger.debug("Prompt file unchanged, skipping reload")
                    return True

                # Parse YAML
                data = yaml.safe_load(content)

                # Validate structure
                if not isinstance(data, dict):
                    logger.error("Invalid prompt file structure: root must be a dictionary")
                    return False

                # Handle both new format (with 'prompts' key) and old format
                if "prompts" in data:
                    prompt_data = data["prompts"]
                else:
                    prompt_data = data

                # Validate each prompt section
                required_prompts = [
                    "plausibility",
                    "technical",
                    "communication",
                    "synthesis",
                    "entity_extraction",
                    "claim_verification",
                ]

                for prompt_name in required_prompts:
                    if prompt_name not in prompt_data:
                        logger.error(f"Missing required prompt: {prompt_name}")
                        return False

                    if "template" not in prompt_data[prompt_name]:
                        logger.error(f"Missing template for prompt: {prompt_name}")
                        return False

                # Store prompts
                self.prompts = prompt_data
                self.file_hash = new_hash
                self.last_loaded = datetime.now()

                logger.info(f"Successfully loaded {len(self.prompts)} prompts from {self.prompt_file}")
                return True

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            return False

    def reload_if_changed(self) -> bool:
        """
        Check if prompt file has changed and reload if necessary.

        Returns:
            True if reloaded or no changes
        """
        try:
            # Get current file modification time
            current_mtime = self.prompt_file.stat().st_mtime

            # Check if we need to reload
            if self.last_loaded:
                last_mtime = self.last_loaded.timestamp()
                if current_mtime > last_mtime:
                    logger.info("Prompt file changed, reloading...")
                    return self.load_prompts()

            return True

        except Exception as e:
            logger.error(f"Failed to check prompt file changes: {e}")
            return False

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Get a specific prompt template.

        Args:
            prompt_name: Name of the prompt (e.g., 'plausibility', 'technical')

        Returns:
            Prompt template string or None if not found
        """
        # Optionally reload if file changed
        self.reload_if_changed()

        if prompt_name in self.prompts:
            return self.prompts[prompt_name].get("template")

        logger.warning(f"Prompt not found: {prompt_name}")
        return None

    def get_all_prompts(self) -> Dict[str, str]:
        """
        Get all prompt templates.

        Returns:
            Dictionary of prompt names to templates
        """
        # Optionally reload if file changed
        self.reload_if_changed()

        return {name: data.get("template") for name, data in self.prompts.items() if "template" in data}

    def get_metadata(self, prompt_name: str) -> Dict:
        """
        Get metadata for a prompt (everything except the template).

        Args:
            prompt_name: Name of the prompt

        Returns:
            Dictionary of metadata
        """
        if prompt_name in self.prompts:
            metadata = self.prompts[prompt_name].copy()
            metadata.pop("template", None)
            return metadata

        return {}

    def validate_prompts(self) -> Dict[str, bool]:
        """
        Validate all prompts for required placeholders.

        Returns:
            Dictionary of prompt names to validation status
        """
        validation_results = {}

        # Define required placeholders for each prompt
        required_placeholders = {
            "plausibility": ["{job_description}", "{question}", "{response}", "{search_results}"],
            "technical": ["{job_description}", "{question}", "{response}", "{technical_context}"],
            "communication": ["{question}", "{response}"],
            "synthesis": ["{evaluations}", "{job_description}"],
            "entity_extraction": ["{response}"],
            "claim_verification": ["{claim}"],
        }

        for prompt_name, template in self.get_all_prompts().items():
            if prompt_name in required_placeholders:
                placeholders = required_placeholders[prompt_name]
                is_valid = all(placeholder in template for placeholder in placeholders)
                validation_results[prompt_name] = is_valid

                if not is_valid:
                    missing = [p for p in placeholders if p not in template]
                    logger.warning(f"Prompt '{prompt_name}' missing placeholders: {missing}")

        return validation_results

    def export_prompts(self, export_path: str) -> bool:
        """
        Export current prompts to a new file.

        Args:
            export_path: Path to export file

        Returns:
            True if successful
        """
        try:
            export_file = Path(export_path)

            # Add metadata
            export_data = {
                "_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "source_file": str(self.prompt_file),
                    "version": self.file_hash,
                },
                "prompts": self.prompts,
            }

            # Write to file
            with open(export_file, "w", encoding="utf-8") as f:
                yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Prompts exported to {export_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export prompts: {e}")
            return False

    def get_prompt_stats(self) -> Dict:
        """
        Get statistics about loaded prompts.

        Returns:
            Dictionary of statistics
        """
        all_prompts = self.get_all_prompts()

        return {
            "total_prompts": len(all_prompts),
            "prompt_names": list(all_prompts.keys()),
            "file_path": str(self.prompt_file),
            "last_loaded": self.last_loaded.isoformat() if self.last_loaded else None,
            "file_hash": self.file_hash,
            "average_prompt_length": sum(len(p) for p in all_prompts.values()) / len(all_prompts) if all_prompts else 0,
            "validation_status": self.validate_prompts(),
        }
