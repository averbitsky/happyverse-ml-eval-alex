"""
Entity verification tools for fact-checking claims in candidate responses.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

import openai
import pandas as pd
from intervieweval.cache.manager import PersistentCache
from intervieweval.config.settings import Settings
from intervieweval.models.evaluation import VerificationEntities
from intervieweval.tools.search import WebSearchTool
from intervieweval.utils.logging import ColoredLogger
from intervieweval.utils.metrics import entities_extracted, llm_calls, rate_limit_errors, verifications_performed
from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Common technologies that are known to exist
COMMON_TECHNOLOGIES = {
    "5g",
    "4g",
    "lte",
    "3gpp",
    "wifi",
    "wifi6",
    "kubernetes",
    "docker",
    "python",
    "java",
    "javascript",
    "react",
    "node.js",
    "redis",
    "kafka",
    "aws",
    "azure",
    "gcp",
    "tcp",
    "udp",
    "http",
    "https",
    "dns",
    "dhcp",
    "ipsec",
    "vpn",
    "sql",
    "nosql",
    "mongodb",
    "postgresql",
    "mysql",
    "rest",
    "graphql",
    "mqtt",
    "grpc",
    "json",
    "xml",
    "yaml",
    "linux",
    "windows",
    "macos",
    "android",
    "ios",
    "git",
    "github",
    "jenkins",
    "ansible",
    "terraform",
    "prometheus",
    "grafana",
    "elasticsearch",
    "open5gs",
    "magma",
    "openran",
    "ran",
    "upf",
    "smf",
    "amf",
    "nrf",
    "pcf",
    "harq",
    "rsrp",
    "rsrq",
    "sinr",
    "cqi",
    "bler",
    "prb",
    "enodeb",
    "gnodeb",
    "cbrs",
    "esim",
    "mec",
    "sdn",
    "nfv",
    "vnf",
    "mano",
    "onap",
    "oran",
}


class EntityVerifier:
    """
    Verifies entities mentioned in candidate responses through web search and knowledge base checking.
    """

    def __init__(
        self, settings: Settings, cache: Optional[PersistentCache] = None, cache_namespace_suffix: str = ""
    ) -> None:
        """
        Initializes an entity verifier.

        :param settings: Configuration settings.
        :param cache: Optional persistent cache.
        :param cache_namespace_suffix: Suffix for cache namespace to prevent cross-contamination.
        :return: None.
        """
        self.settings = settings
        self.cache = cache
        self.cache_namespace_suffix = cache_namespace_suffix

        # Initialize LLM for entity extraction
        self.llm = ChatOpenAI(model=settings.openai_model, temperature=0.2, openai_api_key=settings.openai_api_key)

        # Initialize search tool with namespace suffix
        self.search_tool = WebSearchTool(cache, max_results=5, cache_namespace_suffix=cache_namespace_suffix)

        # Load known technologies
        self.known_technologies = self._load_known_technologies()

        logger.info(f"Initialized entity verifier with {len(self.known_technologies)} known technologies")

    def _load_known_technologies(self) -> Set[str]:
        """
        Loads known technologies from a file and constants.

        :return: Set of known technology names.
        """
        known_techs = set(COMMON_TECHNOLOGIES)

        # Try to load from an Excel file if specified
        tech_file_path = self.settings.get_technologies_file_path()
        if tech_file_path and tech_file_path.exists():
            try:
                df = pd.read_excel(tech_file_path)
                if "T2 Example" in df.columns:
                    for tech in df["T2 Example"].dropna():
                        if isinstance(tech, str):
                            known_techs.add(tech.lower().strip())
                logger.info(f"Loaded technologies from {tech_file_path}")
            except Exception as e:
                logger.warning(f"Could not load technologies file: {e}")

        return known_techs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    async def extract_entities(self, response: str) -> VerificationEntities:
        """
        Extracts entities from a candidate response for verification.

        :param response: Candidate's response text.
        :return: VerificationEntities containing companies, technologies, and claims.
        """
        ColoredLogger.log_agent_thought("Extracting entities for verification")

        prompt = f"""Extract key entities from this response that should be verified through web search:

Response: {response}

Identify:
1. Company names (if any) - only real companies mentioned, not generic terms
2. Specific technologies, protocols, or standards mentioned
3. Technical claims that seem unusual or should be verified

Return as JSON:
{{
    "companies": ["list of company names"],
    "technologies": ["list of specific technologies/protocols/tools"],
    "implementations": ["list of specific claims that should be verified"]
}}

Focus on entities that are:
- Specific and verifiable (not generic terms)
- Mentioned as real technologies or companies
- Important for evaluating the truthfulness of the response"""

        llm_calls.labels(chain_type="ENTITY_EXTRACTION").inc()

        try:
            result = await self.llm.ainvoke(prompt)
            extracted = json.loads(result.content)

            # Track metrics
            entities_extracted.labels(entity_type="companies").inc(len(extracted.get("companies", [])))
            entities_extracted.labels(entity_type="technologies").inc(len(extracted.get("technologies", [])))
            entities_extracted.labels(entity_type="claims").inc(len(extracted.get("implementations", [])))

            ColoredLogger.log_agent_observation(
                f"Extracted {len(extracted.get('companies', []))} companies, "
                f"{len(extracted.get('technologies', []))} technologies, "
                f"{len(extracted.get('implementations', []))} claims"
            )

            return VerificationEntities(**extracted)

        except openai.RateLimitError:
            rate_limit_errors.inc()
            raise
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return VerificationEntities()

    async def verify_entities(self, entities: VerificationEntities) -> Dict[str, Any]:
        """
        Verifies extracted entities through web search and knowledge base.

        :param entities: Entities to verify.
        :return: Dictionary of verification results.
        """
        ColoredLogger.log_agent_thought("Starting entity verification")

        verification_results = {"companies": {}, "technologies": {}, "implementations": {}}

        # Create verification tasks
        tasks = []

        # Verify companies
        for company in entities.companies:
            tasks.append(("companies", company, self._verify_company(company)))

        # Verify technologies
        for tech in entities.technologies:
            tasks.append(("technologies", tech, self._verify_technology(tech)))

        # Verify implementation claims
        for claim in entities.implementations:
            tasks.append(("implementations", claim, self._verify_claim(claim)))

        # Execute all verifications in parallel
        for category, item, task in tasks:
            try:
                result = await task
                verification_results[category][item] = result
                verifications_performed.labels(verification_type=category).inc()
            except Exception as e:
                logger.warning(f"Verification failed for {category}/{item}: {str(e)}")
                verification_results[category][item] = f"Verification failed: {str(e)}"

        return verification_results

    async def _verify_company(self, company_name: str) -> str:
        """
        Verifies a company exists and is legitimate.

        :param company_name: Company name to verify.
        :return: Verification result string.
        """
        ColoredLogger.log_agent_thought(f"Verifying company: {company_name}")

        query = f'"{company_name}" company'
        search_result = await self.search_tool.search_async(query)

        # Check if a company appears in results
        if company_name.lower() not in search_result.lower():
            return f"No clear information found about {company_name}. May not exist or be very small/private."

        # Check for negative indicators
        negative_indicators = ["scam", "fraud", "fake company", "does not exist"]
        for indicator in negative_indicators:
            if indicator in search_result.lower():
                return f"Warning: Found negative information about {company_name}"

        return f"Found information about {company_name}: {search_result[:300]}"

    async def _verify_technology(self, tech_term: str) -> str:
        """
        Verifies a technology exists and is real.

        :param tech_term: Technology term to verify.
        :return: Verification result string.
        """
        ColoredLogger.log_agent_thought(f"Checking technology: {tech_term}")

        # Check if it's a known technology
        if tech_term.lower().strip() in self.known_technologies:
            return f"{tech_term} is a known/valid technology (found in knowledge base)"

        # Search for the technology
        query = f'"{tech_term}"'
        search_result = await self.search_tool.search_async(query)

        # Check results
        tech_lower = tech_term.lower()
        result_lower = search_result.lower()

        if tech_lower not in result_lower:
            return (
                f"No documentation found for {tech_term}. This may be a non-existent or incorrectly named technology."
            )

        # Check for negative indicators
        negative_indicators = ["does not exist", "no such", "not a real", "typo", "mistake", "did you mean"]
        for indicator in negative_indicators:
            if indicator in result_lower:
                return f"Search indicates {tech_term} is not a real/valid technology."

        return f"Found documentation for {tech_term}: {search_result[:300]}"

    async def _verify_claim(self, claim: str) -> str:
        """
        Verifies a specific technical claim.

        :param claim: Technical claim to verify.
        :return: Verification result string.
        """
        ColoredLogger.log_agent_thought(f"Verifying claim: {claim[:100]}...")

        # Extract key terms from the claim for search
        key_terms = self._extract_key_terms(claim)

        if not key_terms:
            return "Could not extract verifiable terms from claim"

        # Search for the claim or its key components
        query = " ".join(key_terms[:5])  # Limit to 5 key terms
        search_result = await self.search_tool.search_async(query)

        # Check if any key terms appear in results
        matches = sum(1 for term in key_terms if term.lower() in search_result.lower())

        if matches == 0:
            return f"No supporting information found for this claim"
        elif matches < len(key_terms) / 2:
            return f"Limited supporting information found. Claim may be exaggerated or partially incorrect."
        else:
            return f"Found some supporting information: {search_result[:300]}"

    @staticmethod
    def _extract_key_terms(text: str) -> List[str]:
        """
        Extracts key technical terms from a text for verification.

        :param text: Text from which to extract terms.
        :return: List of key terms.
        """
        # Extract capitalized words and technical terms
        terms = []

        # Find capitalized words (likely proper nouns or technologies)
        capitalized = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
        terms.extend(capitalized)

        # Find acronyms
        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
        terms.extend(acronyms)

        # Find numbers with units (like "5G", "10ms", etc)
        with_numbers = re.findall(r"\b\d+[A-Za-z]+\b", text)
        terms.extend(with_numbers)

        # Remove common words
        common_words = {"The", "This", "That", "These", "Those", "What", "When", "Where", "Why", "How"}
        terms = [t for t in terms if t not in common_words]

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        return unique_terms[:10]  # Limit to 10 terms
