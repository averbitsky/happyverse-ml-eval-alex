"""
Happyverse ML Evaluation System - Core Evaluator
Uses LangChain and LLMs for intelligent, context-aware evaluation
"""

import os
import re
import json
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.schema import OutputParserException as LangchainOutputParserException
import openai

from ddgs import DDGS
from colorama import init, Fore, Back, Style

# Load environment variables
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found after load_dotenv()"

# Initialize colorama for colored output
init(autoreset=True)

# Configure logging to suppress httpx and openai logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Configure main logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
}


# ============== Pydantic Models ==============


class TechnicalFeasibility(BaseModel):
    """Technical feasibility assessment"""

    assessment: str
    claims_needing_verification: List[str] = Field(default_factory=list)
    impossible_claims: List[str] = Field(default_factory=list)
    questionable_claims: List[str] = Field(default_factory=list)


class Verifiability(BaseModel):
    """Verifiability assessment"""

    verifiable_elements: List[str] = Field(default_factory=list)
    missing_specifics: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)


class AuthenticityIndicators(BaseModel):
    """Authenticity assessment"""

    positive_signs: List[str] = Field(default_factory=list)
    negative_signs: List[str] = Field(default_factory=list)


class PlausibilityResult(BaseModel):
    """Plausibility evaluation result"""

    plausibility_score: int = Field(ge=0, le=100)
    technical_feasibility: TechnicalFeasibility
    verifiability: Verifiability
    authenticity_indicators: AuthenticityIndicators
    overall_assessment: str


class AccuracyAssessment(BaseModel):
    """Technical accuracy assessment"""

    correct_concepts: List[str] = Field(default_factory=list)
    uncertain_concepts: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    terminology_usage: str = ""


class DepthAnalysis(BaseModel):
    """Depth of knowledge analysis"""

    level: str = Field(pattern="^(shallow|moderate|deep)$")
    evidence_of_experience: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    explanation: str


class ProblemSolving(BaseModel):
    """Problem solving assessment"""

    approach_quality: str
    best_practices: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)


class RoleFit(BaseModel):
    """Role fit assessment"""

    alignment_score: int = Field(ge=0, le=100)
    strengths_for_role: List[str] = Field(default_factory=list)
    gaps_for_role: List[str] = Field(default_factory=list)
    explanation: str


class TechnicalResult(BaseModel):
    """Technical proficiency evaluation result"""

    technical_score: int = Field(ge=0, le=100)
    accuracy_assessment: AccuracyAssessment
    depth_analysis: DepthAnalysis
    problem_solving: ProblemSolving
    role_fit: RoleFit
    overall_assessment: str


class ClarityAssessment(BaseModel):
    """Clarity assessment"""

    score: int = Field(ge=0, le=100)
    organization: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class RelevanceAssessment(BaseModel):
    """Relevance assessment"""

    answers_question: bool
    directness_score: int = Field(ge=0, le=100)
    unnecessary_content: List[str] = Field(default_factory=list)
    missing_elements: List[str] = Field(default_factory=list)


class PersuasivenessAssessment(BaseModel):
    """Persuasiveness assessment"""

    score: int = Field(ge=0, le=100)
    compelling_elements: List[str] = Field(default_factory=list)
    weak_areas: List[str] = Field(default_factory=list)
    storytelling_quality: str


class ProfessionalismAssessment(BaseModel):
    """Professionalism assessment"""

    appropriate_tone: bool
    technical_communication: str
    red_flags: List[str] = Field(default_factory=list)


class CommunicationResult(BaseModel):
    """Communication evaluation result"""

    communication_score: int = Field(ge=0, le=100)
    clarity: ClarityAssessment
    relevance: RelevanceAssessment
    persuasiveness: PersuasivenessAssessment
    professionalism: ProfessionalismAssessment
    overall_assessment: str


class SpecificRecommendations(BaseModel):
    """Specific hiring recommendations"""

    if_hire: str
    if_reject: str


class SynthesisResult(BaseModel):
    """Final synthesis and recommendation"""

    recommendation_level: str = Field(pattern="^(Strong Yes|Weak Yes|Weak No|Strong No)$")
    confidence: float = Field(ge=0.0, le=1.0)
    key_strengths: List[str] = Field(default_factory=list)
    critical_concerns: List[str] = Field(default_factory=list)
    deal_breakers: List[str] = Field(default_factory=list)
    hiring_risk: str = Field(pattern="^(low|medium|high)$")
    comparison_to_typical: str
    specific_recommendations: SpecificRecommendations
    detailed_rationale: str


class VerificationEntities(BaseModel):
    """Entities extracted for verification"""

    companies: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    implementations: List[str] = Field(default_factory=list)


class QuestionEvaluation(BaseModel):
    """Complete evaluation for a single question"""

    question: str
    response: str
    plausibility: Optional[PlausibilityResult] = None
    technical: Optional[TechnicalResult] = None
    communication: Optional[CommunicationResult] = None
    verification: Dict[str, Any] = Field(default_factory=dict)


class AggregateScore(BaseModel):
    """Aggregate score statistics"""

    mean: float
    min: float
    max: float
    all_scores: List[float]


class AggregateScores(BaseModel):
    """All aggregate scores"""

    plausibility: AggregateScore
    technical: AggregateScore
    communication: AggregateScore


class EvaluationMetadata(BaseModel):
    """Metadata about the evaluation"""

    timestamp: str
    model_used: str
    questions_evaluated: int
    known_technologies_count: int


class FinalEvaluation(BaseModel):
    """Complete evaluation result"""

    individual_evaluations: List[QuestionEvaluation]
    aggregate_scores: AggregateScores
    recommendation: SynthesisResult
    evaluation_metadata: EvaluationMetadata


class RecommendationLevel(Enum):
    STRONG_YES = "Strong Yes"
    WEAK_YES = "Weak Yes"
    WEAK_NO = "Weak No"
    STRONG_NO = "Strong No"


# ============== Helper Classes ==============


class ColoredLogger:
    """Helper class for colored logging of LLM interactions"""

    @staticmethod
    def log_llm_input(prompt_type: str, content: dict):
        """Log LLM input with color for headers only"""
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f"{Fore.CYAN}LLM INPUT [{prompt_type}]:")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        for key, value in content.items():
            # Only trim search results
            if key == "search_results" and isinstance(value, str) and len(value) > 500:
                print(f"{key}: {value[:500]}...")
            else:
                print(f"{key}: {value}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

    @staticmethod
    def log_llm_output(prompt_type: str, result: dict):
        """Log LLM output with selective coloring"""
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
    def log_agent_thought(thought: str):
        """Log agent reasoning steps in yellow"""
        print(f"{Fore.YELLOW}[AGENT THOUGHT]: {thought}{Style.RESET_ALL}")

    @staticmethod
    def log_agent_action(action: str, input_str: str = None):
        """Log agent actions in cyan"""
        print(f"{Fore.CYAN}[AGENT ACTION]: {action}{Style.RESET_ALL}")
        if input_str:
            print(f"   Input: {input_str}")

    @staticmethod
    def log_agent_observation(observation: str):
        """Log agent observations in magenta"""
        # Truncate very long observations
        if len(observation) > 200:
            observation = observation[:200] + "..."
        print(f"{Fore.MAGENTA}[OBSERVATION]: {observation}{Style.RESET_ALL}")


class WebSearchTool:
    """Custom web search wrapper with caching"""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.cache = {}  # Cache search results
        self.ddgs = DDGS()
        logger.info("Initialized web search tool")

    def run(self, query: str) -> str:
        """Run a web search query with caching"""

        # Check cache first
        if query in self.cache:
            ColoredLogger.log_agent_thought(f"Found cached result for: {query}")
            return self.cache[query]

        ColoredLogger.log_agent_action("Web Search", query)

        try:
            # Use DDGS API
            results = list(self.ddgs.text(query, max_results=self.max_results))
            if results:
                # Format results
                formatted = []
                for r in results:
                    formatted.append(f"{r.get('title', '')}: {r.get('body', '')}")
                result = " ".join(formatted)
            else:
                result = "No results found"

            # Cache the result
            self.cache[query] = result

            ColoredLogger.log_agent_observation(result)

            return result

        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            ColoredLogger.log_agent_observation(error_msg)
            self.cache[query] = error_msg
            return error_msg


# ============== Main Evaluator Class ==============


class CandidateEvaluator:
    """
    Main evaluation pipeline using LangChain and LLMs for intelligent assessment.
    No hardcoded patterns - all evaluation is done through LLM reasoning.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        """
        Initialize the evaluator with LangChain components.

        Args:
            openai_api_key: OpenAI API key for LLM access
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        logger.info(f"Initializing CandidateEvaluator with model: {model}")

        self.llm = ChatOpenAI(model=model, temperature=0.2, openai_api_key=openai_api_key)

        # Use custom search tool wrapper with caching
        self.search_tool = WebSearchTool(max_results=5)

        # Initialize known technologies set
        self.known_technologies = self._initialize_known_technologies()
        logger.info(f"Loaded {len(self.known_technologies)} known technologies")

        # Initialize evaluation chains using modern LangChain patterns
        self.plausibility_chain = self._create_plausibility_chain()
        self.technical_chain = self._create_technical_chain()
        self.communication_chain = self._create_communication_chain()
        self.synthesis_chain = self._create_synthesis_chain()

        # Create verification tools
        self.verification_tools = self._create_verification_tools()

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=3)

        logger.info("CandidateEvaluator initialization complete")

    def _initialize_known_technologies(self) -> Set[str]:
        """
        Initialize a set of known technologies from Excel file in data folder
        """
        known_techs = set()

        # Load technologies from Excel file in data folder
        try:
            df = pd.read_excel("data/Tools and Technology.xlsx")
            if "T2 Example" in df.columns:
                # Extract technologies from T2 Example column
                for tech in df["T2 Example"].dropna():
                    if isinstance(tech, str):
                        # Add lowercase version for case-insensitive matching
                        known_techs.add(tech.lower().strip())
                logger.info(f"Loaded {len(known_techs)} technologies from Excel file")
        except Exception as e:
            logger.warning(f"Could not load Technologies Excel file from data folder: {e}")

        # Add common technologies from constant
        known_techs.update(COMMON_TECHNOLOGIES)

        return known_techs

    def _extract_technologies_from_text(self, text: str) -> Set[str]:
        """
        Extract technology names from text (like job description)
        """
        technologies = set()

        # Common patterns for technology names
        # Look for capitalized words, acronyms, and specific patterns
        patterns = [
            r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",  # Capitalized words
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b\w+\.js\b",  # JavaScript libraries
            r"\b\w+\.\w+\b",  # Dotted names (e.g., Node.js)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Filter out common non-technology words
                if len(match) > 1 and not match.lower() in {"the", "and", "for", "with", "from", "our", "your"}:
                    technologies.add(match.lower())

        return technologies

    def _is_known_technology(self, tech_term: str) -> bool:
        """
        Check if a technology is in our known set
        """
        return tech_term.lower().strip() in self.known_technologies

    def _create_chain_with_logging(self, template: str, chain_name: str):
        """Create a chain with automatic input/output logging and retry logic"""
        prompt = ChatPromptTemplate.from_template(template)
        parser = JsonOutputParser()
        base_chain = prompt | self.llm | parser

        # Create a wrapper that logs inputs and outputs with retry logic
        class LoggedChain:
            def __init__(self, chain, name):
                self.chain = chain
                self.name = name

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=60),
                retry=retry_if_exception_type(openai.RateLimitError),
            )
            def invoke(self, inputs):
                ColoredLogger.log_llm_input(self.name, inputs)
                try:
                    result = self.chain.invoke(inputs)
                    ColoredLogger.log_llm_output(self.name, result)
                    return result
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit hit for {self.name}: {str(e)}")
                    raise

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=60),
                retry=retry_if_exception_type(openai.RateLimitError),
            )
            async def ainvoke(self, inputs):
                ColoredLogger.log_llm_input(self.name, inputs)
                try:
                    result = await self.chain.ainvoke(inputs)
                    ColoredLogger.log_llm_output(self.name, result)
                    return result
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit hit for {self.name}: {str(e)}")
                    raise

        return LoggedChain(base_chain, chain_name)

    def _create_plausibility_chain(self):
        """Create chain for evaluating plausibility of claims"""

        template = """You are an expert technical interviewer evaluating the plausibility of a candidate's response.

Job Context:
{job_description}

Interview Question:
{question}

Candidate Response:
{response}

Search Results (if available):
{search_results}

Carefully analyze this response for plausibility. Consider:

1. TECHNICAL FEASIBILITY
- Are the technical claims physically/logically possible?
- Do the technologies mentioned actually exist? (Check search results if provided)
- Are the described implementations realistic?
- Could the described systems actually work as claimed?

2. CONSISTENCY & LOGIC
- Are the claims internally consistent?
- Do timelines and sequences make logical sense?
- Are there contradictions within the response?

3. VERIFIABILITY
- Can the companies/projects mentioned be verified? (Check search results)
- Are specific details provided that could be fact-checked?
- Does the candidate avoid providing verifiable details when they should?

4. DEPTH & AUTHENTICITY
- Does the response show genuine understanding or just buzzwords?
- Are there signs of fabrication or exaggeration?
- Does the level of detail match claimed experience?

IMPORTANT: If any technology terms or implementations seem unusual, explicitly note them for further verification.

Provide your evaluation in the following JSON format:
{{
    "plausibility_score": <0-100>,
    "technical_feasibility": {{
        "assessment": "detailed assessment of technical claims",
        "claims_needing_verification": ["list of claims that should be web-verified"],
        "impossible_claims": ["list of any technically impossible claims"],
        "questionable_claims": ["list of questionable but not impossible claims"]
    }},
    "verifiability": {{
        "verifiable_elements": ["list of things that can be fact-checked"],
        "missing_specifics": ["areas where specifics should have been provided but weren't"],
        "red_flags": ["specific concerns about verification"]
    }},
    "authenticity_indicators": {{
        "positive_signs": ["signs of genuine experience"],
        "negative_signs": ["signs of fabrication or exaggeration"]
    }},
    "overall_assessment": "comprehensive paragraph explaining the plausibility evaluation"
}}"""

        return self._create_chain_with_logging(template, "PLAUSIBILITY")

    def _create_technical_chain(self):
        """Create chain for evaluating technical proficiency"""

        template = """You are a senior technical expert evaluating a candidate's technical proficiency for a specific role.

Job Requirements:
{job_description}

Interview Question:
{question}

Candidate Response:
{response}

Additional Context (if available):
{technical_context}

Evaluate the technical proficiency demonstrated in this response:

1. TECHNICAL ACCURACY
- Are technical concepts explained correctly?
- Is the terminology used appropriately?
- Are there any technical errors or misunderstandings?

2. DEPTH OF KNOWLEDGE
- Does the response show surface-level or deep understanding?
- Are complex concepts handled appropriately?
- Is there evidence of hands-on experience vs theoretical knowledge?

3. PROBLEM-SOLVING APPROACH
- Is the technical approach sound?
- Are best practices followed?
- Would the described solutions actually work?

4. ROLE ALIGNMENT
- Does the technical level match the job requirements?
- Are the right technologies and skills demonstrated?
- Any critical gaps in knowledge for this role?

IMPORTANT: For any technologies or implementations you're uncertain about, mark them for verification rather than assuming they're wrong.

Provide your evaluation in JSON format:
{{
    "technical_score": <0-100>,
    "accuracy_assessment": {{
        "correct_concepts": ["correctly explained technical concepts"],
        "uncertain_concepts": ["concepts that need verification"],
        "errors": ["clear technical errors or misunderstandings"],
        "terminology_usage": "assessment of terminology use"
    }},
    "depth_analysis": {{
        "level": "shallow/moderate/deep",
        "evidence_of_experience": ["indicators of real experience"],
        "knowledge_gaps": ["identified gaps"],
        "explanation": "detailed assessment of technical depth"
    }},
    "problem_solving": {{
        "approach_quality": "assessment of problem-solving approach",
        "best_practices": ["followed best practices"],
        "concerns": ["technical concerns with approach"]
    }},
    "role_fit": {{
        "alignment_score": <0-100>,
        "strengths_for_role": ["relevant strengths"],
        "gaps_for_role": ["critical gaps"],
        "explanation": "how well they fit the technical requirements"
    }},
    "overall_assessment": "comprehensive technical evaluation paragraph"
}}"""

        return self._create_chain_with_logging(template, "TECHNICAL")

    def _create_communication_chain(self):
        """Create chain for evaluating communication skills"""

        template = """You are evaluating a candidate's communication skills based on their interview response.

Question Asked:
{question}

Candidate Response:
{response}

Evaluate their communication effectiveness:

1. CLARITY & STRUCTURE
- Is the response well-organized?
- Are ideas expressed clearly?
- Is there a logical flow to the answer?

2. DIRECTNESS & RELEVANCE
- Does the response directly answer the question?
- Is there unnecessary tangenting or filler?
- Are concrete examples provided?

3. PERSUASIVENESS & ENGAGEMENT
- Is the response engaging and compelling?
- Does it build confidence in their abilities?
- Is there effective storytelling?

4. PROFESSIONAL COMMUNICATION
- Is the tone appropriate for a technical interview?
- Is there a good balance between technical detail and accessibility?
- Any communication red flags?

Provide your evaluation in JSON format:
{{
    "communication_score": <0-100>,
    "clarity": {{
        "score": <0-100>,
        "organization": "assessment of response organization",
        "strengths": ["communication strengths"],
        "weaknesses": ["areas for improvement"]
    }},
    "relevance": {{
        "answers_question": true/false,
        "directness_score": <0-100>,
        "unnecessary_content": ["tangents or filler content"],
        "missing_elements": ["what should have been addressed but wasn't"]
    }},
    "persuasiveness": {{
        "score": <0-100>,
        "compelling_elements": ["what made the response compelling"],
        "weak_areas": ["where the response was weak"],
        "storytelling_quality": "assessment of narrative ability"
    }},
    "professionalism": {{
        "appropriate_tone": true/false,
        "technical_communication": "assessment of technical communication",
        "red_flags": ["any concerning communication patterns"]
    }},
    "overall_assessment": "comprehensive communication evaluation paragraph"
}}"""

        return self._create_chain_with_logging(template, "COMMUNICATION")

    def _create_synthesis_chain(self):
        """Create chain for synthesizing all evaluations into final recommendation"""

        template = """You are making a final hiring recommendation based on multiple evaluation dimensions.

Evaluation Results:
{evaluations}

Job Description:
{job_description}

Based on all evaluations, provide a comprehensive hiring recommendation:

1. Synthesize the key findings across all dimensions
2. Identify the most critical strengths and weaknesses
3. Assess overall fit for the role
4. Make a clear recommendation with justification

Consider:
- Are there any deal-breakers (e.g., dishonesty, fundamental lack of required skills)?
- How do they compare to typical candidates for this level role?
- What is the risk/benefit of hiring this candidate?

Provide your recommendation in JSON format:
{{
    "recommendation_level": "Strong Yes/Weak Yes/Weak No/Strong No",
    "confidence": <0.0-1.0>,
    "key_strengths": ["top 3-5 strengths"],
    "critical_concerns": ["top 3-5 concerns"],
    "deal_breakers": ["any absolute disqualifiers"],
    "hiring_risk": "low/medium/high",
    "comparison_to_typical": "how they compare to typical candidates for this role",
    "specific_recommendations": {{
        "if_hire": "what to focus on if hiring",
        "if_reject": "feedback for the candidate"
    }},
    "detailed_rationale": "comprehensive explanation of the recommendation"
}}"""

        return self._create_chain_with_logging(template, "SYNTHESIS")

    def _create_verification_tools(self) -> List:
        """Create verification tools for fact-checking"""

        def search_company(company_name: str) -> str:
            """Search for company information"""
            ColoredLogger.log_agent_thought(f"Verifying company: {company_name}")

            query = f'"{company_name}" company'
            results = self.search_tool.run(query)

            # Check if results are relevant
            if company_name.lower() not in results.lower():
                msg = f"No clear information found about {company_name}. This company may not exist or may be very small/private."
                ColoredLogger.log_agent_observation(msg)
                return msg

            ColoredLogger.log_agent_observation(f"Found information about {company_name}")
            return results[:500]  # Truncate to save tokens

        def search_technology(tech_term: str) -> str:
            """Search for technology/protocol information"""
            ColoredLogger.log_agent_thought(f"Checking technology: {tech_term}")

            # First check if it's a known technology
            if self._is_known_technology(tech_term):
                msg = f"{tech_term} is a known/valid technology (found in knowledge base)"
                ColoredLogger.log_agent_observation(msg)
                return msg

            # If not known, search for it
            ColoredLogger.log_agent_thought(f"{tech_term} not in known technologies, searching web...")

            query = f'"{tech_term}"'
            results = self.search_tool.run(query)

            # More nuanced check
            results_lower = results.lower()
            tech_lower = tech_term.lower()

            if tech_lower not in results_lower:
                msg = f"No documentation found for {tech_term}. This may be a non-existent or incorrectly named technology."
                ColoredLogger.log_agent_observation(msg)
                return msg

            # Check for negative indicators
            negative_indicators = ["does not exist", "no such", "not a real", "typo", "mistake", "should be"]
            for indicator in negative_indicators:
                if indicator in results_lower:
                    msg = f"Search indicates {tech_term} is not a real/valid technology."
                    ColoredLogger.log_agent_observation(msg)
                    return msg

            ColoredLogger.log_agent_observation(f"Found documentation for {tech_term}")
            return results[:500]  # Truncate to save tokens

        def verify_technical_claim(claim: str) -> str:
            """Verify a specific technical implementation claim"""
            ColoredLogger.log_agent_thought(f"Verifying claim: {claim}")

            # Focus on verifying the actual claim, not just the technologies
            verify_prompt = f"""Analyze this technical claim and identify what needs verification:

Claim: {claim}

Identify:
1. The core assertion being made (what is actually being claimed?)
2. Any metrics or numbers that seem unrealistic
3. Technical approaches that seem impossible or contradictory

DO NOT just list the technologies mentioned. Focus on the CLAIM itself.
Return 2-3 specific things about this claim that should be verified."""

            try:
                # No retry decorator here - just handle the exception
                # The LLM call itself already has retry logic in the chain
                response = self.llm.invoke(verify_prompt)
                verification_points = response.content
                ColoredLogger.log_agent_thought(f"Verification points: {verification_points}")

                # Search for the actual claim or key parts of it
                search_query = claim[:100]  # Take first part of claim
                search_result = self.search_tool.run(f'"{search_query}"')

                if claim[:50].lower() in search_result.lower():
                    return f"Found references to similar implementations:\n{search_result[:500]}"
                else:
                    # Try searching for the core concept
                    concept_search = self.search_tool.run(claim[:50])
                    return f"Verification results:\n{verification_points}\n\nSearch results:\n{concept_search[:500]}"

            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit during claim verification: {str(e)}")
                # Return a partial result instead of failing
                return "Rate limit reached during verification - claim could not be fully verified"
            except Exception as e:
                logger.warning(f"Claim verification failed: {str(e)}")
                return "Could not verify this specific claim"

        return [search_company, search_technology, verify_technical_claim]

    def _targeted_verification(self, claims_to_verify: VerificationEntities) -> Dict:
        """
        Perform targeted verification of specific claims
        """
        logger.info("=" * 50)
        logger.info("STARTING TARGETED VERIFICATION")

        verification_results = {"companies": {}, "technologies": {}, "implementations": {}}

        # Get verification functions
        search_company, search_technology, verify_technical_claim = self.verification_tools

        # Verify companies
        for company in claims_to_verify.companies:
            result = search_company(company)
            verification_results["companies"][company] = result

        # Verify technologies
        for tech in claims_to_verify.technologies:
            result = search_technology(tech)
            verification_results["technologies"][tech] = result

        # Verify specific implementations
        for impl in claims_to_verify.implementations:
            result = verify_technical_claim(impl)
            verification_results["implementations"][impl] = result

        logger.info("TARGETED VERIFICATION COMPLETE")
        logger.info("=" * 50)

        return verification_results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    def _extract_entities_for_verification(self, response: str) -> VerificationEntities:
        """
        Extract entities that need verification from a response
        """
        ColoredLogger.log_agent_thought("Extracting entities for verification...")

        prompt = f"""Extract key entities from this response that should be verified through web search:

Response: {response}

Identify:
1. Company names (if any)
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

        try:
            result = self.llm.invoke(prompt)
            extracted = json.loads(result.content)
            ColoredLogger.log_agent_observation(
                f"Extracted {len(extracted.get('companies', []))} companies, {len(extracted.get('technologies', []))} technologies"
            )
            return VerificationEntities(**extracted)
        except openai.RateLimitError:
            raise  # Let retry decorator handle this
        except Exception as e:
            logger.warning(f"Entity extraction failed: {str(e)}")
            return VerificationEntities()

    def _clean_question(self, question: str) -> str:
        """Remove 'Question X:' prefix from questions"""
        # Remove patterns like "Question 1:", "Q1:", etc.
        cleaned = re.sub(r"^(Question\s+\d+:|Q\d+:)\s*", "", question, flags=re.IGNORECASE)
        return cleaned.strip()

    async def _evaluate_single_qa_async(self, job_description: str, question: str, response: str) -> QuestionEvaluation:
        """
        Asynchronously evaluate a single question-answer pair
        """
        # Clean the question
        clean_question = self._clean_question(question)

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Evaluating: {clean_question}")

        # First, extract entities to verify
        entities_to_verify = self._extract_entities_for_verification(response)

        # Perform targeted verification if needed
        search_results = {}
        if entities_to_verify.companies or entities_to_verify.technologies:
            search_results = self._targeted_verification(entities_to_verify)

        # Only truncate search results for LLM
        truncated_search_results = {}
        if search_results:
            for key, value in search_results.items():
                if isinstance(value, dict):
                    truncated_search_results[key] = {k: v[:500] if isinstance(v, str) else v for k, v in value.items()}
                else:
                    truncated_search_results[key] = value

        # Run all three evaluations in parallel
        logger.info("Running parallel evaluations...")

        plausibility_task = self.plausibility_chain.ainvoke(
            {
                "job_description": job_description,
                "question": clean_question,
                "response": response,
                "search_results": (
                    json.dumps(truncated_search_results, indent=2)
                    if truncated_search_results
                    else "No search performed"
                ),
            }
        )

        technical_task = self.technical_chain.ainvoke(
            {
                "job_description": job_description,
                "question": clean_question,
                "response": response,
                "technical_context": (
                    json.dumps(truncated_search_results.get("technologies", {}), indent=2)[:2000]
                    if truncated_search_results
                    else "No additional context"
                ),
            }
        )

        communication_task = self.communication_chain.ainvoke({"question": clean_question, "response": response})

        # Wait for all evaluations to complete
        plausibility_result, technical_result, communication_result = await asyncio.gather(
            plausibility_task, technical_task, communication_task
        )

        # Note: The plausibility chain already identifies claims needing verification during its evaluation
        # We've already done the verification upfront, so we don't need to do it again
        # This avoids redundant LLM calls and potential rate limit issues

        return QuestionEvaluation(
            question=clean_question,
            response=response,
            plausibility=PlausibilityResult(**plausibility_result),
            technical=TechnicalResult(**technical_result),
            communication=CommunicationResult(**communication_result),
            verification=search_results,
        )

    async def evaluate_transcript_async(
        self, job_description: str, questions: List[str], transcript: str
    ) -> FinalEvaluation:
        """
        Main evaluation pipeline with async execution
        """
        logger.info("=" * 70)
        logger.info("STARTING TRANSCRIPT EVALUATION (ASYNC)")
        logger.info("=" * 70)

        # Extract technologies from job description and add to known set
        job_techs = self._extract_technologies_from_text(job_description)
        self.known_technologies.update(job_techs)
        logger.info(f"Added {len(job_techs)} technologies from job description to known set")

        # Parse transcript into Q&A pairs
        qa_pairs = self._parse_transcript(transcript, questions)
        logger.info(f"Parsed {len(qa_pairs)} Q&A pairs")

        # Evaluate all Q&A pairs in parallel
        evaluation_tasks = [
            self._evaluate_single_qa_async(job_description, question, response) for question, response in qa_pairs
        ]

        logger.info(f"Evaluating {len(evaluation_tasks)} questions in parallel...")
        all_evaluations = await asyncio.gather(*evaluation_tasks)

        # Synthesize final recommendation
        logger.info("\nSynthesizing final recommendation...")

        # Create a summary of evaluations
        evaluation_summary = []
        for eval in all_evaluations:
            summary = {
                "question": eval.question,
                "scores": {
                    "plausibility": eval.plausibility.plausibility_score if eval.plausibility else 0,
                    "technical": eval.technical.technical_score if eval.technical else 0,
                    "communication": eval.communication.communication_score if eval.communication else 0,
                },
                "key_issues": {
                    "impossible_claims": (
                        eval.plausibility.technical_feasibility.impossible_claims if eval.plausibility else []
                    ),
                    "technical_errors": eval.technical.accuracy_assessment.errors if eval.technical else [],
                    "red_flags": eval.plausibility.verifiability.red_flags if eval.plausibility else [],
                },
            }
            evaluation_summary.append(summary)

        synthesis_result = await self.synthesis_chain.ainvoke(
            {"evaluations": json.dumps(evaluation_summary, indent=2), "job_description": job_description}
        )

        # Compile final results
        final_results = FinalEvaluation(
            individual_evaluations=all_evaluations,
            aggregate_scores=self._calculate_aggregate_scores(all_evaluations),
            recommendation=SynthesisResult(**synthesis_result),
            evaluation_metadata=EvaluationMetadata(
                timestamp=datetime.now().isoformat(),
                model_used=self.llm.model_name,
                questions_evaluated=len(all_evaluations),
                known_technologies_count=len(self.known_technologies),
            ),
        )

        logger.info("=" * 70)
        logger.info("EVALUATION COMPLETE!")
        logger.info("=" * 70)

        return final_results

    def evaluate_transcript(self, job_description: str, questions: List[str], transcript: str) -> FinalEvaluation:
        """
        Main evaluation entry point - uses async evaluation
        """
        return asyncio.run(self.evaluate_transcript_async(job_description, questions, transcript))

    def _parse_transcript(self, transcript: str, questions: List[str]) -> List[Tuple[str, str]]:
        """
        Parse transcript into Q&A pairs using pattern matching and validation
        """
        logger.debug("Parsing transcript into Q&A pairs")
        qa_pairs = []

        # Try to split by Q1, Q2, Q3 patterns
        pattern = r"Q\d+:.*?(?=Q\d+:|$)"
        matches = re.findall(pattern, transcript, re.DOTALL)

        if matches:
            for match in matches:
                parts = match.split(":", 1)
                if len(parts) == 2:
                    response = parts[1].strip()
                    response = re.sub(r"^A\d+:", "", response).strip()

                    q_num = re.search(r"Q(\d+)", parts[0])
                    if q_num and int(q_num.group(1)) <= len(questions):
                        question = questions[int(q_num.group(1)) - 1]
                        qa_pairs.append((question, response))
        else:
            # Fallback parsing
            remaining_text = transcript
            for question in questions:
                if question in remaining_text:
                    parts = remaining_text.split(question, 1)
                    if len(parts) == 2:
                        answer_text = parts[1]
                        next_q_start = len(answer_text)
                        for next_q in questions:
                            if next_q in answer_text:
                                next_q_start = min(next_q_start, answer_text.index(next_q))

                        answer = answer_text[:next_q_start].strip()
                        qa_pairs.append((question, answer))
                        remaining_text = answer_text[next_q_start:]

        return qa_pairs

    def _calculate_aggregate_scores(self, evaluations: List[QuestionEvaluation]) -> AggregateScores:
        """
        Calculate aggregate scores across all evaluations
        """
        if not evaluations:
            return AggregateScores(
                plausibility=AggregateScore(mean=0, min=0, max=0, all_scores=[]),
                technical=AggregateScore(mean=0, min=0, max=0, all_scores=[]),
                communication=AggregateScore(mean=0, min=0, max=0, all_scores=[]),
            )

        plausibility_scores = [e.plausibility.plausibility_score for e in evaluations if e.plausibility]

        technical_scores = [e.technical.technical_score for e in evaluations if e.technical]

        communication_scores = [e.communication.communication_score for e in evaluations if e.communication]

        def create_aggregate(scores: List[float]) -> AggregateScore:
            if not scores:
                return AggregateScore(mean=0, min=0, max=0, all_scores=[])
            return AggregateScore(mean=sum(scores) / len(scores), min=min(scores), max=max(scores), all_scores=scores)

        aggregate = AggregateScores(
            plausibility=create_aggregate(plausibility_scores),
            technical=create_aggregate(technical_scores),
            communication=create_aggregate(communication_scores),
        )

        logger.info(
            f"Aggregate scores - Plausibility: {aggregate.plausibility.mean:.1f}, "
            f"Technical: {aggregate.technical.mean:.1f}, "
            f"Communication: {aggregate.communication.mean:.1f}"
        )

        return aggregate


if __name__ == "__main__":
    import os

    # Example usage
    print(f"{Fore.CYAN}{'=' * 70}")
    print(f"{Fore.CYAN}HAPPYVERSE ML EVALUATION SYSTEM")
    print(f"{Fore.CYAN}{'=' * 70}\n")

    evaluator = CandidateEvaluator(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

    # Load sample data
    logger.info("Loading input data...")

    with open("data/job_description.txt", "r") as f:
        job_description = f.read()

    with open("data/questions.txt", "r") as f:
        questions = f.read().split("\n\n")

    with open("data/transcripts/sample_bad.txt", "r") as f:
        transcript = f.read()

    # Run evaluation
    results = evaluator.evaluate_transcript(job_description=job_description, questions=questions, transcript=transcript)

    # Save results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/evaluation_{timestamp}.json"

    # Convert Pydantic models to dict for JSON serialization
    results_dict = results.model_dump()

    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n{Fore.GREEN}{'=' * 70}")
    print(f"{Fore.GREEN}RESULTS SAVED TO: {output_file}")
    print(f"{Fore.GREEN}{'=' * 70}\n")

    # Print summary
    print(f"{Fore.YELLOW}EVALUATION SUMMARY")
    print(f"{Fore.YELLOW}{'=' * 70}")
    print(f"Recommendation: {Fore.LIGHTGREEN_EX}{results.recommendation.recommendation_level}")
    print(f"Confidence: {Fore.LIGHTGREEN_EX}{results.recommendation.confidence}")
    print(f"\nScores:")
    print(f"  • Plausibility: {Fore.WHITE}{results.aggregate_scores.plausibility.mean:.1f}/100")
    print(f"  • Technical: {Fore.WHITE}{results.aggregate_scores.technical.mean:.1f}/100")
    print(f"  • Communication: {Fore.WHITE}{results.aggregate_scores.communication.mean:.1f}/100")

    if results.recommendation.key_strengths:
        print(f"\n{Fore.GREEN}Key Strengths:")
        for strength in results.recommendation.key_strengths:
            print(f"  • {strength}")

    if results.recommendation.critical_concerns:
        print(f"\n{Fore.RED}Critical Concerns:")
        for concern in results.recommendation.critical_concerns:
            print(f"  • {concern}")

    if results.recommendation.deal_breakers:
        print(f"\n{Fore.LIGHTRED_EX}Deal Breakers:")
        for breaker in results.recommendation.deal_breakers:
            print(f"  ⚠️  {breaker}")

    print(f"{Fore.YELLOW}{'=' * 70}")
