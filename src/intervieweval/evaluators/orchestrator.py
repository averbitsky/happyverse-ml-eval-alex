"""
Evaluation orchestrator that coordinates all evaluators and manages parallel execution.
"""

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from intervieweval.cache.manager import PersistentCache
from intervieweval.config.settings import Settings
from intervieweval.evaluators.communication import CommunicationEvaluator
from intervieweval.evaluators.plausibility import PlausibilityEvaluator
from intervieweval.evaluators.synthesis import SynthesisEvaluator
from intervieweval.evaluators.technical import TechnicalEvaluator
from intervieweval.models.evaluation import (
    AggregateScore,
    AggregateScores,
    BatchEvaluationResult,
    EvaluationMetadata,
    FinalEvaluation,
    QuestionEvaluation,
)
from intervieweval.prompts.manager import PromptManager
from intervieweval.tools.verification import EntityVerifier
from intervieweval.utils.logging import ColoredLogger
from intervieweval.utils.metrics import batch_size as batch_size_metric
from intervieweval.utils.metrics import (
    evaluation_counter,
    evaluation_duration,
    parallel_tasks,
)

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Orchestrates the entire evaluation process, coordinating all evaluators and managing parallel execution.
    """

    def __init__(self, settings: Settings, prompt_manager: PromptManager, cache: Optional[PersistentCache] = None):
        """
        Initializes the evaluation orchestrator.

        :param settings: Configuration settings.
        :param prompt_manager: Prompt template manager.
        :param cache: Optional persistent cache.
        :return: None.
        """
        self.settings = settings
        self.prompt_manager = prompt_manager
        self.cache = cache

        # Note: We'll create evaluators per transcript to avoid state contamination
        # Only initialize synthesis evaluator here as it doesn't have state issues
        self.synthesis_evaluator = SynthesisEvaluator(settings, prompt_manager, cache)

        logger.info("Initialized evaluation orchestrator")

    def _create_evaluators_for_transcript(self, transcript_hash: str):
        """
        Creates new evaluator instances for a specific transcript. This prevents state contamination between different
        transcript evaluations.

        :param transcript_hash: Hash of the transcript for cache namespacing.
        :return: Tuple of (plausibility_evaluator, technical_evaluator, communication_evaluator, verifier).
        """
        # Create evaluators with transcript-specific namespacing
        plausibility_evaluator = PlausibilityEvaluator(
            self.settings, self.prompt_manager, self.cache, cache_namespace_suffix=transcript_hash
        )
        technical_evaluator = TechnicalEvaluator(
            self.settings, self.prompt_manager, self.cache, cache_namespace_suffix=transcript_hash
        )
        communication_evaluator = CommunicationEvaluator(
            self.settings, self.prompt_manager, self.cache, cache_namespace_suffix=transcript_hash
        )

        # Create a fresh verifier for this transcript
        verifier = EntityVerifier(self.settings, self.cache, cache_namespace_suffix=transcript_hash)

        return plausibility_evaluator, technical_evaluator, communication_evaluator, verifier

    @staticmethod
    def _get_transcript_hash(transcript: str) -> str:
        """
        Generates a hash for a transcript for cache namespacing.

        :param transcript: Transcript content.
        :return: Short hash string
        """
        return hashlib.md5(transcript.encode()).hexdigest()[:8]

    async def evaluate_single_qa(
        self,
        job_description: str,
        question: str,
        response: str,
        plausibility_evaluator,
        technical_evaluator,
        communication_evaluator,
        verifier,
    ) -> QuestionEvaluation:
        """
        Evaluates a single question-answer pair.

        :param job_description: Job requirements.
        :param question: Interview question.
        :param response: Candidate's response.
        :param plausibility_evaluator: Plausibility evaluator instance.
        :param technical_evaluator: Technical evaluator instance.
        :param communication_evaluator: Communication evaluator instance.
        :param verifier: Entity verifier instance.
        :return: QuestionEvaluation with all assessment results.
        """
        # Clean question (remove Q1: prefixes etc)
        clean_question = self._clean_question(question)

        logger.info(f"Evaluating question: {clean_question[:100]}...")

        # Extract and verify entities
        entities = await verifier.extract_entities(response)
        verification_results = {}

        if entities.companies or entities.technologies or entities.implementations:
            verification_results = await verifier.verify_entities(entities)

        # Format search results for prompts
        search_results = self._format_verification_results(verification_results)

        # Run all evaluations in parallel
        parallel_tasks.inc()

        try:
            plausibility_task = plausibility_evaluator.evaluate(
                job_description=job_description,
                question=clean_question,
                response=response,
                search_results=search_results,
            )

            technical_task = technical_evaluator.evaluate(
                job_description=job_description,
                question=clean_question,
                response=response,
                technical_context=search_results,
            )

            communication_task = communication_evaluator.evaluate(question=clean_question, response=response)

            # Wait for all evaluations to complete
            plausibility_result, technical_result, communication_result = await asyncio.gather(
                plausibility_task, technical_task, communication_task
            )

        finally:
            parallel_tasks.dec()

        return QuestionEvaluation(
            question=clean_question,
            response=response,
            plausibility=plausibility_result,
            technical=technical_result,
            communication=communication_result,
            verification=verification_results,
        )

    async def evaluate_transcript(self, job_description: str, questions: List[str], transcript: str) -> FinalEvaluation:
        """
        Evaluates a transcript.

        :param job_description: Job requirements.
        :param questions: List of interview questions.
        :param transcript: Full transcript text.
        :return: FinalEvaluation with complete results.
        """
        start_time = time.time()
        evaluation_counter.inc()

        ColoredLogger.log_info("Starting transcript evaluation")

        # Get transcript hash for namespacing
        transcript_hash = self._get_transcript_hash(transcript)

        # Create fresh evaluators for this transcript
        plausibility_evaluator, technical_evaluator, communication_evaluator, verifier = (
            self._create_evaluators_for_transcript(transcript_hash)
        )

        # Parse transcript into Q&A pairs
        qa_pairs = self._parse_transcript(transcript, questions)
        logger.info(f"Parsed {len(qa_pairs)} Q&A pairs from transcript")

        # Evaluate all Q&A pairs with controlled parallelism
        semaphore = asyncio.Semaphore(self.settings.max_parallel_evaluations)

        async def evaluate_with_limit(q, r):
            async with semaphore:
                return await self.evaluate_single_qa(
                    job_description,
                    q,
                    r,
                    plausibility_evaluator,
                    technical_evaluator,
                    communication_evaluator,
                    verifier,
                )

        # Create evaluation tasks
        evaluation_tasks = [evaluate_with_limit(question, response) for question, response in qa_pairs]

        # Execute all evaluations
        all_evaluations = await asyncio.gather(*evaluation_tasks)

        # Synthesize final recommendation
        ColoredLogger.log_info("Synthesizing final recommendation")
        synthesis_result = await self.synthesis_evaluator.evaluate(
            evaluations=all_evaluations, job_description=job_description
        )

        # Calculate aggregate scores
        aggregate_scores = self._calculate_aggregate_scores(all_evaluations)

        # Calculate metrics
        duration = time.time() - start_time
        cache_hit_rate = self._calculate_cache_hit_rate()

        # Create the final evaluation
        final_eval = FinalEvaluation(
            individual_evaluations=all_evaluations,
            aggregate_scores=aggregate_scores,
            recommendation=synthesis_result,
            evaluation_metadata=EvaluationMetadata(
                timestamp=datetime.now().isoformat(),
                model_used=self.settings.openai_model,
                questions_evaluated=len(all_evaluations),
                known_technologies_count=(
                    len(verifier.known_technologies) if hasattr(verifier, "known_technologies") else 0
                ),
                evaluation_duration_seconds=duration,
                cache_hit_rate=cache_hit_rate,
            ),
        )

        evaluation_duration.observe(duration)
        ColoredLogger.log_success(f"Evaluation complete in {duration:.2f} seconds")

        return final_eval

    async def evaluate_batch(
        self, job_description: str, questions: List[str], transcripts: List[str]
    ) -> BatchEvaluationResult:
        """
        Evaluates multiple transcripts in a batch. Each transcript gets its own evaluator instances to prevent
        contamination.

        :param job_description: Job requirements.
        :param questions: List of interview questions.
        :param transcripts: List of transcript texts.
        :return: BatchEvaluationResult with all evaluations.
        """
        batch_start = time.time()
        batch_size_metric.observe(len(transcripts))

        ColoredLogger.log_info(f"Starting batch evaluation of {len(transcripts)} transcripts")

        # Evaluate transcripts sequentially
        evaluations = []
        for i, transcript in enumerate(transcripts, 1):
            ColoredLogger.log_info(f"Evaluating transcript {i}/{len(transcripts)}")
            evaluation = await self.evaluate_transcript(job_description, questions, transcript)
            evaluations.append(evaluation)

        batch_duration = time.time() - batch_start

        return BatchEvaluationResult(
            evaluations=evaluations,
            batch_metadata={
                "batch_size": len(transcripts),
                "total_duration_seconds": batch_duration,
                "average_duration_seconds": batch_duration / len(transcripts) if transcripts else 0,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    def _clean_question(question: str) -> str:
        """
        Removes 'Question X:' or 'QX:' prefixes from questions.

        :param question: Original question text.
        :return: Cleaned question text.
        """
        cleaned = re.sub(r"^(Question\s+\d+:|Q\d+:)\s*", "", question, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _parse_transcript(transcript: str, questions: List[str]) -> List[Tuple[str, str]]:
        """
        Parses a transcript into question-answer pairs and removes any repeated question text from the beginning of
        each extracted answer.

        :param transcript: Full transcript text.
        :param questions: List of interview questions.
        :return: List of (question, answer) tuples.
        """

        def _normalize_answer_prefix(ans: str, quest: str) -> str:
            """
            Iteratively removes leading artifacts from the start of an answer:
            - Any 'A<number>:' prefix (e.g., 'A1:', 'A12:')
            - The question text itself (whitespace-insensitive, with optional quotes/punct)
            Repeats until no further change.

            :param ans: Raw answer text.
            :param quest: Corresponding question text.
            :return: Cleaned answer text.
            """
            a = ans.lstrip()

            # Build a whitespace-insensitive regex for the question
            q_tokens = quest.strip().split()
            if not q_tokens:
                return a
            q_ws_insensitive = r"\s*".join(map(re.escape, q_tokens))

            # Pattern for a leading 'A<number>:' like 'A1:' or 'A12:'
            a_prefix = re.compile(r"^\s*A\d+:\s*", flags=re.IGNORECASE)

            # Pattern for the question at the very start.
            # IMPORTANT: After the question, only allow spaces/tabs (NO newlines)
            # before optional trailing quotes/punctuation, so we don't consume the
            # opening quote of the actual answer on the next line.
            q_prefix = re.compile(
                rf""" ^
                     [ \t]* [\'\"\u201C\u201D\u2018\u2019\`\(\[]*         # optional leading quotes/brackets
                     (?:Question\s*\d*\s*:\s*)?                          # optional 'Question:' prefix
                     {q_ws_insensitive}                                   # question content (flex whitespace)
                     (?:[ \t]*[\'\"\u201C\u201D\u2018\u2019\`\)\]]*)?     # optional trailing quotes (same line)
                     (?:[ \t]*[:\-–—]+)?                                  # optional trailing punctuation (same line)
                     [ \t]*                                               # trailing spaces/tabs only (no newline)
                """,
                flags=re.IGNORECASE | re.VERBOSE,
            )

            for _ in range(4):  # small cap to avoid infinite loops
                before = a
                a = a_prefix.sub("", a)  # remove leading A<n>:
                a = q_prefix.sub("", a)  # remove leading question (tightened trailing part)
                a = a_prefix.sub("", a)  # remove A<n>: again if it reappears
                if a == before:
                    break
                a = a.lstrip()
            return a.lstrip()

        qa_pairs: List[Tuple[str, str]] = []

        # Try 'Qn:'-style blocks first
        pattern = r"Q\d+:.*?(?=Q\d+:|$)"
        matches = re.findall(pattern, transcript, re.DOTALL)

        if matches:
            for match in matches:
                parts = match.split(":", 1)
                if len(parts) == 2:
                    response = parts[1].strip()

                    # If there is an 'A<n>:' immediately at the start, remove it first
                    response = re.sub(r"^A\d+:\s*", "", response).strip()

                    q_num = re.search(r"Q(\d+)", parts[0])
                    if q_num:
                        idx = int(q_num.group(1)) - 1
                        if 0 <= idx < len(questions):
                            question = questions[idx]
                            # Normalize: remove any leading question text / A<n>: artifacts
                            response = _normalize_answer_prefix(response, question)
                            qa_pairs.append((question, response))
        else:
            # Fallback: split by actual question strings in order
            remaining_text = transcript
            for question in questions:
                if question in remaining_text:
                    parts = remaining_text.split(question, 1)
                    if len(parts) == 2:
                        answer_text = parts[1]
                        # locate the earliest next question
                        next_q_start = len(answer_text)
                        for next_q in questions:
                            if next_q in answer_text:
                                next_q_start = min(next_q_start, answer_text.index(next_q))
                        answer = answer_text[:next_q_start].strip()
                        answer = _normalize_answer_prefix(answer, question)
                        qa_pairs.append((question, answer))
                        remaining_text = answer_text[next_q_start:]

        return qa_pairs

    @staticmethod
    def _calculate_aggregate_scores(evaluations: List[QuestionEvaluation]) -> AggregateScores:
        """
        Calculates aggregate scores across all evaluations.

        :param evaluations: List of QuestionEvaluation objects.
        :return: AggregateScores with mean, min, max for each category.
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
            """
            Creates an AggregateScore from a list of scores.

            :param scores: List of individual scores.
            :return: AggregateScore object.
            """
            if not scores:
                return AggregateScore(mean=0, min=0, max=0, all_scores=[])
            return AggregateScore(mean=sum(scores) / len(scores), min=min(scores), max=max(scores), all_scores=scores)

        return AggregateScores(
            plausibility=create_aggregate(plausibility_scores),
            technical=create_aggregate(technical_scores),
            communication=create_aggregate(communication_scores),
        )

    @staticmethod
    def _format_verification_results(verification_results: Dict[str, Any]) -> str:
        """
        Formats verification results for inclusion in prompts.

        :param verification_results: Dictionary of verification results.
        :return: Formatted string.
        """
        if not verification_results:
            return "No verification performed"

        formatted_parts = []

        if "companies" in verification_results:
            formatted_parts.append("Company Verifications:")
            for company, result in verification_results["companies"].items():
                formatted_parts.append(f"  - {company}: {result[:200]}")

        if "technologies" in verification_results:
            formatted_parts.append("\nTechnology Verifications:")
            for tech, result in verification_results["technologies"].items():
                formatted_parts.append(f"  - {tech}: {result[:200]}")

        if "implementations" in verification_results:
            formatted_parts.append("\nClaim Verifications:")
            for claim, result in verification_results["implementations"].items():
                formatted_parts.append(f"  - {claim[:100]}: {result[:200]}")

        return "\n".join(formatted_parts) if formatted_parts else "No verification results"

    def _calculate_cache_hit_rate(self) -> float:
        """
        Calculates cache hit rate from cache statistics.

        :return: Cache hit rate as a float between 0 and 1.
        """
        if not self.cache:
            return 0.0

        stats = self.cache.get_stats()
        total_requests = stats.get("total_hits", 0) + stats.get("total_misses", 0)
        if total_requests == 0:
            return 0.0

        return stats.get("total_hits", 0) / total_requests
