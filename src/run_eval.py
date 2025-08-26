"""
Main entry point for running candidate evaluations with visualization.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from colorama import Fore, Style, init
from dotenv import load_dotenv

from intervieweval.cache.manager import PersistentCache
from intervieweval.config.settings import Settings
from intervieweval.evaluators.orchestrator import EvaluationOrchestrator
from intervieweval.prompts.manager import PromptManager
from intervieweval.utils.logging import setup_logging
from intervieweval.utils.metrics import setup_metrics
from intervieweval.visualization.visualization import EvaluationVisualizer

# Load environment variables
load_dotenv()

# Initialize colorama
init(autoreset=True)


def load_transcripts(transcript_dir: Path) -> Dict[str, str]:
    """
    Loads all transcript files from the transcripts directory.

    :param transcript_dir: Directory containing transcript files.
    :return: Dictionary mapping filename to transcript content.
    """
    transcripts = {}

    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    # Read all .txt files in the transcripts directory
    txt_files = list(transcript_dir.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No transcript files found in {transcript_dir}")

    for file_path in txt_files:
        with open(file_path, "r") as f:
            content = f.read()
            transcripts[file_path.name] = content
            print(f"  ✓ Loaded {file_path.name}")

    return transcripts


def load_data_files(settings: Settings) -> tuple[str, List[str]]:
    """
    Loads job description and questions from files.

    :param settings: Application settings.
    :return: Tuple of (job_description, questions).
    """
    # Load job description
    job_path = settings.get_job_description_path()
    if not job_path.exists():
        raise FileNotFoundError(f"Job description file not found: {job_path}")

    with open(job_path, "r") as f:
        job_description = f.read()

    # Load questions
    questions_path = settings.get_questions_path()
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, "r") as f:
        questions = f.read().split("\n\n")

    return job_description, questions


def print_evaluation_summary(evaluation, candidate_name: str) -> None:
    """
    Prints a formatted summary of an evaluation.

    :param evaluation: Evaluation result object.
    :param candidate_name: Name of the candidate.
    :return: None
    """
    print(f"\n{Fore.CYAN}{'=' * 70}")
    print(f"EVALUATION SUMMARY: {candidate_name}")
    print(f"{'=' * 70}{Style.RESET_ALL}")

    # Overall recommendation
    rec = evaluation.recommendation
    color = Fore.GREEN if "Yes" in rec.recommendation_level else Fore.RED
    print(f"\n{color}RECOMMENDATION: {rec.recommendation_level}")
    print(f"Confidence: {rec.confidence:.2f}{Style.RESET_ALL}")

    # Scores
    scores = evaluation.aggregate_scores
    print(f"\nSCORES:")
    print(f"  • Plausibility:   {scores.plausibility.mean:.1f}/100")
    print(f"  • Technical:      {scores.technical.mean:.1f}/100")
    print(f"  • Communication:  {scores.communication.mean:.1f}/100")

    # Key findings
    if rec.key_strengths:
        print(f"\n{Fore.GREEN}STRENGTHS:{Style.RESET_ALL}")
        for strength in rec.key_strengths[:3]:
            print(f"  ✓ {strength}")

    if rec.critical_concerns:
        print(f"\n{Fore.YELLOW}CONCERNS:{Style.RESET_ALL}")
        for concern in rec.critical_concerns[:3]:
            print(f"  ⚠ {concern}")

    if rec.deal_breakers:
        print(f"\n{Fore.RED}DEAL-BREAKERS:{Style.RESET_ALL}")
        for breaker in rec.deal_breakers:
            print(f"  ✗ {breaker}")

    # Metadata
    meta = evaluation.evaluation_metadata
    print(f"\n{Fore.CYAN}PERFORMANCE:{Style.RESET_ALL}")
    print(f"  • Duration: {meta.evaluation_duration_seconds:.2f} seconds")
    print(f"  • Cache hit rate: {meta.cache_hit_rate:.2%}")


def print_cache_stats(cache: PersistentCache) -> None:
    """
    Print cache statistics.

    :param cache: PersistentCache instance.
    :return: None.
    """
    stats = cache.get_stats()

    print(f"\n{Fore.CYAN}CACHE STATISTICS:{Style.RESET_ALL}")
    print(f"  • Total items: {stats['total_items']}")
    print(f"  • Memory cache: {stats['memory_cache_items']} items")
    print(f"  • Database size: {stats['db_size_mb']:.2f} MB")
    print(f"  • Hit rate: {stats['hit_rate']:.2%}")
    print(f"  • Total hits: {stats['total_hits']}")
    print(f"  • Total misses: {stats['total_misses']}")

    if stats.get("namespaces"):
        print(f"\n  Namespace breakdown:")
        for namespace, count in stats["namespaces"].items():
            print(f"    - {namespace}: {count} items")


async def evaluate_candidates(
    orchestrator: EvaluationOrchestrator,
    job_description: str,
    questions: List[str],
    transcripts: Dict[str, str],
    output_dir: Path,
    generate_visualizations: bool = True,
) -> List:
    """
    Evaluates multiple candidate transcripts with optional visualizations.

    :param orchestrator: Evaluation orchestrator.
    :param job_description: Job description text.
    :param questions: List of interview questions.
    :param transcripts: Dictionary mapping filename to transcript content.
    :param output_dir: Directory to save results.
    :param generate_visualizations: Whether to generate visualization slides.
    :return: List of evaluation results.
    """
    # Create the output directory
    output_dir.mkdir(exist_ok=True)

    # Create the visualization directory if needed
    if generate_visualizations:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        visualizer = EvaluationVisualizer(viz_dir)

    # Extract transcript contents
    transcript_files = list(transcripts.keys())
    transcript_contents = list(transcripts.values())

    # Evaluate transcripts
    if len(transcript_contents) == 1:
        print(f"\n{Fore.YELLOW}Evaluating single transcript: {transcript_files[0]}...{Style.RESET_ALL}")
        evaluation = await orchestrator.evaluate_transcript(job_description, questions, transcript_contents[0])
        evaluations = [evaluation]
    else:
        print(f"\n{Fore.YELLOW}Evaluating {len(transcript_contents)} transcripts in batch...{Style.RESET_ALL}")
        batch_result = await orchestrator.evaluate_batch(job_description, questions, transcript_contents)
        evaluations = batch_result.evaluations

        # Print batch statistics
        print(f"\n{Fore.GREEN}Batch evaluation complete!{Style.RESET_ALL}")
        print(f"  • Total time: {batch_result.batch_metadata['total_duration_seconds']:.2f} seconds")
        print(f"  • Average per transcript: {batch_result.batch_metadata['average_duration_seconds']:.2f} seconds")

    # Save results and generate visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualization_paths = []

    for i, (filename, evaluation) in enumerate(zip(transcript_files, evaluations), 1):
        # Create filename based on the original transcript name
        base_name = Path(filename).stem
        candidate_name = base_name.replace("_", " ").title()

        # Save JSON result
        result_file = output_dir / f"evaluation_{base_name}_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(evaluation.model_dump(), f, indent=2)
        print(f"  ✓ Saved evaluation for {filename} to {result_file}")

        # Generate visualization if requested
        if generate_visualizations:
            try:
                viz_path = visualizer.create_evaluation_slide(evaluation.model_dump(), candidate_name=candidate_name)
                visualization_paths.append(viz_path)
                print(f"  ✓ Generated visualization: {viz_path}")
            except Exception as e:
                print(f"  ⚠ Failed to generate visualization: {e}")

    # Generate comparison visualization if multiple candidates
    if generate_visualizations and len(evaluations) > 1:
        try:
            candidate_names = [Path(f).stem.replace("_", " ").title() for f in transcript_files]
            comparison_path = visualizer.create_batch_comparison([e.model_dump() for e in evaluations], candidate_names)
            print(f"  ✓ Generated comparison visualization: {comparison_path}")
        except Exception as e:
            print(f"  ⚠ Failed to generate comparison visualization: {e}")

    # Save summary if multiple evaluations
    if len(evaluations) > 1:
        summary = {
            "timestamp": timestamp,
            "total_evaluations": len(evaluations),
            "evaluations": [
                {
                    "transcript": filename,
                    "recommendation": eval.recommendation.recommendation_level,
                    "confidence": eval.recommendation.confidence,
                    "risk": eval.recommendation.hiring_risk,
                    "scores": {
                        "plausibility": eval.aggregate_scores.plausibility.mean,
                        "technical": eval.aggregate_scores.technical.mean,
                        "communication": eval.aggregate_scores.communication.mean,
                    },
                }
                for filename, eval in zip(transcript_files, evaluations)
            ],
        }

        summary_file = output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary to {summary_file}")

    return evaluations


def main() -> None:
    """
    Main function to run the candidate evaluation system.
    1. Parse command-line arguments.
    2. Load settings and initialize components (cache, prompts, orchestrator).
    3. Load job description, questions, and transcripts.
    4. Run evaluations and generate visualizations.
    5. Print summaries and cache statistics.
    6. Handle errors gracefully and provide user feedback.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Happyverse ML Evaluation System - Candidate Evaluator")

    # Model configuration
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument("--parallel", type=int, default=3, help="Maximum parallel evaluations")

    # Feature flags
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-metrics", action="store_true", help="Disable Prometheus metrics server")
    parser.add_argument("--no-visualization", action="store_true", help="Disable visualization generation")
    parser.add_argument("--clean-cache", action="store_true", help="Clean expired cache entries before starting")
    parser.add_argument("--export-cache", action="store_true", help="Export cache after evaluation")

    # Optional overrides (rarely needed)
    parser.add_argument("--api-key", default=None, help="OpenAI API key (overrides environment variable)")

    args = parser.parse_args()

    print(f"{Fore.CYAN}{'=' * 70}")
    print("HAPPYVERSE ML EVALUATION SYSTEM")
    print(f"{'=' * 70}{Style.RESET_ALL}\n")

    # Setup logging
    setup_logging()

    # Load settings
    settings_kwargs = {
        "openai_model": args.model,
        "max_parallel_evaluations": args.parallel,
        "enable_metrics": not args.no_metrics,
    }

    if args.api_key:
        settings_kwargs["openai_api_key"] = args.api_key

    try:
        settings = Settings(**settings_kwargs)
    except Exception as e:
        print(f"{Fore.RED}Error loading settings: {e}{Style.RESET_ALL}")
        print("Make sure OPENAI_API_KEY is set in your .env file or passed via --api-key")
        sys.exit(1)

    # Setup metrics server
    if settings.enable_metrics:
        setup_metrics(settings.metrics_port)
        print(f"✓ Metrics server started on port {settings.metrics_port}")

    # Initialize cache
    cache = None
    if not args.no_cache:
        print("Initializing persistent cache...")
        cache_dir = settings.get_cache_dir()
        cache_dir.mkdir(exist_ok=True)

        cache = PersistentCache(cache_dir=str(cache_dir), cache_name=settings.cache_db_name)

        if args.clean_cache:
            expired = cache.clean_expired()
            print(f"  ✓ Cleaned {expired} expired cache entries")

        print_cache_stats(cache)

    # Initialize prompt manager
    print("\nInitializing prompt manager...")
    prompt_file = settings.get_prompts_path()
    prompt_manager = PromptManager(str(prompt_file))

    # Validate prompts
    validation = prompt_manager.validate_prompts()
    if not all(validation.values()):
        print(f"{Fore.RED}Warning: Some prompts failed validation{Style.RESET_ALL}")
        for name, valid in validation.items():
            status = "✓" if valid else "✗"
            print(f"  {status} {name}")
    else:
        print(f"  ✓ All prompts validated successfully")

    # Load data
    print("\nLoading data files...")
    try:
        # Load job description and questions
        job_description, questions = load_data_files(settings)
        print(f"  ✓ Loaded job description")
        print(f"  ✓ Loaded {len(questions)} questions")

        # Load all transcripts from directory
        transcript_dir = settings.get_transcripts_dir()
        print(f"\nLoading transcripts from {transcript_dir}...")
        transcripts = load_transcripts(transcript_dir)
        print(f"  ✓ Loaded {len(transcripts)} transcript(s)")

    except (FileNotFoundError, ValueError) as e:
        print(f"{Fore.RED}Error loading data: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # Initialize orchestrator
    print("\nInitializing evaluation orchestrator...")
    orchestrator = EvaluationOrchestrator(settings, prompt_manager, cache)

    # Run evaluations
    try:
        output_dir = settings.get_output_dir()
        evaluations = asyncio.run(
            evaluate_candidates(
                orchestrator,
                job_description,
                questions,
                transcripts,
                output_dir,
                generate_visualizations=not args.no_visualization,
            )
        )

        # Print summaries
        for filename, evaluation in zip(transcripts.keys(), evaluations):
            candidate_name = Path(filename).stem.replace("_", " ").title()
            print_evaluation_summary(evaluation, candidate_name)

        # Print cache statistics
        if cache:
            print_cache_stats(cache)

            # Export cache if requested
            if args.export_cache:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = output_dir / f"cache_export_{timestamp}.db"
                if cache.export_cache(str(export_path)):
                    print(f"\n✓ Cache exported to {export_path}")

        # Clean expired entries
        if cache and not args.clean_cache:  # Only if not already cleaned
            expired_count = cache.clean_expired()
            if expired_count > 0:
                print(f"\nCleaned {expired_count} expired cache entries")

        # Print comparison if multiple candidates
        if len(evaluations) > 1:
            print(f"\n{Fore.CYAN}CANDIDATE COMPARISON:{Style.RESET_ALL}")
            print(f"{'Candidate':<25} {'Recommendation':<15} {'Confidence':<12} {'Risk':<10}")
            print("-" * 62)

            for filename, eval in zip(transcripts.keys(), evaluations):
                candidate = Path(filename).stem.replace("_", " ").title()
                candidate = candidate[:22] + "..." if len(candidate) > 25 else candidate
                rec = eval.recommendation
                color = Fore.GREEN if "Yes" in rec.recommendation_level else Fore.RED
                print(
                    f"{candidate:<25} "
                    f"{color}{rec.recommendation_level:<15}{Style.RESET_ALL} "
                    f"{rec.confidence:<12.2f} "
                    f"{rec.hiring_risk:<10}"
                )

        print(f"\n{Fore.GREEN}✓ Evaluation complete!{Style.RESET_ALL}")

        if not args.no_visualization:
            print(f"\n{Fore.CYAN} Visualizations have been generated in: {output_dir}/visualizations/{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error during evaluation: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
