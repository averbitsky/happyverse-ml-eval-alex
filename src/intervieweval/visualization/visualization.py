"""
Visualization generator for creating professional evaluation slides.
"""

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

logger = logging.getLogger(__name__)

# Readable defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)
plt.style.use("seaborn-v0_8-whitegrid")


class EvaluationVisualizer:
    """
    Creates a one-slide visualization for a candidate evaluation.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize the visualizer with an output directory.

        :param output_dir: Directory to save visualizations.
        :return: None.
        """
        self.output_dir = output_dir or Path("output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_font_size = 12  # for the score graph

        # Unified palette
        self.colors = {
            "Strong Yes": "#00C851",
            "Weak Yes": "#00BFA5",
            "Weak No": "#FFB300",
            "Strong No": "#FF3547",
            "plausibility": "#9B59B6",
            "technical": "#1ABC9C",
            "communication": "#34495E",
            "background": "#ECF0F1",
            "card_bg": "#FFFFFF",
            "text": "#2C3E50",
            "text_secondary": "#616161",
            "info": "#2196F3",
            "danger": "#FF3547",
        }

        # Shaded fills for bottom boxes
        self.shaded = {
            "strengths_fill": "#E9F7EF",  # pale green
            "concerns_fill": "#FEF5E7",  # pale yellow
            "breakers_fill": "#FDEDEC",  # pale red
        }

        self.score_thresholds = {"excellent": 85, "good": 70, "fair": 50, "poor": 0}

    def create_evaluation_slide(
        self, evaluation_data: Dict[str, Any], candidate_name: str = "Candidate", save_path: Optional[Path] = None
    ) -> Path:
        """
        Creates a professional one-slide evaluation summary for one candidate.

        :param evaluation_data: Dictionary with evaluation results.
        :param candidate_name: Name of the candidate.
        :param save_path: Optional path to save the visualization.
        :return: Path to the saved visualization.
        """
        fig = plt.figure(figsize=(16, 12), dpi=150)
        fig.patch.set_facecolor(self.colors["background"])

        recommendation = evaluation_data["recommendation"]
        scores = evaluation_data["aggregate_scores"]
        individual_evals = evaluation_data.get("individual_evaluations", [])

        # Header
        self._add_header(fig, candidate_name)

        # Layout
        gs = GridSpec(
            5,
            3,
            figure=fig,
            height_ratios=[1.0, 1.6, 2.8, 1.6, 0.55],  # Increased score chart from 2.4 to 2.8
            width_ratios=[1, 1, 1],
            hspace=0.30,
            wspace=0.30,
            left=0.06,
            right=0.94,
            top=0.86,
            bottom=0.06,
        )

        # Row 0: decision strip
        ax_decision = fig.add_subplot(gs[0, :])
        self._create_decision_strip(ax_decision, recommendation)

        # Row 1: candidate profile
        ax_profile = fig.add_subplot(gs[1, :])
        self._create_candidate_profile_card(ax_profile, recommendation)

        # Row 2: scores (enhanced graph)
        ax_scores = fig.add_subplot(gs[2, :])
        self._create_enhanced_score_display(ax_scores, scores, individual_evals)

        # Row 3: three shaded boxes (strengths | concerns | deal-breakers)
        ax_strengths = fig.add_subplot(gs[3, 0])
        ax_concerns = fig.add_subplot(gs[3, 1])
        ax_breakers = fig.add_subplot(gs[3, 2])
        self._add_three_shaded_boxes(ax_strengths, ax_concerns, ax_breakers, recommendation)

        # Row 4: footer
        ax_footer = fig.add_subplot(gs[4, :])
        self._add_metrics_bar(ax_footer, evaluation_data)

        # Save
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"evaluation_{candidate_name.replace(' ', '_')}_{timestamp}.pdf"

        plt.savefig(
            save_path, format="pdf", bbox_inches="tight", facecolor=self.colors["background"], edgecolor="none", dpi=150
        )
        plt.close()
        logger.info(f"Saved evaluation visualization to {save_path}")
        return save_path

    def _add_header(self, fig, candidate_name: str):
        """
        Adds a header with a title and the candidate's name.

        :param fig: Figure object.
        :param candidate_name: Name of the candidate.
        :return: None.
        """
        # Extra top padding before the title
        fig.text(
            0.50,
            0.94,  # lowered from 0.958
            "Interview Evaluation Report",
            ha="center",
            va="top",
            fontsize=24,
            fontweight="400",
            color=self.colors["text"],
        )

        fig.text(
            0.50,
            0.90,  # lowered from 0.918
            f"{candidate_name}",
            ha="center",
            va="top",
            fontsize=30,
            fontweight="bold",
            color=self.colors["text"],
        )

    def _create_decision_strip(self, ax, recommendation: Dict[str, Any]) -> None:
        """
        Creates a decision strip with confidence, decision, and hiring risk.

        :param ax: Axes object.
        :param recommendation: Recommendation dictionary.
        :return: None.
        """
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        card = FancyBboxPatch(
            (0.03, 0.12),
            0.94,
            0.76,
            boxstyle="round,pad=0.012",
            facecolor=self.colors["card_bg"],
            edgecolor=self.colors[recommendation["recommendation_level"]],
            linewidth=2.2,
            alpha=0.99,
        )
        ax.add_patch(card)

        confidence = recommendation["confidence"]
        conf_color = self._get_confidence_color(confidence)
        risk = recommendation["hiring_risk"]
        rec_level = recommendation["recommendation_level"]

        # Map risk colors to decision palette to avoid multiple reds
        risk_colors = {
            "low": self.colors["Strong Yes"],
            "medium": self.colors["Weak No"],
            "high": self.colors["Strong No"],
        }

        # Confidence (left)
        ax.text(0.18, 0.70, "CONFIDENCE", fontsize=13, fontweight="600", color=self.colors["text_secondary"])
        ax.text(0.18, 0.44, f"{confidence:.0%}", fontsize=23, fontweight="bold", color=conf_color)
        bar_x, bar_y, bar_w, bar_h = 0.11, 0.28, 0.26, 0.06
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="#E0E0E0", alpha=0.36))
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w * max(0.0, min(1.0, confidence)), bar_h, facecolor=conf_color))

        # Decision (middle) — swapped with Risk per request
        ax.text(0.50, 0.70, "DECISION", fontsize=13, fontweight="600", ha="center", color=self.colors["text_secondary"])
        ax.text(0.50, 0.42, rec_level, fontsize=20, fontweight="bold", ha="center", color=self.colors[rec_level])

        # Risk (right)
        ax.text(
            0.82, 0.70, "HIRING RISK", fontsize=13, fontweight="600", ha="center", color=self.colors["text_secondary"]
        )
        ax.text(0.82, 0.42, risk.upper(), fontsize=22, fontweight="bold", ha="center", color=risk_colors[risk])

    def _create_candidate_profile_card(self, ax, recommendation: Dict[str, Any]) -> None:
        """
        Creates a candidate profile card with detailed rationale.

        :param ax: Axes object.
        :param recommendation: Recommendation dictionary.
        :return: None.
        """
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        card = FancyBboxPatch(
            (0.03, 0.05),
            0.94,
            0.90,
            boxstyle="round,pad=0.018",
            facecolor=self.colors["card_bg"],
            edgecolor="#D6D6D6",
            linewidth=1.6,
            alpha=0.99,
        )
        ax.add_patch(card)

        # Title
        ax.text(
            0.50,
            0.77,
            "Candidate Profile",
            fontsize=16,
            fontweight="700",
            ha="center",
            color=self.colors["text_secondary"],
        )

        # Use detailed_rationale with fallback
        profile = recommendation.get(
            "detailed_rationale", recommendation.get("comparison_to_typical", "No rationale provided.")
        )

        # Allow maximum width in the box
        wrap_chars_profile = 150
        prof_lines = textwrap.fill(profile, width=wrap_chars_profile).split("\n")
        line_h = 0.13
        block_h = len(prof_lines) * line_h

        # Vertically center the text block
        start_y = 0.36 + block_h / 2.0
        y = start_y
        for line in prof_lines:
            ax.text(0.50, y, line, fontsize=13.5, color=self.colors["text"], ha="center", va="center")
            y -= line_h

    def _create_enhanced_score_display(self, ax, scores: Dict[str, Any], individual_evals: List[Dict]) -> None:
        """
        Creates an enhanced score display with proper question circles.

        :param ax: Axes object.
        :param scores: Aggregate the scores dictionary.
        :param individual_evals: List of individual question evaluations.
        :return: None.
        """
        ax.clear()
        ax.set_xlim(-10, 110)
        ax.set_ylim(-2.0, 3.5)

        categories = [
            ("Communication", scores["communication"]["mean"], self.colors["communication"]),
            ("Technical", scores["technical"]["mean"], self.colors["technical"]),
            ("Plausibility", scores["plausibility"]["mean"], self.colors["plausibility"]),
        ]

        # Collect question-by-question scores per category
        question_scores = {"Communication": [], "Technical": [], "Plausibility": []}
        for eval in individual_evals:
            question_scores["Communication"].append(
                eval.get("communication", {}).get("communication_score")
                if isinstance(eval.get("communication"), dict)
                else None
            )
            question_scores["Technical"].append(
                eval.get("technical", {}).get("technical_score") if isinstance(eval.get("technical"), dict) else None
            )
            question_scores["Plausibility"].append(
                eval.get("plausibility", {}).get("plausibility_score")
                if isinstance(eval.get("plausibility"), dict)
                else None
            )

        for i, (label, avg_score, color) in enumerate(categories):
            y_pos = 2 - i  # reverse vertical order

            # Background bar and main bar
            ax.barh(y_pos, 100, height=0.6, color="#E0E0E0", alpha=0.3, zorder=1)
            ax.barh(y_pos, avg_score, height=0.6, color=color, alpha=0.7, zorder=2)

            # Circles for individual questions
            q_scores = question_scores[label]
            num_questions = len(q_scores)
            for q_idx, q_score in enumerate(q_scores):
                if q_score is None:
                    continue
                if num_questions == 1:
                    y_offset = 0
                elif num_questions == 2:
                    y_offset = (q_idx - 0.5) * 0.2
                else:
                    y_offset = (q_idx - (num_questions - 1) / 2) * 0.15

                ax.scatter(q_score, y_pos + y_offset, s=100, color=color, edgecolors="white", linewidth=2, zorder=3)

                # Q labels under each bar
                ax.text(
                    q_score,
                    y_pos - 0.35,
                    f"Q{q_idx + 1}",
                    fontsize=self.base_font_size - 4,
                    ha="center",
                    va="top",
                    color=self.colors["text_secondary"],
                )

            # Average score numeric label
            text_x = avg_score + 3 if avg_score < 85 else avg_score - 3
            text_color = "white" if avg_score >= 85 else self.colors["text"]
            ax.text(
                text_x,
                y_pos,
                f"{avg_score:.0f}",
                fontsize=self.base_font_size + 1,
                fontweight="bold",
                va="center",
                ha="left" if avg_score < 85 else "right",
                color=text_color,
            )

            # Category label
            ax.text(
                -3,
                y_pos,
                label,
                fontsize=self.base_font_size + 1,
                fontweight="bold",
                va="center",
                ha="right",
                color=self.colors["text"],
            )

            # Performance text
            performance = self._get_performance_level(avg_score)
            ax.text(
                103,
                y_pos,
                performance,
                fontsize=self.base_font_size - 1,
                va="center",
                ha="left",
                style="italic",
                color=self._get_score_color(avg_score),
            )

        # Caption (legend)
        ax.text(
            50,
            -0.90,
            "Bars: Average Score  |  Circles: Individual Question Scores",
            fontsize=self.base_font_size - 2,
            ha="center",
            style="italic",
            color="gray",
        )

        ax.set_xlim(-10, 115)
        ax.set_ylim(-2.0, 3.5)
        ax.axis("off")

    def _add_three_shaded_boxes(self, ax_strengths, ax_concerns, ax_breakers, recommendation: Dict[str, Any]) -> None:
        """
        Adds three shaded boxes for strengths, concerns, and deal-breakers.

        :param ax_strengths: Axes for strengths.
        :param ax_concerns: Axes for concerns".
        :param ax_breakers: Axes for dealbreakers.
        :param recommendation: Recommendation dictionary.
        :return: None.
        """
        for ax in (ax_strengths, ax_concerns, ax_breakers):
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        cfgs = [
            (
                ax_strengths,
                "STRENGTHS",
                self.colors["Strong Yes"],
                self.shaded["strengths_fill"],
                recommendation.get("key_strengths", [])[:5],
            ),
            (
                ax_concerns,
                "CONCERNS",
                self.colors["Weak No"],
                self.shaded["concerns_fill"],
                recommendation.get("critical_concerns", [])[:5],
            ),
            (
                ax_breakers,
                "DEAL-BREAKERS",
                self.colors["Strong No"],
                self.shaded["breakers_fill"],
                recommendation.get("deal_breakers", [])[:5] or ["None"],
            ),
        ]

        for ax, title, edge_color, fill_color, items in cfgs:
            card = FancyBboxPatch(
                (0.03, 0.03),
                0.94,
                0.95,
                boxstyle="round,pad=0.020",
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=2.0,
                alpha=0.99,
            )
            ax.add_patch(card)

            # Title placement within the box
            ax.text(0.50, 0.80, title, fontsize=14, fontweight="bold", ha="center", color=edge_color)

            # Wrap and measure bullets for vertical centering; left-aligned text
            wrap_width = 42
            line_spacing = 0.085
            gap_between_bullets = 0.09

            wrapped_bullets: List[List[str]] = []
            for item in items:
                lines = textwrap.fill(item, width=wrap_width).split("\n")
                if lines:
                    wrapped_bullets.append(lines)

            # Total block height to center vertically
            total_h = 0.0
            for lines in wrapped_bullets:
                total_h += len(lines) * line_spacing + gap_between_bullets
            if wrapped_bullets:
                total_h -= gap_between_bullets

            start_y = 0.50 + total_h / 2.0
            bullet_x, text_x = 0.08, 0.12
            y = start_y
            for lines in wrapped_bullets:
                ax.text(bullet_x, y, "•", fontsize=13, color=edge_color, ha="left", va="top")
                ax.text(text_x, y, lines[0], fontsize=11.5, color=self.colors["text"], ha="left", va="top")
                for j, line in enumerate(lines[1:], start=1):
                    ax.text(
                        text_x,
                        y - j * line_spacing,
                        line,
                        fontsize=11.5,
                        color=self.colors["text"],
                        ha="left",
                        va="top",
                    )
                y -= len(lines) * line_spacing + gap_between_bullets
                if y < 0.12:
                    break

    def _add_metrics_bar(self, ax, evaluation_data: Dict[str, Any]) -> None:
        """
        Adds a metrics bar at the bottom with evaluation time, question count, and timestamp.

        :param ax: Axes object.
        :param evaluation_data: Evaluation data dictionary.
        :return: None.
        """
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        metadata = evaluation_data.get("evaluation_metadata", {})
        metrics = []
        duration = metadata.get("evaluation_duration_seconds", 0)
        if duration and duration > 0:
            metrics.append(f"Evaluation Time: {duration:.1f}s")
        q_count = metadata.get("questions_evaluated", 0)
        if q_count and q_count > 0:
            metrics.append(f"Questions Evaluated: {q_count}")

        if metrics:
            ax.text(
                0.50,
                0.50,
                " | ".join(metrics),
                fontsize=10.8,
                ha="center",
                va="center",
                color=self.colors["text_secondary"],
            )

        timestamp = metadata.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%B %d, %Y at %I:%M %p")
        except Exception:
            formatted_time = timestamp

        ax.text(0.98, 0.50, formatted_time, fontsize=10.2, ha="right", va="center", color="#9E9E9E", style="italic")

    def _get_score_color(self, score: float) -> str:
        """
        Gets color based on score thresholds.

        :param score: Score value (0-100).
        :return: Color hex string.
        """
        if score >= self.score_thresholds["excellent"]:
            return self.colors["Strong Yes"]
        elif score >= self.score_thresholds["good"]:
            return self.colors["info"]
        elif score >= self.score_thresholds["fair"]:
            return self.colors["Weak No"]
        else:
            return self.colors["Strong No"]

    def _get_confidence_color(self, confidence: float) -> str:
        """
        Gets color based on confidence thresholds.

        :param confidence: Confidence value (0.0-1.0).
        :return: Color hex string.
        """
        if confidence >= 0.85:
            return self.colors["Strong Yes"]
        elif confidence >= 0.70:
            return self.colors["info"]
        elif confidence >= 0.50:
            return self.colors["Weak No"]
        else:
            return self.colors["Strong No"]

    def _get_performance_level(self, score: float) -> str:
        """
        Gets performance level text based on score.

        :param score: Score value (0-100).
        :return: Performance level string.
        """
        if score >= self.score_thresholds["excellent"]:
            return "Excellent"
        elif score >= self.score_thresholds["good"]:
            return "Good"
        elif score >= self.score_thresholds["fair"]:
            return "Fair"
        else:
            return "Poor"


def generate_visualizations_from_json(
    json_file: Path, candidate_name: Optional[str] = None, output_dir: Optional[Path] = None
) -> Path:
    """
    Generates a visualization from a JSON evaluation file.

    :param json_file: Path to the JSON file.
    :param candidate_name: Optional candidate name.
    :param output_dir: Optional output directory.
    :return: Path to the saved visualization.
    """
    with open(json_file, "r") as f:
        evaluation_data = json.load(f)

    if candidate_name is None:
        filename = json_file.stem
        parts = filename.split("_")
        candidate_name = parts[1].replace("_", " ").title() if len(parts) >= 2 else "Candidate"

    visualizer = EvaluationVisualizer(output_dir)
    return visualizer.create_evaluation_slide(evaluation_data, candidate_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument("json_file", type=str, help="Path to evaluation JSON file")
    parser.add_argument("--name", type=str, help="Candidate name")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    output_path = Path(args.output) if args.output else None

    viz_path = generate_visualizations_from_json(json_path, candidate_name=args.name, output_dir=output_path)
    logger.info(f"Visualization saved to: {viz_path}")
