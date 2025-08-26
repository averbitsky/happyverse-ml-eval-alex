"""
Pydantic models for evaluation results.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationLevel(Enum):
    """
    Enum for recommendation levels.
    """

    STRONG_YES = "Strong Yes"
    WEAK_YES = "Weak Yes"
    WEAK_NO = "Weak No"
    STRONG_NO = "Strong No"


class HiringRisk(Enum):
    """
    Enum for hiring risk levels.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TechnicalFeasibility(BaseModel):
    """
    Technical feasibility assessment.
    """

    assessment: str
    claims_needing_verification: List[str] = Field(default_factory=list)
    impossible_claims: List[str] = Field(default_factory=list)
    questionable_claims: List[str] = Field(default_factory=list)


class Verifiability(BaseModel):
    """
    Verifiability assessment.
    """

    verifiable_elements: List[str] = Field(default_factory=list)
    missing_specifics: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)


class AuthenticityIndicators(BaseModel):
    """
    Authenticity assessment.
    """

    positive_signs: List[str] = Field(default_factory=list)
    negative_signs: List[str] = Field(default_factory=list)


class PlausibilityResult(BaseModel):
    """
    Plausibility evaluation result.
    """

    plausibility_score: int = Field(ge=0, le=100)
    technical_feasibility: TechnicalFeasibility
    verifiability: Verifiability
    authenticity_indicators: AuthenticityIndicators
    overall_assessment: str


class AccuracyAssessment(BaseModel):
    """
    Technical accuracy assessment.
    """

    correct_concepts: List[str] = Field(default_factory=list)
    uncertain_concepts: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    terminology_usage: str = ""


class DepthAnalysis(BaseModel):
    """
    Depth of knowledge analysis.
    """

    level: str = Field(pattern="^(shallow|moderate|deep)$")
    evidence_of_experience: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    explanation: str


class ProblemSolving(BaseModel):
    """
    Problem solving assessment.
    """

    approach_quality: str
    best_practices: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)


class RoleFit(BaseModel):
    """
    Role fit assessment.
    """

    alignment_score: int = Field(ge=0, le=100)
    strengths_for_role: List[str] = Field(default_factory=list)
    gaps_for_role: List[str] = Field(default_factory=list)
    explanation: str


class TechnicalResult(BaseModel):
    """
    Technical proficiency evaluation result.
    """

    technical_score: int = Field(ge=0, le=100)
    accuracy_assessment: AccuracyAssessment
    depth_analysis: DepthAnalysis
    problem_solving: ProblemSolving
    role_fit: RoleFit
    overall_assessment: str


class ClarityAssessment(BaseModel):
    """
    Clarity assessment.
    """

    score: int = Field(ge=0, le=100)
    organization: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class RelevanceAssessment(BaseModel):
    """
    Relevance assessment.
    """

    answers_question: bool
    directness_score: int = Field(ge=0, le=100)
    unnecessary_content: List[str] = Field(default_factory=list)
    missing_elements: List[str] = Field(default_factory=list)


class PersuasivenessAssessment(BaseModel):
    """
    Persuasiveness assessment.
    """

    score: int = Field(ge=0, le=100)
    compelling_elements: List[str] = Field(default_factory=list)
    weak_areas: List[str] = Field(default_factory=list)
    storytelling_quality: str


class ProfessionalismAssessment(BaseModel):
    """
    Professionalism assessment.
    """

    appropriate_tone: bool
    technical_communication: str
    red_flags: List[str] = Field(default_factory=list)


class CommunicationResult(BaseModel):
    """
    Communication evaluation result.
    """

    communication_score: int = Field(ge=0, le=100)
    clarity: ClarityAssessment
    relevance: RelevanceAssessment
    persuasiveness: PersuasivenessAssessment
    professionalism: ProfessionalismAssessment
    overall_assessment: str


class SpecificRecommendations(BaseModel):
    """
    Specific hiring recommendations.
    """

    if_hire: str
    if_reject: str


class SynthesisResult(BaseModel):
    """
    Final synthesis and recommendation.
    """

    recommendation_level: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_strengths: List[str] = Field(default_factory=list)
    critical_concerns: List[str] = Field(default_factory=list)
    deal_breakers: List[str] = Field(default_factory=list)
    hiring_risk: str
    comparison_to_typical: str
    specific_recommendations: SpecificRecommendations
    detailed_rationale: str


class VerificationEntities(BaseModel):
    """
    Entities extracted for verification.
    """

    companies: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    implementations: List[str] = Field(default_factory=list)


class QuestionEvaluation(BaseModel):
    """
    Complete evaluation for a single question.
    """

    question: str
    response: str
    plausibility: Optional[PlausibilityResult] = None
    technical: Optional[TechnicalResult] = None
    communication: Optional[CommunicationResult] = None
    verification: Dict[str, Any] = Field(default_factory=dict)


class AggregateScore(BaseModel):
    """
    Aggregate score statistics.
    """

    mean: float
    min: float
    max: float
    all_scores: List[float]


class AggregateScores(BaseModel):
    """
    All aggregate scores.
    """

    plausibility: AggregateScore
    technical: AggregateScore
    communication: AggregateScore


class EvaluationMetadata(BaseModel):
    """
    Metadata about the evaluation.
    """

    timestamp: str
    model_used: str
    questions_evaluated: int
    known_technologies_count: int
    evaluation_duration_seconds: float = 0.0
    cache_hit_rate: float = 0.0


class FinalEvaluation(BaseModel):
    """
    Complete evaluation result.
    """

    individual_evaluations: List[QuestionEvaluation]
    aggregate_scores: AggregateScores
    recommendation: SynthesisResult
    evaluation_metadata: EvaluationMetadata


class BatchEvaluationResult(BaseModel):
    """
    Result of batch evaluation.
    """

    evaluations: List[FinalEvaluation]
    batch_metadata: Dict[str, Any]
