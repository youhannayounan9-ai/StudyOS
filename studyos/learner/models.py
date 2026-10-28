"""Pydantic domain models for StudyOS learner modeling, concept catalog, mastery tracking, and revision."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class MasteryCategory(str, Enum):
    """Categorical classification of learner concept mastery."""

    UNASSESSED = "unassessed"
    NEEDS_EVIDENCE = "needs_evidence"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class ReviewStatus(str, Enum):
    """Lifecycle status of a scheduled spaced-repetition review task."""

    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ReviewRating(str, Enum):
    """Student feedback rating on a completed review."""

    AGAIN = "again"  # Complete recall failure (reset interval)
    HARD = "hard"    # Difficult recall (sub-linear progression)
    GOOD = "good"    # Successful recall (normal progression)
    EASY = "easy"    # Effortless recall (accelerated progression)


class LearnerEventType(str, Enum):
    """Types of logged learner activity events."""

    EXAM_QUESTION_GRADED = "exam_question_graded"
    REVIEW_COMPLETED = "review_completed"
    MANUAL_GRADE_FINALIZED = "manual_grade_finalized"


class LearnerProfile(BaseModel):
    """Persistent local learner profile."""

    id: str = Field(description="Unique local learner profile identifier")
    name: str = Field(description="Display name of the learner")
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        clean = (v or "").strip()
        if not clean:
            raise ValueError("Learner name cannot be empty.")
        return clean


class ConceptReviewStatus(str, Enum):
    """Review and acceptance status of a concept in the course catalog."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ConceptProvenance(str, Enum):
    """Source provenance of a concept."""

    MANUAL = "manual"
    ASSESSMENT_DERIVED = "assessment_derived"
    DISCOVERED = "discovered"


class RelationshipType(str, Enum):
    """Type of semantic relationship between two concepts."""

    PREREQUISITE = "prerequisite"  # source should precede target
    RELATED = "related"            # meaningful semantic association without ordering


class RelationshipStatus(str, Enum):
    """Review status of a concept relationship edge."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RelationshipProvenance(str, Enum):
    """Origin of a concept relationship."""

    MANUAL = "manual"
    METADATA_DERIVED = "metadata_derived"
    MODEL_PROPOSED = "model_proposed"
    DISCOVERED = "discovered"


class ConceptRelationship(BaseModel):
    """Course-scoped directed relationship edge between two concepts."""

    id: str
    course_id: str
    source_concept_id: str
    target_concept_id: str
    relationship_type: RelationshipType
    provenance: RelationshipProvenance = Field(default=RelationshipProvenance.MANUAL)
    source_evidence: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    review_status: RelationshipStatus = Field(default=RelationshipStatus.ACCEPTED)
    created_at: str | None = None
    updated_at: str | None = None
    source_concept_name: str | None = None
    target_concept_name: str | None = None


class DiscoveredConcept(BaseModel):
    """A concept candidate discovered directly from ingested source material."""

    name: str
    normalized_name: str
    description: str | None = None
    learning_objectives: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    provenance: ConceptProvenance = Field(default=ConceptProvenance.DISCOVERED)
    review_status: ConceptReviewStatus = Field(default=ConceptReviewStatus.ACCEPTED)
    source_locations: list[dict[str, Any]] = Field(default_factory=list)


class DiscoveryResult(BaseModel):
    """Summary outcome of a direct concept discovery run."""

    run_id: str
    course_id: str
    document_id: str | None = None
    status: str
    concepts_discovered: list[DiscoveredConcept] = Field(default_factory=list)
    total_discovered: int = 0
    new_concepts_created: int = 0
    existing_concepts_updated: int = 0
    chunks_scanned: int = 0
    error_message: str | None = None


class Concept(BaseModel):
    """Course-scoped concept catalog entry."""

    id: str
    course_id: str
    name: str
    normalized_name: str
    description: str | None = None
    review_status: ConceptReviewStatus = Field(default=ConceptReviewStatus.ACCEPTED)
    provenance: ConceptProvenance = Field(default=ConceptProvenance.ASSESSMENT_DERIVED)
    learning_objectives: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        clean = (v or "").strip()
        if not clean:
            raise ValueError("Concept name cannot be empty.")
        return clean


class ConceptMastery(BaseModel):
    """Persistent concept-level mastery and spaced-repetition scheduling state."""

    id: str
    learner_id: str
    concept_id: str
    course_id: str
    concept_name: str | None = None
    mastery_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Deterministic mastery score estimate (0.0 to 1.0)",
    )
    mastery_category: MasteryCategory = Field(default=MasteryCategory.UNASSESSED)
    observation_count: int = Field(
        default=0,
        ge=0,
        description="Count of valid graded assessment attempts supporting this estimate",
    )
    correct_count: float = Field(default=0.0, ge=0.0, description="Normalized success credits (k)")
    incorrect_count: float = Field(default=0.0, ge=0.0, description="Normalized missed credits (n - k)")
    uncertainty: float = Field(
        default=0.2887,
        ge=0.0,
        le=1.0,
        description="Approximate posterior uncertainty (standard deviation) under Beta(1,1)",
    )
    has_sufficient_evidence: bool = Field(
        default=False,
        description="True if observation_count >= 3",
    )
    explanation_reason: str | None = Field(
        default=None,
        description="Transparent human-readable explanation of the current estimate",
    )
    last_assessed_at: str | None = None
    next_review_due: str | None = None
    interval_days: float = Field(default=1.0, ge=1.0, le=365.0)
    ease_factor: float = Field(default=2.5, ge=1.3, le=3.0)
    repetition_count: int = Field(default=0, ge=0)
    stability: float = Field(default=1.0, ge=0.1, description="Memory stability in days (R=0.90 interval)")
    difficulty: float = Field(default=5.0, ge=1.0, le=10.0, description="Inherent concept difficulty (1.0 to 10.0)")
    retrievability: float = Field(default=1.0, ge=0.0, le=1.0, description="Estimated current memory retention")
    scheduler_policy: str = Field(default="local-fsrs-v1", description="Active scheduler policy")
    updated_at: str | None = None


class LearnerEvent(BaseModel):
    """Chronological learner activity event record."""

    id: str
    learner_id: str
    course_id: str
    concept_id: str | None = None
    event_type: str
    reference_id: str | None = None
    question_id: str | None = None
    score_achieved: float | None = None
    score_max: float | None = None
    is_correct: bool | None = None
    details_json: str | None = None
    created_at: str | None = None


class ReviewTask(BaseModel):
    """Scheduled review task for a concept."""

    id: str
    learner_id: str
    concept_id: str
    course_id: str
    concept_name: str | None = None
    due_date: str
    status: ReviewStatus = Field(default=ReviewStatus.SCHEDULED)
    is_due: bool = False
    is_overdue: bool = False
    completed_at: str | None = None
    rating: ReviewRating | None = None
    scheduler_policy: str = Field(default="local-fsrs-v1")
    scheduled_interval: float = Field(default=1.0)
    estimated_retention: float = Field(default=1.0)
    created_at: str | None = None


class ConceptProgressSummary(BaseModel):
    """Summary of progress for a single concept."""

    concept_id: str
    concept_name: str
    course_id: str
    mastery_score: float
    mastery_category: MasteryCategory
    observation_count: int
    correct_count: float
    incorrect_count: float
    uncertainty: float = 0.2887
    has_sufficient_evidence: bool
    last_assessed_at: str | None = None
    next_review_due: str | None = None
    is_review_due: bool = False
    is_review_overdue: bool = False
    stability: float = 1.0
    difficulty: float = 5.0
    retrievability: float = 1.0
    scheduler_policy: str = "local-fsrs-v1"
    prerequisites: list[str] = Field(default_factory=list)
    unmet_prerequisites: list[str] = Field(default_factory=list)
    status_display: str
    explanation_reason: str


class CourseProgressSummary(BaseModel):
    """Summary of learner progress aggregated across a course."""

    course_id: str
    course_name: str
    total_concepts: int = 0
    unassessed_count: int = 0
    needs_evidence_count: int = 0
    weak_count: int = 0
    moderate_count: int = 0
    strong_count: int = 0
    total_graded_attempts: int = 0
    total_correct: float = 0.0
    accuracy_rate: float | None = None
    due_reviews_count: int = 0
    overdue_reviews_count: int = 0


class LearnerProgressSummary(BaseModel):
    """Overall learner progress summary across all courses or scoped to one course."""

    learner_id: str
    course_id: str | None = None
    total_concepts: int = 0
    unassessed_count: int = 0
    needs_evidence_count: int = 0
    weak_count: int = 0
    moderate_count: int = 0
    strong_count: int = 0
    total_graded_attempts: int = 0
    total_correct: float = 0.0
    accuracy_rate: float | None = None
    pending_reviews_count: int = 0
    overdue_reviews_count: int = 0
    concept_summaries: list[ConceptProgressSummary] = Field(default_factory=list)
    recent_activity: list[LearnerEvent] = Field(default_factory=list)


class StudyRecommendation(BaseModel):
    """Explainable recommendation for what the learner should study next."""

    concept_id: str
    concept_name: str
    course_id: str
    action: str
    reason: str
    priority: int = Field(description="1 (highest) to 5 (lowest)")
    mastery_category: MasteryCategory
    observation_count: int
    supporting_evidence: dict[str, Any] = Field(default_factory=dict)
