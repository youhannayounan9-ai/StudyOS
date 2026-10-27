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


class Concept(BaseModel):
    """Course-scoped concept catalog entry."""

    id: str
    course_id: str
    name: str
    normalized_name: str
    description: str | None = None
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
    interval_days: float = Field(default=1.0, ge=1.0, le=180.0)
    ease_factor: float = Field(default=2.5, ge=1.3, le=3.0)
    repetition_count: int = Field(default=0, ge=0)
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
