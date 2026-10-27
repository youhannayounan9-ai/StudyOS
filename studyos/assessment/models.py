"""Typed domain models for StudyOS exam generation, answer keys, attempts, and grading."""

from __future__ import annotations

from enum import Enum
import json
from typing import Any
from pydantic import BaseModel, Field, field_validator

from studyos.tutoring.models import ValidatedCitation


class QuestionType(str, Enum):
    """Supported assessment question types."""

    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


class Difficulty(str, Enum):
    """Question and exam difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"


class AttemptStatus(str, Enum):
    """Lifecycle status of an exam attempt."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PENDING_REVIEW = "pending_review"


class GradingStatus(str, Enum):
    """Grading status of an individual question."""

    GRADED = "graded"
    PENDING_MANUAL_REVIEW = "pending_manual_review"


class MCQOption(BaseModel):
    """An option choice for a multiple-choice question."""

    id: str = Field(description="Unique choice identifier, e.g. 'A', 'B', 'C', 'D'")
    text: str = Field(description="Display text of the option choice")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        clean = (v or "").strip().upper()
        if not clean:
            raise ValueError("Option ID cannot be empty.")
        return clean

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        clean = (v or "").strip()
        if not clean:
            raise ValueError("Option text cannot be empty.")
        return clean


class ShortAnswerRubric(BaseModel):
    """Scoring rubric for evaluating free-text short-answer questions."""

    criteria: list[str] = Field(default_factory=list, description="Specific assessment criteria")
    key_points: list[str] = Field(default_factory=list, description="Key concepts or keywords expected")
    max_score: float = Field(default=1.0, ge=0.5, le=10.0, description="Maximum points for question")


class QuestionSourceRef(BaseModel):
    """Authoritative source chunk reference linked to an assessment question."""

    chunk_id: str | None = Field(default=None, description="Authoritative SQLite chunk ID (live)")
    original_chunk_id: str | None = Field(default=None, description="Original chunk ID at generation time")
    evidence_marker: str = Field(description="Temporary prompt marker, e.g. 'E1'")
    filename: str | None = None
    location_type: str | None = None
    location_index: int | None = None
    location_title: str | None = None
    excerpt: str | None = None
    is_stale: bool = Field(default=False, description="True if source chunk was deleted or modified")


class ExamRequest(BaseModel):
    """Request contract for generating a grounded practice exam."""

    course_id: str = Field(description="Course ID to generate exam from")
    document_ids: list[str] | None = Field(default=None, description="Optional document filter")
    question_count: int = Field(default=5, ge=1, le=20, description="Total questions to generate (1-20)")
    question_types: list[QuestionType] | None = Field(
        default=None,
        description="Allowed question types (default: all supported types)",
    )
    difficulty: Difficulty = Field(default=Difficulty.MEDIUM, description="Overall target difficulty")
    topic: str | None = Field(default=None, max_length=500, description="Optional topic or chapter focus")
    title: str | None = Field(default=None, max_length=200, description="Optional custom exam title")

    @field_validator("course_id")
    @classmethod
    def validate_course(cls, v: str) -> str:
        clean = (v or "").strip()
        if not clean:
            raise ValueError("course_id cannot be empty or whitespace.")
        return clean

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str | None) -> str | None:
        if v is None:
            return None
        clean = v.strip()
        return clean if clean else None


# ---------------------------------------------------------------------------
# Internal Exam Models (Contains Answer Keys and Private Explanations)
# ---------------------------------------------------------------------------

class ExamQuestion(BaseModel):
    """Internal question model storing question text, options, answer key, and rubric."""

    id: str
    exam_id: str
    question_index: int
    question_type: QuestionType
    difficulty: Difficulty
    question_text: str
    options: list[MCQOption] | None = None
    correct_answer: str = Field(description="Authoritative correct answer (PRIVATE)")
    explanation: str = Field(description="Explanation of correct answer (PRIVATE until graded)")
    rubric: ShortAnswerRubric | None = None
    sources: list[QuestionSourceRef] = Field(default_factory=list)
    points: float = Field(default=1.0, ge=0.5)

    def options_to_json(self) -> str | None:
        if not self.options:
            return None
        return json.dumps([opt.model_dump() for opt in self.options])

    def rubric_to_json(self) -> str | None:
        if not self.rubric:
            return None
        return self.rubric.model_dump_json()


class Exam(BaseModel):
    """Internal exam record storing complete exam structure and answer keys."""

    id: str
    course_id: str
    title: str
    topic: str | None = None
    difficulty: Difficulty
    question_count: int
    questions: list[ExamQuestion] = Field(default_factory=list)
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Public Exam Presentation Models (ZERO ANSWER KEYS OR RUBRICS)
# ---------------------------------------------------------------------------

class PublicQuestion(BaseModel):
    """Public question presentation model safe for taking exams.

    Guarantees zero leakage of correct answers, reference solutions, or grading rubrics.
    """

    id: str
    question_index: int
    question_type: QuestionType
    difficulty: Difficulty
    question_text: str
    options: list[MCQOption] | None = None
    points: float = 1.0


class PublicExam(BaseModel):
    """Public exam model presented to students during an exam session.

    Contains only question text and choices. Excludes all answer keys.
    """

    id: str
    course_id: str
    title: str
    topic: str | None = None
    difficulty: Difficulty
    question_count: int
    questions: list[PublicQuestion] = Field(default_factory=list)
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Attempt & Grading Models
# ---------------------------------------------------------------------------

class ExamAttempt(BaseModel):
    """Record of a student's attempt on an exam."""

    id: str
    exam_id: str
    status: AttemptStatus
    score: float | None = None
    provisional_score: float | None = None
    max_score: float | None = None
    percentage: float | None = None
    started_at: str | None = None
    completed_at: str | None = None


class SubmitAnswersRequest(BaseModel):
    """Submitted student answers mapped by question ID."""

    answers: dict[str, str] = Field(
        description="Mapping of question_id -> student's answer choice or text response"
    )


class QuestionResult(BaseModel):
    """Grading result for an individual question in an attempt."""

    question_id: str
    question_index: int
    question_text: str
    question_type: QuestionType
    submitted_answer: str | None
    correct_answer: str
    is_correct: bool | None = Field(
        default=None,
        description="True if correct, False if incorrect/skipped, None if pending manual review",
    )
    status: GradingStatus = Field(
        default=GradingStatus.GRADED,
        description="Grading status: 'graded' or 'pending_manual_review'",
    )
    score: float | None = Field(
        default=None,
        description="Earned points if graded. None if pending manual review (never false zero)",
    )
    max_score: float
    explanation: str
    feedback: str
    citations: list[QuestionSourceRef] = Field(default_factory=list)


class ExamResult(BaseModel):
    """Complete grading report returned after submitting an exam attempt."""

    attempt_id: str
    exam_id: str
    course_id: str
    title: str
    status: AttemptStatus = Field(
        default=AttemptStatus.COMPLETED,
        description="'completed' if all questions graded, 'pending_review' if awaiting manual review",
    )
    score: float | None = Field(
        default=None,
        description="Final earned score. None if attempt is pending manual review",
    )
    provisional_score: float = Field(
        default=0.0,
        description="Provisional subtotal of points earned on completed/graded questions",
    )
    max_score: float = Field(
        description="Total possible points across all questions on the exam"
    )
    percentage: float | None = Field(
        default=None,
        description="Final percentage score. None if attempt is pending manual review",
    )
    provisional_percentage: float | None = Field(
        default=None,
        description="Provisional percentage based on current provisional_score / max_score",
    )
    completed_at: str | None = None
    questions: list[QuestionResult] = Field(default_factory=list)
    needs_manual_review: bool = Field(
        default=False,
        description="True if one or more short-answer responses require instructor review",
    )
