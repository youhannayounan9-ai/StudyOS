"""Public service facade for StudyOS exam generation, attempt lifecycle, and grading."""

from __future__ import annotations

import logging

from studyos.assessment.generator import ExamGenerator
from studyos.assessment.grader import ExamGrader
from studyos.assessment.models import (
    AttemptStatus,
    Exam,
    ExamAttempt,
    ExamRequest,
    ExamResult,
    PublicExam,
    SubmitAnswersRequest,
)
from studyos.assessment.repository import AssessmentRepository
from studyos.core.config import AppConfig, get_default_config
from studyos.inference.protocol import ChatProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.storage.db import Database

logger = logging.getLogger(__name__)


class AssessmentService:
    """Unified service for practice exam generation, attempt submission, and verified grading."""

    def __init__(
        self,
        db: Database,
        retrieval_service: HybridRetrievalService,
        chat_provider: ChatProvider,
        config: AppConfig | None = None,
        repository: AssessmentRepository | None = None,
        grader: ExamGrader | None = None,
    ) -> None:
        self.db = db
        self.retrieval_service = retrieval_service
        self.chat_provider = chat_provider
        self.config = config or get_default_config()
        self.repository = repository or AssessmentRepository(db)
        self.grader = grader or ExamGrader(chat_provider=chat_provider)
        self.generator = ExamGenerator(
            retrieval_service=retrieval_service,
            chat_provider=chat_provider,
            repository=self.repository,
            config=self.config,
        )

    def generate_exam(self, request: ExamRequest) -> Exam:
        """Generate a new grounded exam and save it to the database."""
        return self.generator.generate_exam(request)

    def get_exam(self, exam_id: str) -> Exam | None:
        """Retrieve full internal exam record with answer keys (for review/instructor workflows)."""
        return self.repository.get_exam(exam_id)

    def get_exam_for_taking(self, exam_id: str) -> PublicExam:
        """Retrieve public exam view for taking.

        Guarantees zero leakage of correct answers, reference answers, or grading rubrics.
        """
        public_exam = self.repository.get_public_exam(exam_id)
        if not public_exam:
            raise ValueError(f"Exam '{exam_id}' not found.")
        return public_exam

    def list_exams(self, course_id: str | None = None) -> list[Exam]:
        """List all available exams, optionally filtered by course."""
        return self.repository.list_exams(course_id=course_id)

    def start_attempt(self, exam_id: str) -> ExamAttempt:
        """Begin an exam attempt."""
        exam = self.repository.get_exam(exam_id)
        if not exam:
            raise ValueError(f"Exam '{exam_id}' not found.")
        return self.repository.create_attempt(exam_id)

    def submit_attempt(self, attempt_id: str, request: SubmitAnswersRequest) -> ExamResult:
        """Submit student answers for an attempt, execute grading, and persist results."""
        attempt = self.repository.get_attempt(attempt_id)
        if not attempt:
            raise ValueError(f"Attempt '{attempt_id}' not found.")

        if attempt.status in (AttemptStatus.COMPLETED, AttemptStatus.PENDING_REVIEW):
            raise ValueError(f"Attempt '{attempt_id}' has already been submitted.")

        exam = self.repository.get_exam(attempt.exam_id)
        if not exam:
            raise ValueError(f"Exam '{attempt.exam_id}' referenced by attempt does not exist.")

        # Grade all questions
        question_results = self.grader.grade_exam(exam, request.answers)

        provisional_score = sum(r.score for r in question_results if r.score is not None)
        max_score = sum(r.max_score for r in question_results)

        # Persist completed results atomically
        self.repository.save_attempt_results(
            attempt_id=attempt_id,
            results=question_results,
            provisional_score=provisional_score,
            max_score=max_score,
        )

        result = self.repository.get_attempt_result(attempt_id)
        if not result:
            raise RuntimeError(f"Failed to retrieve graded attempt result for '{attempt_id}'.")
        return result

    def grade_short_answer_manually(
        self,
        attempt_id: str,
        question_id: str,
        score: float,
        feedback: str | None = None,
    ) -> ExamResult:
        """Complete manual instructor grading for an ungraded short-answer response."""
        return self.repository.grade_question_manually(
            attempt_id=attempt_id,
            question_id=question_id,
            awarded_score=score,
            feedback=feedback,
        )

    def get_attempt_result(self, attempt_id: str) -> ExamResult | None:
        """Retrieve the completed report and explanations for a previous attempt."""
        return self.repository.get_attempt_result(attempt_id)
