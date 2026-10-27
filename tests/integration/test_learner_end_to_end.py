"""End-to-end integration test of the complete StudyOS learner model lifecycle:
Ingestion -> Hybrid Retrieval -> Exam Generation -> Attempt Submission -> Grading ->
Concept Mapping -> Learner Model Update -> Mastery Tracking -> Review Scheduling -> Recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    ExamRequest,
    GradingStatus,
    QuestionType,
    SubmitAnswersRequest,
)
from studyos.assessment.service import AssessmentService
from studyos.core.config import AppConfig
from studyos.inference.fake import FakeChatProvider
from studyos.ingestion.service import IngestionService
from studyos.learner.models import MasteryCategory, ReviewRating, ReviewStatus
from studyos.learner.service import LearnerService
from studyos.retrieval.embedding import FakeEmbeddingProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "synthetic"


class TestLearnerEndToEndPipeline:
    """Verifies complete learner model integration with synthetic course content and offline deterministic providers."""

    def test_full_learner_lifecycle(self, tmp_path, fixtures_dir):
        # 1. Setup isolated database and vector store
        data_dir = tmp_path / "studyos_learner_data"
        config = AppConfig(data_dir=data_dir)
        config.ensure_directories()

        db = Database(config.db_path)
        db.migrate()

        embed_provider = FakeEmbeddingProvider(dimension=384)
        vector_store = ChromaVectorStore(chroma_dir=config.chroma_dir, provider=embed_provider)

        # 2. Ingest synthetic lecture
        ingestion = IngestionService(db=db, config=config, vector_store=vector_store)
        course = db.create_course("Algorithms and Data Structures", "CS101")
        res = ingestion.ingest_file(fixtures_dir / "synthetic_lecture.pdf", course_id=course.id)
        assert res.status == "ready"

        retrieval = HybridRetrievalService(db=db, vector_store=vector_store, rrf_k=60)

        # 3. Configure FakeChatProvider with 3 grounded questions
        fake_payload = {
            "questions": [
                {
                    "type": "mcq",
                    "question": "What is the Big-O asymptotic notation used for?",
                    "difficulty": "medium",
                    "options": [
                        {"id": "A", "text": "Hardware voltage"},
                        {"id": "B", "text": "Describing upper bound on execution growth rate"},
                        {"id": "C", "text": "Counting cache lines"},
                        {"id": "D", "text": "Measuring clock cycles"}
                    ],
                    "correct_answer": "B",
                    "explanation": "Big-O characterizes upper asymptotic growth rate [E1].",
                    "evidence_markers": ["E1"]
                },
                {
                    "type": "true_false",
                    "question": "An algorithm with O(n^2) scales faster than O(n) for large n.",
                    "difficulty": "easy",
                    "correct_answer": "False",
                    "explanation": "Quadratic growth rate is strictly slower than linear growth [E1].",
                    "evidence_markers": ["E1"]
                },
                {
                    "type": "short_answer",
                    "question": "Define asymptotic worst-case complexity.",
                    "difficulty": "hard",
                    "correct_answer": "The maximum number of basic operations executed by an algorithm for an input of size n.",
                    "rubric": {
                        "criteria": ["Mentions maximum operations", "Considers input size n"],
                        "key_points": ["maximum", "input size"],
                        "max_score": 1.0
                    },
                    "explanation": "Worst-case analysis guarantees an upper bound on all inputs [E1].",
                    "evidence_markers": ["E1"]
                }
            ]
        }
        fake_chat = FakeChatProvider(default_response=json.dumps(fake_payload))

        learner_service = LearnerService(db=db, chat_provider=fake_chat, config=config)
        assessment_service = AssessmentService(
            db=db,
            retrieval_service=retrieval,
            chat_provider=fake_chat,
            config=config,
            learner_service=learner_service,
        )

        # 4. Generate exam
        exam = assessment_service.generate_exam(
            ExamRequest(
                course_id=course.id,
                question_count=3,
                difficulty=Difficulty.MEDIUM,
                topic="Asymptotic Complexity",
                title="CS101 Complexity Midterm",
            )
        )
        assert len(exam.questions) == 3

        # 5. Take exam (zero answer keys leaked)
        public_view = assessment_service.get_exam_for_taking(exam.id)
        assert hasattr(public_view.questions[0], "correct_answer") is False

        # 6. Start attempt and submit answers:
        # - Q1 (MCQ): Correct ("B")
        # - Q2 (True/False): Incorrect ("True")
        # - Q3 (Short Answer): Student text -> pending review
        q_mcq = exam.questions[0]
        q_tf = exam.questions[1]
        q_short = exam.questions[2]

        attempt = assessment_service.start_attempt(exam.id)
        answers = {
            q_mcq.id: "B",
            q_tf.id: "True",
            q_short.id: "Maximum steps for input size n",
        }

        graded_result = assessment_service.submit_attempt(
            attempt.id,
            SubmitAnswersRequest(answers=answers),
        )
        assert graded_result.status == AttemptStatus.PENDING_REVIEW

        # 7. Verify Learner Model state before manual review:
        # - Concepts automatically created and mapped
        # - 2 objective questions graded (1 correct, 1 incorrect)
        # - Pending short answer is NOT counted as failure!
        summary_before = learner_service.get_progress_summary(course_id=course.id)
        assert summary_before.total_concepts >= 1
        assert summary_before.total_graded_attempts == 2
        assert summary_before.total_correct == 1
        assert summary_before.accuracy_rate == 50.0

        # Check concept mastery: 2 observations -> category NEEDS_EVIDENCE (< 3 observations)
        cpt_summary = summary_before.concept_summaries[0]
        assert cpt_summary.mastery_category == MasteryCategory.NEEDS_EVIDENCE
        assert cpt_summary.observation_count == 2
        assert cpt_summary.correct_count == 1
        assert cpt_summary.incorrect_count == 1
        assert cpt_summary.has_sufficient_evidence is False

        # 8. Complete instructor manual review on the short answer
        reviewed_result = assessment_service.grade_short_answer_manually(
            attempt_id=attempt.id,
            question_id=q_short.id,
            score=1.0,
            feedback="Accurate definition of worst-case complexity.",
        )
        assert reviewed_result.status == AttemptStatus.COMPLETED

        # 9. Verify Learner Model state after manual review:
        # - Total graded attempts now 3 (2 correct, 1 incorrect)
        # - Mastery recomputed from authoritative events: (2+1)/(3+2) = 0.600 -> category MODERATE
        # - Sufficient evidence flag is now True
        summary_after = learner_service.get_progress_summary(course_id=course.id)
        assert summary_after.total_graded_attempts == 3
        assert summary_after.total_correct == 2
        assert summary_after.accuracy_rate == 66.7

        cpt_after = summary_after.concept_summaries[0]
        assert cpt_after.observation_count == 3
        assert cpt_after.correct_count == 2
        assert cpt_after.incorrect_count == 1
        assert cpt_after.mastery_score == 0.600
        assert cpt_after.mastery_category == MasteryCategory.MODERATE
        assert cpt_after.has_sufficient_evidence is True

        # 10. Check Review Queue:
        # Review task exists and is scheduled
        reviews = learner_service.list_pending_reviews(course_id=course.id)
        assert len(reviews) >= 1
        active_task = reviews[0]
        assert active_task.status == ReviewStatus.SCHEDULED

        # 11. Complete a spaced repetition review task with rating GOOD
        completed_task = learner_service.record_review_outcome(active_task.id, ReviewRating.GOOD)
        assert completed_task.status == ReviewStatus.COMPLETED
        assert completed_task.rating == ReviewRating.GOOD

        # Next review task queued with advanced interval
        next_reviews = learner_service.list_pending_reviews(course_id=course.id)
        assert len(next_reviews) >= 1
        assert next_reviews[0].id != active_task.id

        # 12. Check Recommendations:
        # Transparent rule-based recommendation generated with explicit reason
        recs = learner_service.get_recommendations(course_id=course.id)
        assert len(recs) >= 1
        assert recs[0].concept_id == cpt_after.concept_id
        assert recs[0].reason != ""
