"""Unit tests for LearnerService lifecycle, grading integration, summaries, and recommendations."""

from datetime import datetime, timedelta, timezone
import pytest

from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    Exam,
    ExamQuestion,
    ExamResult,
    GradingStatus,
    QuestionResult,
    QuestionType,
)
from studyos.assessment.repository import AssessmentRepository
from studyos.learner.models import MasteryCategory, ReviewRating, ReviewStatus
from studyos.learner.service import LearnerService
from studyos.storage.db import Database


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_learner_service.db"
    db = Database(db_file)
    db.migrate()
    return db


def test_process_exam_attempt_results_and_idempotency(test_db):
    assess_repo = AssessmentRepository(test_db)
    service = LearnerService(db=test_db, assessment_repo=assess_repo)

    crs = test_db.create_course("Algorithms", "CS101")
    exam = assess_repo.save_exam(
        Exam(
            id="ex1",
            course_id=crs.id,
            title="Dijkstra and Graphs",
            difficulty=Difficulty.MEDIUM,
            question_count=2,
            questions=[
                ExamQuestion(
                    id="q1",
                    exam_id="ex1",
                    question_index=1,
                    question_type=QuestionType.MCQ,
                    difficulty=Difficulty.MEDIUM,
                    question_text="What does Dijkstra do?",
                    correct_answer="A",
                    explanation="Shortest path on non-negative weights.",
                ),
                ExamQuestion(
                    id="q2",
                    exam_id="ex1",
                    question_index=2,
                    question_type=QuestionType.TRUE_FALSE,
                    difficulty=Difficulty.EASY,
                    question_text="Does Dijkstra handle negative weights?",
                    correct_answer="False",
                    explanation="Negative weights require Bellman-Ford.",
                ),
            ],
        )
    )

    attempt = assess_repo.create_attempt("ex1")
    q_results = [
        QuestionResult(
            question_id="q1",
            question_index=1,
            question_text="What does Dijkstra do?",
            question_type=QuestionType.MCQ,
            submitted_answer="A",
            correct_answer="A",
            is_correct=True,
            status=GradingStatus.GRADED,
            score=1.0,
            max_score=1.0,
            explanation="Correct.",
            feedback="Well done.",
        ),
        QuestionResult(
            question_id="q2",
            question_index=2,
            question_text="Does Dijkstra handle negative weights?",
            question_type=QuestionType.TRUE_FALSE,
            submitted_answer="True",
            correct_answer="False",
            is_correct=False,
            status=GradingStatus.GRADED,
            score=0.0,
            max_score=1.0,
            explanation="Incorrect.",
            feedback="Dijkstra assumes non-negative edge weights.",
        ),
    ]

    assess_repo.save_attempt_results(
        attempt_id=attempt.id,
        results=q_results,
        provisional_score=1.0,
        max_score=2.0,
    )
    exam_result = assess_repo.get_attempt_result(attempt.id)
    assert exam_result is not None

    # First integration processing
    service.process_exam_attempt_results(attempt.id, exam_result)

    summary = service.get_progress_summary(course_id=crs.id)
    assert summary.total_graded_attempts == 2
    assert summary.total_correct == 1
    assert summary.accuracy_rate == 50.0

    # Repeat call with same attempt: must be strictly idempotent (no double counting!)
    service.process_exam_attempt_results(attempt.id, exam_result)
    summary_repeat = service.get_progress_summary(course_id=crs.id)
    assert summary_repeat.total_graded_attempts == 2
    assert summary_repeat.total_correct == 1


def test_pending_manual_review_does_not_penalize_mastery(test_db):
    assess_repo = AssessmentRepository(test_db)
    service = LearnerService(db=test_db, assessment_repo=assess_repo)

    crs = test_db.create_course("Operating Systems", "CS102")
    exam = assess_repo.save_exam(
        Exam(
            id="ex_os",
            course_id=crs.id,
            title="Virtual Memory",
            difficulty=Difficulty.MEDIUM,
            question_count=2,
            questions=[
                ExamQuestion(
                    id="q_mcq",
                    exam_id="ex_os",
                    question_index=1,
                    question_type=QuestionType.MCQ,
                    difficulty=Difficulty.MEDIUM,
                    question_text="What is a page table?",
                    correct_answer="A",
                    explanation="Translates virtual to physical addresses.",
                ),
                ExamQuestion(
                    id="q_short",
                    exam_id="ex_os",
                    question_index=2,
                    question_type=QuestionType.SHORT_ANSWER,
                    difficulty=Difficulty.HARD,
                    question_text="Explain TLB thrashing.",
                    correct_answer="Reference solution",
                    explanation="Occurs when working set exceeds TLB reach.",
                ),
            ],
        )
    )

    attempt = assess_repo.create_attempt("ex_os")
    q_results = [
        QuestionResult(
            question_id="q_mcq",
            question_index=1,
            question_text="What is a page table?",
            question_type=QuestionType.MCQ,
            submitted_answer="A",
            correct_answer="A",
            is_correct=True,
            status=GradingStatus.GRADED,
            score=1.0,
            max_score=1.0,
            explanation="Correct.",
            feedback="Good.",
        ),
        QuestionResult(
            question_id="q_short",
            question_index=2,
            question_text="Explain TLB thrashing.",
            question_type=QuestionType.SHORT_ANSWER,
            submitted_answer="Student response",
            correct_answer="Reference solution",
            is_correct=None,  # PENDING MANUAL REVIEW
            status=GradingStatus.PENDING_MANUAL_REVIEW,
            score=None,  # Nullable, never false zero
            max_score=1.0,
            explanation="Pending instructor grading.",
            feedback="Awaiting manual review.",
        ),
    ]

    assess_repo.save_attempt_results(
        attempt_id=attempt.id,
        results=q_results,
        provisional_score=1.0,
        max_score=2.0,
    )
    exam_result = assess_repo.get_attempt_result(attempt.id)
    assert exam_result.status == AttemptStatus.PENDING_REVIEW

    service.process_exam_attempt_results(attempt.id, exam_result)

    # Concept mastery must reflect ONLY the 1 graded MCQ, NOT penalizing the pending short answer!
    summary = service.get_progress_summary(course_id=crs.id)
    assert summary.total_graded_attempts == 1
    assert summary.total_correct == 1
    assert summary.accuracy_rate == 100.0

    # Now finalize manual grading
    service.process_manual_review_finalized(
        attempt_id=attempt.id,
        question_id="q_short",
        awarded_score=1.0,
        max_score=1.0,
        is_correct=True,
    )

    summary_after = service.get_progress_summary(course_id=crs.id)
    assert summary_after.total_graded_attempts == 2
    assert summary_after.total_correct == 2
    assert summary_after.accuracy_rate == 100.0


def test_course_isolation_in_progress_summaries(test_db):
    assess_repo = AssessmentRepository(test_db)
    service = LearnerService(db=test_db, assessment_repo=assess_repo)

    crs_a = test_db.create_course("Course Alpha", "CA")
    crs_b = test_db.create_course("Course Beta", "CB")

    cpt_a = service.create_concept(crs_a.id, "Alpha Concept")
    cpt_b = service.create_concept(crs_b.id, "Beta Concept")

    # Record activity in Course Alpha only
    service.repository.record_event(
        learner_id="learner_default",
        course_id=crs_a.id,
        concept_id=cpt_a.id,
        event_type="exam_question_graded",
        reference_id="att_a",
        is_correct=True,
        score_achieved=1.0,
        score_max=1.0,
    )
    service.repository.recompute_mastery_for_concept("learner_default", cpt_a.id)

    sum_a = service.get_progress_summary(course_id=crs_a.id)
    assert sum_a.total_concepts == 1
    assert sum_a.total_graded_attempts == 1
    assert sum_a.total_correct == 1

    sum_b = service.get_progress_summary(course_id=crs_b.id)
    assert sum_b.total_concepts == 1
    assert sum_b.total_graded_attempts == 0
    assert sum_b.total_correct == 0
    assert sum_b.unassessed_count == 1


def test_recommendations_ranking_and_diagnostics(test_db):
    assess_repo = AssessmentRepository(test_db)
    service = LearnerService(db=test_db, assessment_repo=assess_repo)

    crs = test_db.create_course("Algorithms", "CS101")
    c_unassessed = service.create_concept(crs.id, "Dynamic Programming")
    c_weak = service.create_concept(crs.id, "Red-Black Trees")
    c_overdue = service.create_concept(crs.id, "Dijkstra Algorithm")

    # c_weak: 3 incorrect attempts -> WEAK category
    for i in range(3):
        service.repository.record_event(
            learner_id="learner_default",
            course_id=crs.id,
            concept_id=c_weak.id,
            event_type="exam_question_graded",
            reference_id=f"att_w_{i}",
            is_correct=False,
            score_achieved=0.0,
            score_max=1.0,
        )
    service.repository.recompute_mastery_for_concept("learner_default", c_weak.id)

    # c_overdue: create a review task due 2 days ago
    past_due = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    service.repository.create_or_update_review_task(
        learner_id="learner_default",
        concept_id=c_overdue.id,
        course_id=crs.id,
        due_date=past_due,
    )
    m_overdue = service.repository.recompute_mastery_for_concept("learner_default", c_overdue.id)
    m_overdue.next_review_due = past_due
    service.repository.upsert_concept_mastery(m_overdue)

    recs = service.get_recommendations(course_id=crs.id)
    assert len(recs) >= 3

    # Overdue task is Priority 1
    assert recs[0].concept_id == c_overdue.id
    assert recs[0].priority == 1
    assert recs[0].action == "Complete Overdue Review"

    # Weak concept is Priority 2
    assert recs[1].concept_id == c_weak.id
    assert recs[1].priority == 2
    assert recs[1].action == "Targeted Practice on Weak Concept"

    # Unassessed concept is Priority 5
    unassessed_rec = next(r for r in recs if r.concept_id == c_unassessed.id)
    assert unassessed_rec.priority == 5
    assert unassessed_rec.action == "Initial Diagnostic Quiz"


def test_review_recording_lifecycle(test_db):
    service = LearnerService(db=test_db)
    crs = test_db.create_course("Algorithms", "CS101")
    cpt = service.create_concept(crs.id, "Sorting")

    due_str = datetime.now(timezone.utc).isoformat()
    task = service.repository.create_or_update_review_task(
        learner_id="learner_default",
        concept_id=cpt.id,
        course_id=crs.id,
        due_date=due_str,
    )
    assert task.status == ReviewStatus.SCHEDULED

    # Complete review with rating GOOD
    completed = service.record_review_outcome(task.id, ReviewRating.GOOD)
    assert completed.status == ReviewStatus.COMPLETED
    assert completed.rating == ReviewRating.GOOD

    # A new review task should be queued
    pending = service.list_pending_reviews(course_id=crs.id)
    assert len(pending) == 1
    assert pending[0].id != task.id
    assert pending[0].concept_id == cpt.id
