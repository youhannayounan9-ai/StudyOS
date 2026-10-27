"""Unit tests for objective and short-answer exam grading, scoring policies, and attempt lifecycle."""

from __future__ import annotations

import json
import pytest

from studyos.assessment.grader import ExamGrader
from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    Exam,
    ExamQuestion,
    GradingStatus,
    MCQOption,
    QuestionType,
    ShortAnswerRubric,
    SubmitAnswersRequest,
)
from studyos.assessment.repository import AssessmentRepository
from studyos.assessment.service import AssessmentService
from studyos.inference.fake import FakeChatProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.storage.db import Database


@pytest.fixture
def test_setup(tmp_path):
    db = Database(tmp_path / "grading_test.db")
    db.migrate()
    course = db.create_course("Operating Systems", "CS301")

    q1 = ExamQuestion(
        id="q_mcq",
        exam_id="ex_grade_01",
        question_index=1,
        question_type=QuestionType.MCQ,
        difficulty=Difficulty.EASY,
        question_text="Which system call creates a new process in POSIX?",
        options=[
            MCQOption(id="A", text="exec"),
            MCQOption(id="B", text="fork"),
            MCQOption(id="C", text="clone"),
            MCQOption(id="D", text="spawn")
        ],
        correct_answer="B",
        explanation="fork() creates a duplicate child process.",
        points=2.0,
    )
    q2 = ExamQuestion(
        id="q_tf",
        exam_id="ex_grade_01",
        question_index=2,
        question_type=QuestionType.TRUE_FALSE,
        difficulty=Difficulty.MEDIUM,
        question_text="Virtual memory paging completely eliminates external fragmentation.",
        correct_answer="True",
        explanation="Fixed-size pages eliminate external fragmentation.",
        points=1.0,
    )
    q3 = ExamQuestion(
        id="q_sa",
        exam_id="ex_grade_01",
        question_index=3,
        question_type=QuestionType.SHORT_ANSWER,
        difficulty=Difficulty.HARD,
        question_text="Define a race condition.",
        correct_answer="A situation where program behavior depends on the non-deterministic interleaving or timing of threads.",
        explanation="Concurrent unsynchronized access to shared state.",
        rubric=ShortAnswerRubric(
            criteria=["Mentions concurrent execution", "Mentions timing/interleaving dependence"],
            key_points=["concurrency", "timing"],
            max_score=3.0,
        ),
        points=3.0,
    )

    exam = Exam(
        id="ex_grade_01",
        course_id=course.id,
        title="OS Concurrency & Processes",
        difficulty=Difficulty.MEDIUM,
        question_count=3,
        questions=[q1, q2, q3],
    )

    repo = AssessmentRepository(db)
    repo.save_exam(exam)

    return db, repo, exam


def test_deterministic_mcq_and_true_false_grading(test_setup):
    """MCQ and True/False questions are graded deterministically against stored keys."""
    _, _, exam = test_setup
    grader = ExamGrader()

    # 1. Correct submission
    answers = {
        "q_mcq": "B",
        "q_tf": "true",  # case-insensitive normalization
        "q_sa": "A situation where timing determines outcome.",
    }
    results = grader.grade_exam(exam, answers)

    # Question 1: MCQ correct
    assert results[0].is_correct is True
    assert results[0].score == 2.0
    assert results[0].max_score == 2.0
    assert "Correct" in results[0].feedback

    # Question 2: True/False correct
    assert results[1].is_correct is True
    assert results[1].score == 1.0
    assert results[1].max_score == 1.0
    assert "Correct" in results[1].feedback


def test_mcq_full_text_fallback_match(test_setup):
    """Submitting the full text of the correct choice awards full credit."""
    _, _, exam = test_setup
    grader = ExamGrader()

    answers = {
        "q_mcq": "fork",  # Submitted text instead of choice ID 'B'
        "q_tf": "False",
    }
    results = grader.grade_exam(exam, answers)

    assert results[0].is_correct is True
    assert results[0].score == 2.0


def test_incorrect_mcq_and_true_false(test_setup):
    """Incorrect choices award 0 points and provide explanatory feedback."""
    _, _, exam = test_setup
    grader = ExamGrader()

    answers = {
        "q_mcq": "A",  # Wrong! Correct was B
        "q_tf": "False",  # Wrong! Correct was True
    }
    results = grader.grade_exam(exam, answers)

    assert results[0].is_correct is False
    assert results[0].score == 0.0
    assert "Correct answer was B" in results[0].feedback

    assert results[1].is_correct is False
    assert results[1].score == 0.0
    assert "Statement is True" in results[1].feedback


def test_skipped_question_policy(test_setup):
    """Questions omitted from the submitted answers or submitted as empty are scored 0 as skipped."""
    _, _, exam = test_setup
    grader = ExamGrader()

    # q_mcq is submitted, q_tf is omitted, q_sa is empty string
    answers = {
        "q_mcq": "B",
        "q_sa": "   ",
    }
    results = grader.grade_exam(exam, answers)

    # Omitted question
    assert results[1].submitted_answer is None
    assert results[1].is_correct is False
    assert results[1].score == 0.0
    assert results[1].feedback == "Question was skipped."

    # Whitespace question
    assert results[2].submitted_answer is None
    assert results[2].is_correct is False
    assert results[2].score == 0.0
    assert results[2].feedback == "Question was skipped."


def test_reject_submitted_question_id_outside_exam(test_setup):
    """Submitting an answer for a question ID that does not belong to the exam raises ValueError."""
    _, _, exam = test_setup
    grader = ExamGrader()

    answers = {
        "q_mcq": "B",
        "q_unrelated_id": "Option A",
    }
    with pytest.raises(ValueError, match="does not belong to exam"):
        grader.grade_exam(exam, answers)


def test_short_answer_conservative_manual_review_fallback(test_setup):
    """Without an active local chat provider, short-answer questions fall back to manual review with NULL score."""
    _, _, exam = test_setup
    grader = ExamGrader(chat_provider=None)

    answers = {"q_sa": "Timing dependent execution."}
    results = grader.grade_exam(exam, answers)

    sa_res = results[2]
    assert sa_res.is_correct is None  # Pending review
    assert sa_res.status == GradingStatus.PENDING_MANUAL_REVIEW
    assert sa_res.score is None  # Must NOT be 0.0! Points are unawarded/pending
    assert sa_res.max_score == 3.0
    assert "Pending manual review" in sa_res.feedback


def test_mcq_incorrect_gets_zero_points_and_graded_status(test_setup):
    """An MCQ graded incorrect receives zero awarded points (0.0) and has status 'graded'."""
    _, _, exam = test_setup
    grader = ExamGrader()

    answers = {"q_mcq": "A"}  # Wrong choice (correct is B)
    results = grader.grade_exam(exam, answers)

    mcq_res = results[0]
    assert mcq_res.is_correct is False
    assert mcq_res.status == GradingStatus.GRADED
    assert mcq_res.score == 0.0  # Genuine zero points earned
    assert mcq_res.score is not None


def test_skipped_objective_question_gets_zero_and_graded_status(test_setup):
    """An unanswered objective question follows the documented skipped policy: score 0.0, status graded."""
    _, _, exam = test_setup
    grader = ExamGrader()

    answers = {}  # All omitted
    results = grader.grade_exam(exam, answers)

    mcq_res = results[0]
    assert mcq_res.is_correct is False
    assert mcq_res.status == GradingStatus.GRADED
    assert mcq_res.score == 0.0
    assert mcq_res.feedback == "Question was skipped."

    tf_res = results[1]
    assert tf_res.is_correct is False
    assert tf_res.status == GradingStatus.GRADED
    assert tf_res.score == 0.0
    assert tf_res.feedback == "Question was skipped."


def test_exam_with_pending_short_answer_has_null_final_score_and_provisional_subtotal(test_setup):
    """An exam with a pending short answer exposes final score as unavailable (None) with a provisional subtotal."""
    db, repo, exam = test_setup
    service = AssessmentService(db=db, retrieval_service=None, chat_provider=None, repository=repo)

    attempt = service.start_attempt(exam.id)
    # Student submits MCQ correct (2.0), TF correct (1.0), and short-answer text (pending review)
    sub_req = SubmitAnswersRequest(
        answers={
            "q_mcq": "B",
            "q_tf": "True",
            "q_sa": "Race conditions occur when threads interleave.",
        }
    )
    result = service.submit_attempt(attempt.id, sub_req)

    # 1. Attempt is pending review, NOT completed
    assert result.status == AttemptStatus.PENDING_REVIEW
    assert result.needs_manual_review is True

    # 2. Final score and percentage must be None (not falsely presented as completed)
    assert result.score is None
    assert result.percentage is None

    # 3. Provisional scores are accurately separated
    assert result.provisional_score == 3.0  # 2.0 + 1.0 from graded questions
    assert result.max_score == 6.0
    assert result.provisional_percentage == 50.0

    # 4. In individual question results:
    assert result.questions[0].score == 2.0
    assert result.questions[0].status == GradingStatus.GRADED

    assert result.questions[1].score == 1.0
    assert result.questions[1].status == GradingStatus.GRADED

    assert result.questions[2].score is None  # Undetermined!
    assert result.questions[2].status == GradingStatus.PENDING_MANUAL_REVIEW
    assert result.questions[2].is_correct is None


def test_manual_review_completion_path_updates_score_without_altering_exam_key(test_setup):
    """Manual review completion path records verified score and finalizes attempt without touching exam key."""
    db, repo, exam = test_setup
    service = AssessmentService(db=db, retrieval_service=None, chat_provider=None, repository=repo)

    attempt = service.start_attempt(exam.id)
    service.submit_attempt(
        attempt.id,
        SubmitAnswersRequest(
            answers={
                "q_mcq": "B",
                "q_tf": "True",
                "q_sa": "Race condition definition.",
            }
        ),
    )

    # 1. Attempting to grade out of bounds raises ValueError
    with pytest.raises(ValueError, match="out of bounds"):
        service.grade_short_answer_manually(attempt.id, "q_sa", score=5.0)  # Max is 3.0

    # 2. Complete review with valid awarded score
    updated = service.grade_short_answer_manually(
        attempt.id, "q_sa", score=2.5, feedback="Good explanation of thread concurrency."
    )

    # 3. Final score is now computed and attempt is completed
    assert updated.status == AttemptStatus.COMPLETED
    assert updated.needs_manual_review is False
    assert updated.score == 5.5  # 2.0 + 1.0 + 2.5
    assert updated.percentage == round(5.5 / 6.0 * 100.0, 2)
    assert updated.provisional_score == 5.5
    assert updated.questions[2].score == 2.5
    assert updated.questions[2].status == GradingStatus.GRADED
    assert updated.questions[2].feedback == "Good explanation of thread concurrency."

    # 4. Crucial safety check: stored exam question and rubric are completely unchanged!
    unaltered_exam = repo.get_exam(exam.id)
    assert unaltered_exam.questions[2].correct_answer == exam.questions[2].correct_answer
    assert unaltered_exam.questions[2].rubric.max_score == 3.0


def test_serialization_distinguishes_zero_and_ungraded(test_setup):
    """Serialization to dict and JSON preserves distinction between 0.0 and None/null score."""
    db, repo, exam = test_setup
    service = AssessmentService(db=db, retrieval_service=None, chat_provider=None, repository=repo)

    attempt = service.start_attempt(exam.id)
    # MCQ wrong (0.0), TF correct (1.0), SA pending (None)
    result = service.submit_attempt(
        attempt.id,
        SubmitAnswersRequest(
            answers={
                "q_mcq": "A",  # Wrong -> 0.0
                "q_tf": "True",  # Correct -> 1.0
                "q_sa": "Candidate answer text.",  # Pending -> None
            }
        ),
    )

    dumped = result.model_dump()
    assert dumped["score"] is None
    assert dumped["status"] == "pending_review"
    assert dumped["provisional_score"] == 1.0

    # Question 0: MCQ incorrect has explicit float 0.0
    assert dumped["questions"][0]["score"] == 0.0
    assert dumped["questions"][0]["status"] == "graded"

    # Question 2: Short answer pending has None (null)
    assert dumped["questions"][2]["score"] is None
    assert dumped["questions"][2]["status"] == "pending_manual_review"

    # JSON serialization check
    json_str = result.model_dump_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["score"] is None
    assert parsed_json["questions"][0]["score"] == 0.0
    assert parsed_json["questions"][2]["score"] is None
    assert parsed_json["questions"][2]["status"] == "pending_manual_review"


def test_full_attempt_submission_lifecycle(test_setup):
    """Start attempt -> submit answers -> verify score totals and attempt completion."""
    db, repo, exam = test_setup

    service = AssessmentService(
        db=db,
        retrieval_service=None,  # Not needed for submission
        chat_provider=None,
        repository=repo,
    )

    # 1. Start attempt
    attempt = service.start_attempt(exam.id)
    assert attempt.status == AttemptStatus.IN_PROGRESS

    # 2. Submit answers: q_mcq correct (2.0), q_tf correct (1.0), q_sa skipped (0.0)
    sub_req = SubmitAnswersRequest(
        answers={
            "q_mcq": "B",
            "q_tf": "True",
        }
    )
    result = service.submit_attempt(attempt.id, sub_req)

    assert result.attempt_id == attempt.id
    assert result.exam_id == exam.id
    assert result.score == 3.0
    assert result.max_score == 6.0  # 2.0 + 1.0 + 3.0
    assert result.percentage == 50.0
    assert len(result.questions) == 3

    # 3. Verify attempt status is updated in database
    saved_att = repo.get_attempt(attempt.id)
    assert saved_att.status == AttemptStatus.COMPLETED
    assert saved_att.score == 3.0

    # 4. Attempt re-submission must be rejected
    with pytest.raises(ValueError, match="already been submitted"):
        service.submit_attempt(attempt.id, sub_req)
