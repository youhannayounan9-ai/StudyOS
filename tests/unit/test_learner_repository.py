"""Unit tests for LearnerRepository persistence, constraints, and transactions."""

import pytest
from studyos.assessment.models import Difficulty, Exam, ExamQuestion, QuestionType
from studyos.assessment.repository import AssessmentRepository
from studyos.ingestion.models import DocumentChunk
from studyos.learner.models import MasteryCategory, ReviewRating, ReviewStatus
from studyos.learner.repository import LearnerRepository
from studyos.storage.db import Database


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_learner.db"
    db = Database(db_file)
    db.migrate()
    return db


def test_profile_idempotency(test_db):
    repo = LearnerRepository(test_db)
    p1 = repo.get_or_create_profile(learner_id="u1", name="Learner 1")
    assert p1.id == "u1"
    assert p1.name == "Learner 1"

    # Second call returns existing profile
    p2 = repo.get_or_create_profile(learner_id="u1", name="Different Name")
    assert p2.id == "u1"
    assert p2.name == "Learner 1"


def test_concept_deduplication_within_course(test_db):
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Algorithms", "CS101")

    c1 = repo.get_or_create_concept(crs.id, "Dynamic Programming")
    c2 = repo.get_or_create_concept(crs.id, "  dynamic   programming  ")
    assert c1.id == c2.id
    assert c1.normalized_name == "dynamic programming"


def test_concept_isolation_across_different_courses(test_db):
    repo = LearnerRepository(test_db)
    crs1 = test_db.create_course("Algorithms", "CS101")
    crs2 = test_db.create_course("Advanced Algorithms", "CS201")

    # Identical concept names in different courses must NOT be merged
    c1 = repo.get_or_create_concept(crs1.id, "Recursion")
    c2 = repo.get_or_create_concept(crs2.id, "Recursion")

    assert c1.id != c2.id
    assert c1.course_id == crs1.id
    assert c2.course_id == crs2.id


def test_question_concept_cross_course_validation(test_db):
    repo = LearnerRepository(test_db)
    assess_repo = AssessmentRepository(test_db)

    crs1 = test_db.create_course("Course 1", "C1")
    crs2 = test_db.create_course("Course 2", "C2")

    exam1 = assess_repo.save_exam(
        Exam(
            id="ex1",
            course_id=crs1.id,
            title="Exam 1",
            difficulty=Difficulty.MEDIUM,
            question_count=1,
            questions=[
                ExamQuestion(
                    id="q1",
                    exam_id="ex1",
                    question_index=1,
                    question_type=QuestionType.MCQ,
                    difficulty=Difficulty.MEDIUM,
                    question_text="Q1 text",
                    correct_answer="A",
                    explanation="Exp",
                )
            ],
        )
    )

    cpt_crs1 = repo.get_or_create_concept(crs1.id, "Concept C1")
    cpt_crs2 = repo.get_or_create_concept(crs2.id, "Concept C2")

    # Mapping question in C1 to concept in C1 succeeds
    repo.map_question_to_concept("q1", cpt_crs1.id)
    mapped = repo.get_concepts_for_question("q1")
    assert len(mapped) == 1
    assert mapped[0].id == cpt_crs1.id

    # Mapping question in C1 to concept in C2 must be rejected!
    with pytest.raises(ValueError, match="Cross-course mapping rejected"):
        repo.map_question_to_concept("q1", cpt_crs2.id)


def test_concept_chunk_cross_course_validation_and_stale_detection(test_db):
    repo = LearnerRepository(test_db)
    crs1 = test_db.create_course("Course 1", "C1")
    crs2 = test_db.create_course("Course 2", "C2")

    doc1 = test_db.create_document(crs1.id, "lec1.pdf", "pdf", 1000, "hash1", "ready")
    chunk1 = DocumentChunk(
        chunk_id="chk1",
        document_id=doc1.id,
        course_id=crs1.id,
        chunk_index=1,
        location_type="page",
        location_index=1,
        location_title="Title",
        content="Content text",
        token_count=20,
        content_hash="chash1",
    )
    test_db.insert_chunks([chunk1])

    cpt1 = repo.get_or_create_concept(crs1.id, "Concept 1")
    cpt2 = repo.get_or_create_concept(crs2.id, "Concept 2")

    # Mapping chunk in C1 to concept in C1 succeeds
    repo.map_concept_to_chunk(cpt1.id, "chk1")
    chunks = repo.get_chunks_for_concept(cpt1.id)
    assert len(chunks) == 1
    assert chunks[0]["is_live"] is True
    assert chunks[0]["is_stale"] is False

    # Mapping chunk in C1 to concept in C2 is rejected
    with pytest.raises(ValueError, match="Cross-course mapping rejected"):
        repo.map_concept_to_chunk(cpt2.id, "chk1")

    # Deleting document cascades to chunk, setting concept_chunks.chunk_id = NULL
    test_db.delete_document(doc1.id)
    chunks_after = repo.get_chunks_for_concept(cpt1.id)
    assert len(chunks_after) == 1
    assert chunks_after[0]["is_stale"] is True
    assert chunks_after[0]["is_live"] is False


def test_course_cascade_deletes_concepts_and_mastery(test_db):
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Course To Delete", "DEL101")
    cpt = repo.get_or_create_concept(crs.id, "Ephemeral Concept")

    repo.recompute_mastery_for_concept(learner_id="learner_default", concept_id=cpt.id)
    m = repo.get_concept_mastery("learner_default", cpt.id)
    assert m is not None

    # Delete course
    test_db.delete_course(crs.id)
    assert repo.get_concept(cpt.id) is None
    assert repo.get_concept_mastery("learner_default", cpt.id) is None


def test_recompute_mastery_from_events(test_db):
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Algorithms", "CS101")
    cpt = repo.get_or_create_concept(crs.id, "Dijkstra")

    # Initially 0 observations -> unassessed
    m0 = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    assert m0.mastery_category == MasteryCategory.UNASSESSED
    assert m0.observation_count == 0

    # Record 2 correct attempts and 1 incorrect attempt
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=1.0,
        score_max=1.0,
        is_correct=True,
    )
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=1.0,
        score_max=1.0,
        is_correct=True,
    )
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=0.0,
        score_max=1.0,
        is_correct=False,
    )

    m1 = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    assert m1.observation_count == 3
    assert m1.correct_count == 2.0
    assert m1.incorrect_count == 1.0
    # (2+1)/(3+2) = 3/5 = 0.600 -> MODERATE
    assert m1.mastery_score == 0.600
    assert m1.mastery_category == MasteryCategory.MODERATE
    assert m1.has_sufficient_evidence is True


def test_repository_normalized_partial_credit_and_different_maxima(test_db):
    """Verify repository normalizes partial credit (e.g. 1/1, 2/3, 4/5) into fractional credits."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Algorithms", "CS101")
    cpt = repo.get_or_create_concept(crs.id, "Dynamic Programming")

    # 1. Full credit 1/1
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=1.0,
        score_max=1.0,
        is_correct=True,
    )
    # 2. Partial credit 2/3 (not 2 raw successes)
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=2.0,
        score_max=3.0,
        is_correct=False,
    )
    # 3. Partial credit 4/5
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=4.0,
        score_max=5.0,
        is_correct=True,
    )

    m = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    # n = 3, k = 1.0 + 2/3 + 0.8 = 2.4667
    # M = (2.4667 + 1) / (3 + 2) = 3.4667 / 5 = 0.693
    assert m.observation_count == 3
    assert round(m.correct_count, 3) == 2.467
    assert m.mastery_score == 0.693
    assert m.mastery_category == MasteryCategory.MODERATE
    assert m.has_sufficient_evidence is True
    assert 0.1870 <= m.uncertainty <= 0.1890


def test_pending_manual_review_never_affects_mastery(test_db):
    """6. Pending manual-review answers must be excluded and never affect mastery."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Algorithms", "CS101")
    cpt = repo.get_or_create_concept(crs.id, "Binary Search")

    # 1. One graded correct answer
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=1.0,
        score_max=1.0,
        is_correct=True,
    )
    m_before = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    assert m_before.observation_count == 1
    assert m_before.correct_count == 1.0
    assert m_before.mastery_score == 0.667

    # 2. Add pending manual review question (unscored)
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="pending_manual_review",
        reference_id="att1",
        question_id=None,
        score_achieved=None,
        score_max=5.0,
        is_correct=None,
        details={"status": "pending_manual_review"},
    )
    m_after = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    # Pending answer must NOT drag down mastery as a zero or increase n!
    assert m_after.observation_count == 1
    assert m_after.correct_count == 1.0
    assert m_after.mastery_score == 0.667


def test_invalid_or_missing_score_possible_cannot_corrupt_estimate(test_db):
    """7. An invalid or non-positive score_possible cannot create an invalid estimate."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Algorithms", "CS101")
    cpt = repo.get_or_create_concept(crs.id, "Sorting")

    # Record valid observation
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att1",
        question_id=None,
        score_achieved=1.0,
        score_max=1.0,
        is_correct=True,
    )

    # Record corrupted/invalid observations:
    # Non-positive max
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att_bad_1",
        question_id=None,
        score_achieved=2.0,
        score_max=0.0,
        is_correct=True,
    )
    # Negative earned score
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att_bad_2",
        question_id=None,
        score_achieved=-1.0,
        score_max=5.0,
        is_correct=False,
    )
    # Score earned exceeds score max
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="exam_question_graded",
        reference_id="att_bad_3",
        question_id=None,
        score_achieved=10.0,
        score_max=5.0,
        is_correct=True,
    )

    m = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    # Only the 1 valid observation is counted
    assert m.observation_count == 1
    assert m.correct_count == 1.0
    assert m.mastery_score == 0.667
    import math
    assert math.isfinite(m.mastery_score)
    assert math.isfinite(m.uncertainty)


def test_corrected_or_finalized_grade_recomputes_without_duplication(test_db):
    """8. A corrected or manually finalized grade recomputes mastery without double-counting."""
    assess_repo = AssessmentRepository(test_db)
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Algorithms", "CS101")
    assess_repo.save_exam(
        Exam(
            id="ex_final_1",
            course_id=crs.id,
            title="Short Answer Exam",
            difficulty=Difficulty.MEDIUM,
            question_count=1,
            questions=[
                ExamQuestion(
                    id="q_sa_1",
                    exam_id="ex_final_1",
                    question_index=1,
                    question_type=QuestionType.SHORT_ANSWER,
                    difficulty=Difficulty.MEDIUM,
                    question_text="Explain Dijkstra",
                    correct_answer="Greedy shortest path",
                    explanation="Uses min-priority queue",
                )
            ],
        )
    )
    cpt = repo.get_or_create_concept(crs.id, "Shortest Paths")
    repo.map_question_to_concept("q_sa_1", cpt.id)

    # Record pending event
    repo.record_event(
        learner_id="learner_default",
        course_id=crs.id,
        concept_id=cpt.id,
        event_type="pending_manual_review",
        reference_id="att_final_1",
        question_id="q_sa_1",
        score_achieved=None,
        score_max=3.0,
        is_correct=None,
    )
    m0 = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    assert m0.observation_count == 0

    # Finalize manual grade: 2/3 credit
    repo.update_or_record_finalized_grade(
        learner_id="learner_default",
        course_id=crs.id,
        attempt_id="att_final_1",
        question_id="q_sa_1",
        concept_id=cpt.id,
        awarded_score=2.0,
        max_score=3.0,
        is_correct=False,
    )
    m1 = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    assert m1.observation_count == 1
    assert round(m1.correct_count, 3) == 0.667
    assert m1.mastery_score == 0.556

    # Update/correct grade to 3/3 credit (supersedes previous grade)
    repo.update_or_record_finalized_grade(
        learner_id="learner_default",
        course_id=crs.id,
        attempt_id="att_final_1",
        question_id="q_sa_1",
        concept_id=cpt.id,
        awarded_score=3.0,
        max_score=3.0,
        is_correct=True,
    )
    m2 = repo.recompute_mastery_for_concept("learner_default", cpt.id)
    # Must STILL be exactly 1 observation (no duplicate event!)
    assert m2.observation_count == 1
    assert m2.correct_count == 1.0
    assert m2.mastery_score == 0.667
