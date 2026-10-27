"""Unit tests for assessment repository persistence, atomic rollback, and answer-key security."""

from __future__ import annotations

import json
import pytest

from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    Exam,
    ExamQuestion,
    MCQOption,
    PublicExam,
    QuestionSourceRef,
    QuestionType,
    ShortAnswerRubric,
)
from studyos.assessment.repository import AssessmentRepository
from studyos.ingestion.models import DocumentChunk
from studyos.storage.db import Database


@pytest.fixture
def repo(tmp_path) -> AssessmentRepository:
    db = Database(tmp_path / "test_assessment.db")
    db.migrate()
    return AssessmentRepository(db)


def _create_sample_exam(course_id: str, exam_id: str = "ex_test_01") -> Exam:
    q1 = ExamQuestion(
        id="q_01",
        exam_id=exam_id,
        question_index=1,
        question_type=QuestionType.MCQ,
        difficulty=Difficulty.EASY,
        question_text="What data structure is used in Dijkstra?",
        options=[
            MCQOption(id="A", text="Stack"),
            MCQOption(id="B", text="Priority Queue"),
            MCQOption(id="C", text="Array"),
            MCQOption(id="D", text="Set")
        ],
        correct_answer="B",
        explanation="Priority queue extracts minimum distance vertex efficiently.",
        points=1.0,
    )
    q2 = ExamQuestion(
        id="q_02",
        exam_id=exam_id,
        question_index=2,
        question_type=QuestionType.TRUE_FALSE,
        difficulty=Difficulty.MEDIUM,
        question_text="Bellman-Ford operates on DAGs only.",
        correct_answer="False",
        explanation="Bellman-Ford operates on general directed graphs.",
        points=1.0,
    )
    q3 = ExamQuestion(
        id="q_03",
        exam_id=exam_id,
        question_index=3,
        question_type=QuestionType.SHORT_ANSWER,
        difficulty=Difficulty.HARD,
        question_text="Explain Dijkstra's greedy choice property.",
        correct_answer="Visiting the unvisited vertex with minimal tentative distance guarantees optimal path.",
        explanation="Because edge weights are non-negative, distance cannot decrease.",
        rubric=ShortAnswerRubric(
            criteria=["Mentions minimal tentative distance", "Explains non-negative weights"],
            key_points=["minimal", "non-negative"],
            max_score=2.0,
        ),
        points=2.0,
    )
    return Exam(
        id=exam_id,
        course_id=course_id,
        title="Graph Algorithms Practice Exam",
        topic="Graphs",
        difficulty=Difficulty.MIXED,
        question_count=3,
        questions=[q1, q2, q3],
    )


def test_save_and_get_exam(repo: AssessmentRepository):
    """Saving an exam persists all questions, choices, rubrics, and source references."""
    course = repo.db.create_course("Algorithms", "CS301")
    exam = _create_sample_exam(course.id)

    repo.save_exam(exam)

    retrieved = repo.get_exam(exam.id)
    assert retrieved is not None
    assert retrieved.id == exam.id
    assert retrieved.course_id == course.id
    assert retrieved.title == "Graph Algorithms Practice Exam"
    assert len(retrieved.questions) == 3

    # Check question ordering
    assert [q.question_index for q in retrieved.questions] == [1, 2, 3]

    # Check question details
    q1 = retrieved.questions[0]
    assert q1.question_type == QuestionType.MCQ
    assert q1.correct_answer == "B"
    assert len(q1.options) == 4
    assert q1.options[1].id == "B"

    q3 = retrieved.questions[2]
    assert q3.question_type == QuestionType.SHORT_ANSWER
    assert q3.rubric is not None
    assert q3.rubric.max_score == 2.0


def test_public_exam_security_zero_answer_leakage(repo: AssessmentRepository):
    """The public exam taking view and its serialized JSON must NOT contain answer keys, explanations, or rubrics."""
    course = repo.db.create_course("Algorithms", "CS301")
    exam = _create_sample_exam(course.id)
    repo.save_exam(exam)

    public_exam = repo.get_public_exam(exam.id)
    assert public_exam is not None
    assert isinstance(public_exam, PublicExam)
    assert len(public_exam.questions) == 3

    # 1. Attribute check on PublicQuestion
    q1 = public_exam.questions[0]
    assert not hasattr(q1, "correct_answer")
    assert not hasattr(q1, "explanation")
    assert not hasattr(q1, "rubric")

    # 2. Strict serialization check on dictionary output
    dumped_dict = public_exam.model_dump()
    for q_data in dumped_dict["questions"]:
        assert "correct_answer" not in q_data
        assert "explanation" not in q_data
        assert "rubric" not in q_data

    # 3. Strict serialized JSON string check
    dumped_json = public_exam.model_dump_json()
    assert "correct_answer" not in dumped_json
    assert "explanation" not in dumped_json
    assert "rubric" not in dumped_json
    # The actual secret answers must not appear anywhere in the serialized payload
    assert "Priority queue extracts minimum distance" not in dumped_json
    assert "Visiting the unvisited vertex with minimal tentative" not in dumped_json


def test_stale_source_chunk_detection(repo: AssessmentRepository):
    """When a referenced source chunk is deleted, get_exam preserves historical snapshot and marks citation stale."""
    course = repo.db.create_course("Algorithms", "CS301")
    doc = repo.db.create_document(course.id, "graph_notes.pdf", "pdf", 1024, "hash123")
    chunk = DocumentChunk.create(
        document_id=doc.id,
        course_id=course.id,
        chunk_index=0,
        location_type="page",
        location_index=3,
        content="Shortest path properties.",
        token_count=3,
    )
    repo.db.insert_chunks([chunk])

    exam = _create_sample_exam(course.id, exam_id="ex_stale_test")
    exam.questions[0].sources = [
        QuestionSourceRef(
            chunk_id=chunk.chunk_id,
            evidence_marker="E1",
            filename=doc.filename,
            location_type=chunk.location_type,
            location_index=chunk.location_index,
            excerpt=chunk.content,
        )
    ]
    repo.save_exam(exam)

    # 1. Initially source is active, verified, and resolves current authoritative metadata
    ex_loaded = repo.get_exam(exam.id)
    src_active = ex_loaded.questions[0].sources[0]
    assert src_active.is_stale is False
    assert src_active.chunk_id == chunk.chunk_id
    assert src_active.filename == "graph_notes.pdf"
    assert src_active.location_index == 3
    assert src_active.excerpt == "Shortest path properties."

    # 2. Delete the document (which cascades and deletes the chunk)
    with repo.db.transaction() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?;", (doc.id,))

    # 3. Reopening the exam: question and answer key remain completely intact
    ex_stale = repo.get_exam(exam.id)
    assert ex_stale is not None
    q0 = ex_stale.questions[0]
    assert q0.correct_answer == "B"
    assert q0.question_text == "What data structure is used in Dijkstra?"

    # 4. Source reference: live chunk reference became None, is_stale is True, historical snapshot preserved
    src_stale = q0.sources[0]
    assert src_stale.is_stale is True
    assert src_stale.chunk_id is None  # Live chunk is gone
    assert src_stale.original_chunk_id == chunk.chunk_id  # Historical chunk ID preserved
    assert src_stale.filename == "graph_notes.pdf"  # Historical filename preserved
    assert src_stale.location_index == 3  # Historical location preserved
    assert src_stale.excerpt == "Shortest path properties."  # Historical excerpt preserved


def test_historical_citation_lifecycle_direct_chunk_deletion(repo: AssessmentRepository):
    """Direct deletion of a single chunk retains snapshot and marks citation stale without touching other questions."""
    course = repo.db.create_course("Data Structures", "CS201")
    doc = repo.db.create_document(course.id, "trees.pdf", "pdf", 2048, "hash_trees")
    chunk1 = DocumentChunk.create(
        document_id=doc.id,
        course_id=course.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="Binary search tree invariant.",
        token_count=4,
    )
    chunk2 = DocumentChunk.create(
        document_id=doc.id,
        course_id=course.id,
        chunk_index=1,
        location_type="page",
        location_index=2,
        content="AVL tree balance factor.",
        token_count=4,
    )
    repo.db.insert_chunks([chunk1, chunk2])

    exam = _create_sample_exam(course.id, exam_id="ex_partial_del")
    exam.questions[0].sources = [
        QuestionSourceRef(chunk_id=chunk1.chunk_id, evidence_marker="E1", filename=doc.filename, location_index=1)
    ]
    exam.questions[1].sources = [
        QuestionSourceRef(chunk_id=chunk2.chunk_id, evidence_marker="E2", filename=doc.filename, location_index=2)
    ]
    repo.save_exam(exam)

    # Delete ONLY chunk1 directly
    with repo.db.transaction() as conn:
        conn.execute("DELETE FROM chunks WHERE id = ?;", (chunk1.chunk_id,))

    reloaded = repo.get_exam(exam.id)
    # Question 0's source is now stale
    assert reloaded.questions[0].sources[0].is_stale is True
    assert reloaded.questions[0].sources[0].original_chunk_id == chunk1.chunk_id
    assert reloaded.questions[0].sources[0].filename == "trees.pdf"

    # Question 1's source remains active and verified
    assert reloaded.questions[1].sources[0].is_stale is False
    assert reloaded.questions[1].sources[0].chunk_id == chunk2.chunk_id
    assert reloaded.questions[1].sources[0].filename == "trees.pdf"


def test_current_source_metadata_is_authoritative_while_chunk_exists(repo: AssessmentRepository):
    """While source chunk exists, current SQLite metadata overrides snapshot if metadata was updated."""
    course = repo.db.create_course("Databases", "CS401")
    doc = repo.db.create_document(course.id, "initial_name.pdf", "pdf", 1024, "hash_db")
    chunk = DocumentChunk.create(
        document_id=doc.id,
        course_id=course.id,
        chunk_index=0,
        location_type="slide",
        location_index=10,
        content="Original content.",
        token_count=2,
    )
    repo.db.insert_chunks([chunk])

    exam = _create_sample_exam(course.id, exam_id="ex_authoritative_test")
    exam.questions[0].sources = [
        QuestionSourceRef(chunk_id=chunk.chunk_id, evidence_marker="E1", filename="initial_name.pdf")
    ]
    repo.save_exam(exam)

    # Update document filename in SQLite
    with repo.db.transaction() as conn:
        conn.execute("UPDATE documents SET filename = 'updated_name.pdf' WHERE id = ?;", (doc.id,))
        conn.execute("UPDATE chunks SET content = 'Updated live content from course material.' WHERE id = ?;", (chunk.chunk_id,))

    loaded = repo.get_exam(exam.id)
    src = loaded.questions[0].sources[0]
    assert src.is_stale is False
    # Authoritative current database row is reflected!
    assert src.filename == "updated_name.pdf"
    assert "Updated live content" in src.excerpt


def test_atomic_persistence_rollback_on_failure(repo: AssessmentRepository):
    """If database write fails mid-exam insertion, no partial exam records remain."""
    course = repo.db.create_course("Algorithms", "CS301")
    exam = _create_sample_exam(course.id, exam_id="ex_rollback_test")

    # Tamper with question 3 by pointing to non-existent exam ID in an inconsistent way or simulate error
    # We can test by injecting a duplicate question ID into questions list
    duplicate_q = ExamQuestion(
        id="q_01",  # Same as q1! Primary key collision on exam_questions
        exam_id=exam.id,
        question_index=4,
        question_type=QuestionType.TRUE_FALSE,
        difficulty=Difficulty.EASY,
        question_text="Duplicate ID test",
        correct_answer="True",
        explanation="Exp",
        points=1.0,
    )
    exam.questions.append(duplicate_q)

    with pytest.raises(Exception):
        repo.save_exam(exam)

    # Exam record must NOT exist in the database (rolled back)
    assert repo.get_exam(exam.id) is None


def test_course_cascade_deletes_exams(repo: AssessmentRepository):
    """Deleting a course cascades and deletes all associated exams."""
    course = repo.db.create_course("Algorithms", "CS301")
    exam = _create_sample_exam(course.id, exam_id="ex_cascade_test")
    repo.save_exam(exam)

    assert repo.get_exam(exam.id) is not None

    with repo.db.transaction() as conn:
        conn.execute("DELETE FROM courses WHERE id = ?;", (course.id,))

    assert repo.get_exam(exam.id) is None
