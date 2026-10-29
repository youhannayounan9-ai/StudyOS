"""Unit and integration boundary tests for StudyOS Streamlit UI components and services."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest

from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    ExamRequest,
    GradingStatus,
    PublicExam,
    PublicQuestion,
    QuestionType,
    SubmitAnswersRequest,
)
from studyos.core.config import AppConfig
from studyos.learner.models import MasteryCategory
from studyos.storage.db import Database
from studyos.ui.state import (
    UIServices,
    clean_temp_file,
    create_services,
    format_mastery_badge,
    init_session_state,
)


@pytest.fixture
def temp_ui_db(tmp_path: Path):
    """Fixture providing isolated temporary database and config."""
    db_file = tmp_path / "ui_test.db"
    chroma_dir = tmp_path / "chroma_test"
    cfg = AppConfig(
        db_path=str(db_file),
        chroma_dir=str(chroma_dir),
        data_dir=tmp_path,
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )
    return cfg


def test_create_services_with_fake_inference(temp_ui_db: AppConfig):
    """Verify UIServices creation with fake inference works 100% offline."""
    services = create_services(config=temp_ui_db, use_fake_inference=True)

    assert isinstance(services, UIServices)
    assert services.ollama_ready is True
    assert services.embedding_ready is True
    assert services.tutor_service is not None
    assert services.assessment_service is not None
    assert services.learner_service is not None
    assert services.research_service is not None


def test_create_services_offline_ollama_resilience(temp_ui_db: AppConfig):
    """Verify UIServices initializes safely even when Ollama is offline."""
    # Base URL pointing to unused port
    temp_ui_db.ollama_base_url = "http://127.0.0.1:59999"
    services = create_services(config=temp_ui_db, use_fake_inference=False)

    assert isinstance(services, UIServices)
    assert services.ollama_ready is False
    assert services.embedding_ready is False
    # Database and ingestion service remain fully functional
    assert services.db is not None
    assert services.ingestion_service is not None
    assert services.hybrid_service is not None


def test_format_mastery_badge():
    """Verify badge formatting across all MasteryCategory variants."""
    categories = [
        MasteryCategory.STRONG,
        MasteryCategory.MODERATE,
        MasteryCategory.WEAK,
        MasteryCategory.NEEDS_EVIDENCE,
        MasteryCategory.UNASSESSED,
    ]
    for cat in categories:
        label, bg, fg = format_mastery_badge(cat)
        assert isinstance(label, str)
        assert bg.startswith("#")
        assert fg.startswith("#")


def test_clean_temp_file():
    """Verify clean_temp_file safely deletes existing files and ignores None/non-existent."""
    clean_temp_file(None)
    clean_temp_file("non_existent_file_path_12345.xyz")

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"sample data")
        temp_name = tf.name

    assert Path(temp_name).exists()
    clean_temp_file(temp_name)
    assert not Path(temp_name).exists()


def test_answer_key_isolation_in_taking_interface(temp_ui_db: AppConfig):
    """Verify that get_exam_for_taking produces PublicExam with ZERO correct answers or rubrics."""
    services = create_services(config=temp_ui_db, use_fake_inference=True)

    # 1. Create Course and Ingest a synthetic document
    course = services.db.create_course("Computer Science", "CS101")
    txt_file = temp_ui_db.data_dir / "test_notes.txt"
    txt_file.write_text(
        "Dijkstra's algorithm finds single-source shortest paths on non-negative weighted graphs.",
        encoding="utf-8",
    )
    services.ingestion_service.ingest_file(txt_file, course_id=course.id)

    # 2. Generate Exam
    exam = services.assessment_service.generate_exam(
        ExamRequest(
            course_id=course.id,
            question_count=2,
            difficulty=Difficulty.EASY,
            question_types=[QuestionType.MCQ, QuestionType.TRUE_FALSE],
        )
    )

    # Verify internal exam contains answer key
    assert len(exam.questions) > 0
    for q in exam.questions:
        assert hasattr(q, "correct_answer")
        assert q.correct_answer is not None

    # 3. Retrieve PublicExam for taking (as used by UI)
    public_exam = services.assessment_service.get_exam_for_taking(exam.id)
    assert isinstance(public_exam, PublicExam)
    assert public_exam.id == exam.id
    assert len(public_exam.questions) == len(exam.questions)

    # STRICT INVARIANT: PublicQuestion has no correct_answer, rubric, or explanation
    for pq in public_exam.questions:
        assert isinstance(pq, PublicQuestion)
        assert not hasattr(pq, "correct_answer")
        assert not hasattr(pq, "rubric")
        assert not hasattr(pq, "explanation")

    # 4. Attempt and Submit
    attempt = services.assessment_service.start_attempt(exam.id)
    # Student submits answers
    answers = {public_exam.questions[0].id: "B"}
    result = services.assessment_service.submit_attempt(
        attempt_id=attempt.id,
        request=SubmitAnswersRequest(answers=answers),
    )
    assert result.attempt_id == attempt.id
    assert result.max_score > 0.0


def test_pending_manual_review_provisional_score(temp_ui_db: AppConfig):
    """Verify that exams with short-answer questions return provisional scores without false zero."""
    services = create_services(config=temp_ui_db, use_fake_inference=True)

    course = services.db.create_course("Operating Systems", "CS301")
    txt_file = temp_ui_db.data_dir / "os_notes.txt"
    txt_file.write_text(
        "Virtual memory uses paging to isolate address spaces and protect process execution.",
        encoding="utf-8",
    )
    services.ingestion_service.ingest_file(txt_file, course_id=course.id)

    # Generate exam with short-answer
    exam = services.assessment_service.generate_exam(
        ExamRequest(
            course_id=course.id,
            question_count=2,
            difficulty=Difficulty.MEDIUM,
            question_types=[QuestionType.SHORT_ANSWER],
        )
    )

    public_exam = services.assessment_service.get_exam_for_taking(exam.id)
    attempt = services.assessment_service.start_attempt(exam.id)

    # Student submits short answer
    answers = {public_exam.questions[0].id: "Virtual memory isolates processes through paging."}
    result = services.assessment_service.submit_attempt(
        attempt_id=attempt.id,
        request=SubmitAnswersRequest(answers=answers),
    )

    assert result.status == AttemptStatus.PENDING_REVIEW
    assert result.needs_manual_review is True
    # Final score is None (not false 0.0)
    assert result.score is None
    assert result.percentage is None
    # Provisional score is tracked
    assert result.provisional_score is not None


def test_external_research_opt_in_boundary(temp_ui_db: AppConfig):
    """Verify that external research strictly enforces explicit user opt-in."""
    services = create_services(config=temp_ui_db, use_fake_inference=True)

    # Calling research without explicit opt-in must be rejected
    with pytest.raises(Exception):
        services.research_service.research(
            query="test inquiry",
            explicit_opt_in=False,
        )

    # With explicit opt-in, it completes safely using fake provider
    summary = services.research_service.research(
        query="test inquiry",
        explicit_opt_in=True,
    )
    assert summary is not None
    assert summary.query == "test inquiry"


def test_ingestion_duplicate_and_force_reindex(temp_ui_db: AppConfig):
    """Verify document duplicate detection and force_reindex replacement."""
    services = create_services(config=temp_ui_db, use_fake_inference=True)
    course = services.db.create_course("Algorithms", "CS201")

    doc_file = temp_ui_db.data_dir / "algo.txt"
    doc_file.write_text("Binary search requires a sorted array with random access.", encoding="utf-8")

    # First ingestion succeeds
    res1 = services.ingestion_service.ingest_file(doc_file, course_id=course.id)
    assert res1.status == "ready"

    # Second ingestion of same content without force_reindex returns duplicate status
    res2 = services.ingestion_service.ingest_file(doc_file, course_id=course.id, force_reindex=False)
    assert res2.status == "duplicate"
    assert "already exists" in res2.warnings[0]

    # Ingestion with force_reindex=True replaces the document cleanly
    res3 = services.ingestion_service.ingest_file(doc_file, course_id=course.id, force_reindex=True)
    assert res3.status == "ready"


def test_adaptive_study_plan_generation(temp_ui_db: AppConfig):
    """Verify adaptive study planner creates valid time-budgeted plans."""
    services = create_services(config=temp_ui_db, use_fake_inference=True)
    course = services.db.create_course("Databases", "CS401")

    # Add concepts
    c1 = services.learner_service.create_concept(course.id, "B-Trees", "Self-balancing search tree")
    c2 = services.learner_service.create_concept(course.id, "ACID Properties", "Transaction guarantees")

    plan = services.learner_service.generate_adaptive_study_plan(
        course_id=course.id,
        time_budget_minutes=30,
        max_items=5,
    )
    assert plan is not None
    assert plan.time_budget_minutes == 30
    assert plan.allocated_minutes <= 30


def test_streamlit_apptest_headless_run(tmp_path: Path, monkeypatch):
    """Test full Streamlit app execution in headless AppTest mode."""
    from streamlit.testing.v1 import AppTest

    db_file = tmp_path / "apptest.db"
    chroma_dir = tmp_path / "apptest_chroma"
    monkeypatch.setenv("STUDYOS_DB_PATH", str(db_file))
    monkeypatch.setenv("STUDYOS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STUDYOS_CHROMA_DIR", str(chroma_dir))

    app_path = Path(__file__).resolve().parents[2] / "studyos" / "ui" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()
    assert not at.exception

