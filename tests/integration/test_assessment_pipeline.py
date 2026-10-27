"""Integration tests for the complete assessment pipeline: ingestion -> retrieval -> generation -> public view -> attempt -> grading."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from studyos.assessment.generator import InsufficientEvidenceError
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
from studyos.retrieval.embedding import FakeEmbeddingProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "synthetic"


class TestAssessmentPipelineIntegration:
    """Full end-to-end integration test of practice exam generation, attempt lifecycle, and verified grading."""

    def test_full_exam_lifecycle(self, tmp_path, fixtures_dir):
        # 1. Setup isolated database and Chroma vector store
        data_dir = tmp_path / "studyos_exam_data"
        config = AppConfig(data_dir=data_dir)
        config.ensure_directories()

        db = Database(config.db_path)
        db.migrate()

        embed_provider = FakeEmbeddingProvider(dimension=384)
        vector_store = ChromaVectorStore(chroma_dir=config.chroma_dir, provider=embed_provider)

        # 2. Ingest course documents
        ingestion = IngestionService(db=db, config=config, vector_store=vector_store)
        course_algo = db.create_course("Algorithms and Complexity", "CS101")
        course_net = db.create_course("Computer Networking", "CS102")
        course_empty = db.create_course("Empty Art Course", "ART101")

        r_pdf = ingestion.ingest_file(fixtures_dir / "synthetic_lecture.pdf", course_id=course_algo.id)
        assert r_pdf.status == "ready"

        r_pptx = ingestion.ingest_file(fixtures_dir / "synthetic_slides.pptx", course_id=course_algo.id)
        assert r_pptx.status == "ready"

        r_txt = ingestion.ingest_file(fixtures_dir / "synthetic_doc.txt", course_id=course_net.id)
        assert r_txt.status == "ready"

        retrieval = HybridRetrievalService(db=db, vector_store=vector_store, rrf_k=60)

        # 3. Configure FakeChatProvider to return 3 structured questions
        fake_payload = {
            "questions": [
                {
                    "type": "mcq",
                    "question": "What is the Big-O asymptotic notation used for?",
                    "difficulty": "medium",
                    "options": [
                        {"id": "A", "text": "Measuring exact CPU clock cycles"},
                        {"id": "B", "text": "Describing upper bound on execution growth rate"},
                        {"id": "C", "text": "Computing hardware cache misses"},
                        {"id": "D", "text": "Measuring memory bus voltage"}
                    ],
                    "correct_answer": "B",
                    "explanation": "Big-O characterizes upper asymptotic growth rate [E1].",
                    "evidence_markers": ["E1"]
                },
                {
                    "type": "true_false",
                    "question": "An algorithm with O(n^2) complexity scales faster than O(n) for large n.",
                    "difficulty": "easy",
                    "correct_answer": "False",
                    "explanation": "Quadratic growth rate increases significantly faster than linear growth [E1].",
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
                        "max_score": 2.0
                    },
                    "explanation": "Worst-case analysis guarantees an upper bound on all inputs [E1].",
                    "evidence_markers": ["E1"]
                }
            ]
        }

        fake_chat = FakeChatProvider(default_response=json.dumps(fake_payload))

        service = AssessmentService(
            db=db,
            retrieval_service=retrieval,
            chat_provider=fake_chat,
            config=config,
        )

        # 4. Generate Practice Exam
        exam_req = ExamRequest(
            course_id=course_algo.id,
            question_count=3,
            difficulty=Difficulty.MEDIUM,
            topic="asymptotic complexity",
            title="Algorithms Midterm Review",
        )
        exam = service.generate_exam(exam_req)

        assert exam.id.startswith("ex_")
        assert exam.course_id == course_algo.id
        assert exam.title == "Algorithms Midterm Review"
        assert len(exam.questions) == 3

        # Confirm persisted in repository
        stored_exams = service.list_exams(course_id=course_algo.id)
        assert len(stored_exams) == 1
        assert stored_exams[0].id == exam.id

        # 5. Retrieve Public Exam (Test Zero Answer-Key Leakage)
        public_exam = service.get_exam_for_taking(exam.id)
        assert len(public_exam.questions) == 3

        # Assert no answer keys in public serialized JSON
        public_json = public_exam.model_dump_json()
        assert "correct_answer" not in public_json
        assert "explanation" not in public_json
        assert "rubric" not in public_json

        # 6. Start Attempt & Submit Answers
        attempt = service.start_attempt(exam.id)
        assert attempt.status == AttemptStatus.IN_PROGRESS

        # Student submits:
        # Q1 (MCQ): Correct ("B") -> 1.0 pt
        # Q2 (T/F): Incorrect ("True", correct is "False") -> 0.0 pt
        # Q3 (SA): Skipped (omitted from answers dict) -> 0.0 pt
        sub_req = SubmitAnswersRequest(
            answers={
                exam.questions[0].id: "B",
                exam.questions[1].id: "True",
            }
        )
        result = service.submit_attempt(attempt.id, sub_req)

        # 7. Verify Graded Report
        assert result.attempt_id == attempt.id
        assert result.exam_id == exam.id
        assert result.score == 1.0
        assert result.max_score == 3.0  # 1.0 + 1.0 + 1.0 (defaults points=1.0)
        assert round(result.percentage, 1) == 33.3

        # Check per-question reports
        q1_res = result.questions[0]
        assert q1_res.is_correct is True
        assert q1_res.score == 1.0
        assert len(q1_res.citations) >= 1
        assert q1_res.citations[0].filename in ("synthetic_lecture.pdf", "synthetic_slides.pptx")

        q2_res = result.questions[1]
        assert q2_res.is_correct is False
        assert q2_res.score == 0.0

        q3_res = result.questions[2]
        assert q3_res.submitted_answer is None
        assert q3_res.is_correct is False
        assert q3_res.feedback == "Question was skipped."

        # 8. Course Filter Isolation Check: Empty course raises InsufficientEvidenceError
        fake_chat.clear()
        req_empty = ExamRequest(
            course_id=course_empty.id,
            question_count=3,
        )
        with pytest.raises(InsufficientEvidenceError, match="no relevant course materials found"):
            service.generate_exam(req_empty)
        # LLM must NOT have been called
        assert len(fake_chat.requests) == 0

        vector_store.close()
        db.close()

    def test_historical_citation_lifecycle_pipeline(self, tmp_path, fixtures_dir):
        """End-to-end citation lifecycle: generation, verification, document deletion, and stale provenance."""
        data_dir = tmp_path / "studyos_exam_data_lifecycle"
        config = AppConfig(data_dir=data_dir)
        config.ensure_directories()

        db = Database(config.db_path)
        db.migrate()

        embed_provider = FakeEmbeddingProvider(dimension=384)
        vector_store = ChromaVectorStore(chroma_dir=config.chroma_dir, provider=embed_provider)

        ingestion = IngestionService(db=db, config=config, vector_store=vector_store)
        course = db.create_course("Operating Systems", "CS204")
        r_pdf = ingestion.ingest_file(fixtures_dir / "synthetic_lecture.pdf", course_id=course.id)
        assert r_pdf.status == "ready"

        retrieval = HybridRetrievalService(db=db, vector_store=vector_store, rrf_k=60)

        fake_payload = {
            "questions": [
                {
                    "type": "mcq",
                    "question": "What is virtual memory?",
                    "difficulty": "medium",
                    "options": [
                        {"id": "A", "text": "Hardware abstraction mapping virtual to physical addresses"},
                        {"id": "B", "text": "Solid state hard drive storage"},
                        {"id": "C", "text": "Network buffer cache"},
                        {"id": "D", "text": "Graphics video memory"}
                    ],
                    "correct_answer": "A",
                    "explanation": "Virtual memory provides memory isolation and virtualization [E1].",
                    "evidence_markers": ["E1"]
                },
                {
                    "type": "short_answer",
                    "question": "Explain page faults.",
                    "difficulty": "medium",
                    "correct_answer": "An interrupt raised when a program accesses a page not mapped into physical memory.",
                    "rubric": {
                        "criteria": ["Page not in physical memory", "Trap or exception to kernel"],
                        "key_points": ["page", "physical memory", "interrupt"],
                        "max_score": 2.0
                    },
                    "explanation": "Page faults trap to the OS kernel to load the missing page [E1].",
                    "evidence_markers": ["E1"]
                }
            ]
        }

        fake_chat = FakeChatProvider(default_response=json.dumps(fake_payload))
        service = AssessmentService(
            db=db,
            retrieval_service=retrieval,
            chat_provider=fake_chat,
            config=config,
        )

        # 1. Generate and persist exam with citations
        exam_req = ExamRequest(
            course_id=course.id,
            question_count=2,
            topic="virtual memory",
            title="OS Virtual Memory Quiz",
        )
        exam = service.generate_exam(exam_req)
        assert len(exam.questions) == 2
        q1 = exam.questions[0]
        assert len(q1.sources) >= 1
        src1 = q1.sources[0]
        assert src1.is_stale is False
        assert src1.chunk_id is not None
        assert src1.filename == "synthetic_lecture.pdf"
        assert src1.location_type in ("page", "slide", "section")

        # 2. Start attempt and submit answers (MCQ correct, short-answer submitted)
        attempt = service.start_attempt(exam.id)
        sub_req = SubmitAnswersRequest(
            answers={
                exam.questions[0].id: "A",
                exam.questions[1].id: "A page fault occurs when a virtual address is not in RAM.",
            }
        )
        # Model grader will fallback to manual review because fake_chat returns questions JSON
        res_pending = service.submit_attempt(attempt.id, sub_req)
        assert res_pending.status == AttemptStatus.PENDING_REVIEW
        assert res_pending.score is None
        assert res_pending.percentage is None
        assert res_pending.provisional_score == 1.0  # MCQ awarded 1.0
        assert res_pending.questions[1].status == GradingStatus.PENDING_MANUAL_REVIEW
        assert res_pending.questions[1].score is None

        # 3. Complete manual grading
        res_completed = service.grade_short_answer_manually(
            attempt_id=attempt.id,
            question_id=exam.questions[1].id,
            score=1.0,
            feedback="Accurate description of page fault handling.",
        )
        assert res_completed.status == AttemptStatus.COMPLETED
        assert res_completed.score == 2.0  # 1.0 + 1.0
        assert res_completed.max_score == 2.0
        assert res_completed.percentage == 100.0

        # 4. Delete the source document using supported database workflow
        docs = db.list_documents(course.id)
        assert len(docs) == 1
        doc_id = docs[0].id
        with db.transaction() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))

        # 5. Retrieve existing exam: question and answer key intact, citation marked stale
        exam_after = service.get_exam(exam.id)
        assert exam_after is not None
        assert len(exam_after.questions) == 2
        assert exam_after.questions[0].correct_answer == "A"
        assert exam_after.questions[0].question_text == "What is virtual memory?"

        src_stale = exam_after.questions[0].sources[0]
        assert src_stale.is_stale is True
        assert src_stale.chunk_id is None
        assert src_stale.original_chunk_id == src1.chunk_id
        assert src_stale.filename == "synthetic_lecture.pdf"
        assert src_stale.excerpt is not None

        # 6. Retrieve graded attempt result: historical grading and attempt data intact
        res_after = service.get_attempt_result(attempt.id)
        assert res_after.status == AttemptStatus.COMPLETED
        assert res_after.score == 2.0
        assert res_after.percentage == 100.0
        assert res_after.questions[0].citations[0].is_stale is True
        assert res_after.questions[0].citations[0].chunk_id is None

        vector_store.close()
        db.close()

