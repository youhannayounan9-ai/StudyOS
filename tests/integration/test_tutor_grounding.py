"""Integration tests for end-to-end AI Tutor grounding, citation resolution, and boundary enforcement."""

from __future__ import annotations

from pathlib import Path
import pytest

from studyos.core.config import AppConfig
from studyos.inference.fake import FakeChatProvider
from studyos.ingestion.service import IngestionService
from studyos.retrieval.embedding import FakeEmbeddingProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievalMode
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database
from studyos.tutoring.models import ResponseStyle, TutorRequest
from studyos.tutoring.service import TutorService


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "synthetic"


class TestTutorGroundingIntegration:
    """End-to-end verification of document ingestion, hybrid retrieval, tutor prompting, and citation provenance."""

    def test_full_pipeline_grounded_answer(self, tmp_path, fixtures_dir):
        """Ingest synthetic files -> Hybrid Retrieval -> Tutor Service -> Validated Citations."""
        data_dir = tmp_path / "studyos_tutor_data"
        config = AppConfig(data_dir=data_dir)
        config.ensure_directories()

        db = Database(config.db_path)
        db.migrate()

        provider = FakeEmbeddingProvider(dimension=384)
        vector_store = ChromaVectorStore(chroma_dir=config.chroma_dir, provider=provider)

        ingestion = IngestionService(db=db, config=config, vector_store=vector_store)

        # 1. Create courses
        course_algo = db.create_course("Algorithms and Systems", "CS101")
        course_net = db.create_course("Computer Networking", "CS102")
        course_empty = db.create_course("Empty Course", "CS999")

        # Ingest documents
        pdf_path = fixtures_dir / "synthetic_lecture.pdf"
        pptx_path = fixtures_dir / "synthetic_slides.pptx"
        txt_path = fixtures_dir / "synthetic_doc.txt"

        r_pdf = ingestion.ingest_file(pdf_path, course_id=course_algo.id)
        assert r_pdf.status == "ready"

        r_pptx = ingestion.ingest_file(pptx_path, course_id=course_algo.id)
        assert r_pptx.status == "ready"

        r_txt = ingestion.ingest_file(txt_path, course_id=course_net.id)
        assert r_txt.status == "ready"

        retrieval = HybridRetrievalService(db=db, vector_store=vector_store, rrf_k=60)

        # 2. Configure FakeChatProvider with citation to [E1]
        fake_chat = FakeChatProvider(
            default_response="Big-O notation describes the upper bound of execution time as input size scales [E1]."
        )

        tutor = TutorService(
            retrieval_service=retrieval,
            chat_provider=fake_chat,
            config=config,
        )

        # 3. Test Query 1: Grounded answer from Algorithms course
        req_algo = TutorRequest(
            query="Explain asymptotic Big-O complexity",
            course_id=course_algo.id,
            top_k=3,
        )
        resp_algo = tutor.answer(req_algo)

        assert resp_algo.unsupported is False
        assert len(resp_algo.citations) == 1
        assert resp_algo.citations[0].marker == "E1"
        assert resp_algo.citations[0].course_id == course_algo.id
        assert resp_algo.citations[0].course_name == "Algorithms and Systems"
        assert resp_algo.citations[0].filename in ("synthetic_lecture.pdf", "synthetic_slides.pptx")
        assert resp_algo.citations[0].location_index >= 1
        assert resp_algo.retrieval_mode == RetrievalMode.HYBRID
        assert resp_algo.evidence_count > 0

        # Verify chat prompt contained delimited evidence context
        last_req = fake_chat.requests[-1]
        user_msg = [m for m in last_req.messages if m.role_str == "user"][0]
        assert "<evidence_context>" in user_msg.content
        assert '<passage id="E1"' in user_msg.content
        assert "</evidence_context>" in user_msg.content

        # 4. Test Query 2: Cross-course isolation check
        # Querying CS101 must NEVER return or cite chunks from CS102 (synthetic_doc.txt)
        req_cross = TutorRequest(
            query="computer networking virtual memory",
            course_id=course_algo.id,
            top_k=5,
        )
        resp_cross = tutor.answer(req_cross)
        for c in resp_cross.citations:
            assert c.course_id == course_algo.id
            assert c.filename != "synthetic_doc.txt"

        # 5. Test Query 3: Empty course produces immediate refusal without LLM call
        fake_chat.clear()
        req_empty = TutorRequest(
            query="Explain asymptotic Big-O complexity",
            course_id=course_empty.id,
            top_k=3,
        )
        resp_empty = tutor.answer(req_empty)
        assert resp_empty.unsupported is True
        assert len(resp_empty.citations) == 0
        assert "The uploaded materials do not provide enough evidence" in resp_empty.answer
        assert len(fake_chat.requests) == 0

        # 6. Test Query 4: Document filter isolation
        req_doc = TutorRequest(
            query="algorithms",
            course_id=course_algo.id,
            document_ids=[r_pptx.document_id],
            top_k=5,
        )
        resp_doc = tutor.answer(req_doc)
        if resp_doc.citations:
            for c in resp_doc.citations:
                assert c.document_id == r_pptx.document_id
                assert c.filename == "synthetic_slides.pptx"

        # 7. Test Query 5: Model-level refusal on insufficient evidence
        fake_chat.clear()
        fake_chat.queue_response(
            "The uploaded materials do not provide enough evidence to answer this question reliably. "
            "The retrieved material discusses virtual memory rather than cellular biology."
        )
        req_irrelevant = TutorRequest(
            query="What is photosynthesis and cellular respiration?",
            course_id=course_net.id,
        )
        resp_irrelevant = tutor.answer(req_irrelevant)

        assert resp_irrelevant.unsupported is True
        assert len(resp_irrelevant.citations) == 0
        assert "The uploaded materials do not provide enough evidence" in resp_irrelevant.answer

        # 8. Test Query 6: Degraded lexical search answering
        retrieval_degraded = HybridRetrievalService(db=db, vector_store=None)
        tutor_degraded = TutorService(
            retrieval_service=retrieval_degraded,
            chat_provider=fake_chat,
            config=config,
        )
        fake_chat.queue_response("Big-O notation characterizes growth rates [E1].")

        req_deg = TutorRequest(
            query="Big-O",
            course_id=course_algo.id,
        )
        resp_deg = tutor_degraded.answer(req_deg)

        assert resp_deg.retrieval_mode == RetrievalMode.DEGRADED_LEXICAL
        assert any("DEGRADED_LEXICAL" in w for w in resp_deg.warnings)
        assert len(resp_deg.citations) == 1
        assert resp_deg.citations[0].marker == "E1"

        vector_store.close()
        db.close()
