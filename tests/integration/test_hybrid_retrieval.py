"""Integration tests for end-to-end document ingestion, vector index synchronization, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path
import time
import pytest

from studyos.core.config import AppConfig
from studyos.ingestion.service import IngestionService
from studyos.retrieval.embedding import FakeEmbeddingProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievalMode
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "synthetic"


class TestHybridRetrievalIntegration:
    def test_full_ingestion_and_hybrid_retrieval_pipeline(self, tmp_path, fixtures_dir):
        """End-to-end test: Ingest documents -> Sync ChromaDB -> Perform filtered hybrid search."""
        data_dir = tmp_path / "studyos_data"
        config = AppConfig(data_dir=data_dir)
        config.ensure_directories()

        db = Database(config.db_path)
        db.migrate()

        provider = FakeEmbeddingProvider(dimension=384)
        vector_store = ChromaVectorStore(chroma_dir=config.chroma_dir, provider=provider)

        # 1. Initialize Ingestion Service wired with vector store
        ingestion = IngestionService(db=db, config=config, vector_store=vector_store)

        course_algo = db.create_course("Algorithms and Systems", "CS101")
        course_net = db.create_course("Computer Networking", "CS102")

        pdf_path = fixtures_dir / "synthetic_lecture.pdf"
        pptx_path = fixtures_dir / "synthetic_slides.pptx"
        docx_path = fixtures_dir / "synthetic_notes.docx"
        txt_path = fixtures_dir / "synthetic_doc.txt"

        # Ingest multi-format documents into course_algo
        r_pdf = ingestion.ingest_file(pdf_path, course_id=course_algo.id)
        assert r_pdf.status == "ready"

        r_pptx = ingestion.ingest_file(pptx_path, course_id=course_algo.id)
        assert r_pptx.status == "ready"

        r_docx = ingestion.ingest_file(docx_path, course_id=course_algo.id)
        assert r_docx.status == "ready"

        # Ingest txt into course_net
        r_txt = ingestion.ingest_file(txt_path, course_id=course_net.id)
        assert r_txt.status == "ready"

        # Verify all chunks in SQLite are indexed in Chroma
        total_chunks = db.count_fts_rows()
        assert total_chunks > 0
        assert vector_store.count() == total_chunks

        # 2. Execute Hybrid Retrieval Service
        retrieval = HybridRetrievalService(db=db, vector_store=vector_store, rrf_k=60)

        # Test query 1: Search for Asymptotic Complexity in CS101
        start_time = time.perf_counter()
        res_algo = retrieval.search(
            query="asymptotic complexity Big O notation",
            top_k=5,
            course_id=course_algo.id,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Performance acceptance criteria: < 500ms
        assert latency_ms < 500.0, f"Retrieval took {latency_ms:.2f}ms (must be < 500ms)"
        assert res_algo.mode == RetrievalMode.HYBRID
        assert len(res_algo.results) > 0

        # Top match should be from the PDF or DOCX mentioning Big-O
        top_match = res_algo.results[0]
        assert top_match.course_id == course_algo.id
        assert "Big-O" in top_match.content or "complexity" in top_match.content.lower()
        assert top_match.score > 0.0
        assert len(top_match.sources) >= 1

        # Test query 2: Search with strict document filtering
        res_doc = retrieval.search(
            query="algorithms",
            top_k=5,
            course_id=course_algo.id,
            document_ids=[r_pptx.document_id],
        )
        assert len(res_doc.results) > 0
        for chunk in res_doc.results:
            assert chunk.document_id == r_pptx.document_id
            assert chunk.course_id == course_algo.id

        # Test query 3: Cross-course isolation check
        # Search for terms in CS102 (networking), filtered to CS101 (algorithms)
        res_isolated = retrieval.search(
            query="StudyOS Document Normalization",
            top_k=5,
            course_id=course_algo.id,
        )
        for chunk in res_isolated.results:
            assert chunk.course_id == course_algo.id
            assert chunk.document_id != r_txt.document_id

        vector_store.close()
        db.close()

    def test_sync_existing_stage1_chunks_without_reingesting(self, tmp_path, fixtures_dir):
        """Pre-existing chunks in SQLite can be indexed by Chroma without re-parsing/ingesting files."""
        config = AppConfig(data_dir=tmp_path / "stage1_existing")
        config.ensure_directories()
        db = Database(config.db_path)
        db.migrate()

        # Ingest without vector store (Stage 1 ingestion service)
        ingestion_s1 = IngestionService(db=db, config=config, vector_store=None)
        course = db.create_course("Operating Systems", "CS201")
        r = ingestion_s1.ingest_file(fixtures_dir / "synthetic_lecture.pdf", course_id=course.id)
        assert r.status == "ready"

        # At this point, Chroma has 0 vectors
        provider = FakeEmbeddingProvider(dimension=384)
        store = ChromaVectorStore(chroma_dir=config.chroma_dir, provider=provider)
        assert store.count() == 0

        # Synchronize pre-existing SQLite chunks
        stats = store.sync(db)
        assert stats.added == r.chunk_count
        assert store.count() == r.chunk_count

        # Search now works
        service = HybridRetrievalService(db=db, vector_store=store)
        res = service.search("Foundations of Algorithms", top_k=3, course_id=course.id)
        assert len(res.results) > 0
        assert res.results[0].document_id == r.document_id

        store.close()
        db.close()
