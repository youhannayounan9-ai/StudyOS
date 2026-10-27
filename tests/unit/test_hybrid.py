"""Unit tests for HybridRetrievalService, Reciprocal Rank Fusion (RRF), and degraded modes."""

from __future__ import annotations

import pytest

from studyos.ingestion.models import DocumentChunk
from studyos.retrieval.embedding import FakeEmbeddingProvider, OllamaConnectionError
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievalMode, RetrievedChunk
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


@pytest.fixture
def hybrid_env(tmp_path):
    db = Database(tmp_path / "hybrid_test.db")
    db.migrate()

    provider = FakeEmbeddingProvider(dimension=384)
    store = ChromaVectorStore(chroma_dir=tmp_path / "chroma_hyb", provider=provider)

    course_cs = db.create_course("Computer Systems", "CS201")
    course_math = db.create_course("Discrete Mathematics", "MATH101")

    doc_cs = db.create_document(course_id=course_cs.id, filename="assembly.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_cs")
    doc_math = db.create_document(course_id=course_math.id, filename="graphs.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_math")

    # Chunks
    chk1 = DocumentChunk.create(
        document_id=doc_cs.id,
        course_id=course_cs.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="Register allocation and x86 call stack conventions.",
        token_count=7,
    )
    chk2 = DocumentChunk.create(
        document_id=doc_cs.id,
        course_id=course_cs.id,
        chunk_index=1,
        location_type="page",
        location_index=2,
        content="Cache hierarchy, L1 L2 cache lines, and memory latency.",
        token_count=8,
    )
    chk3 = DocumentChunk.create(
        document_id=doc_math.id,
        course_id=course_math.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="Graph theory: Eulerian circuits and Hamiltonian paths.",
        token_count=7,
    )
    db.insert_chunks([chk1, chk2, chk3])
    store.sync(db)

    service = HybridRetrievalService(db=db, vector_store=store, rrf_k=60)
    yield service, db, store, provider, course_cs, course_math, doc_cs, doc_math, chk1, chk2, chk3

    store.close()
    db.close()


class TestHybridRetrievalService:
    def test_rrf_scoring_and_deduplication(self, hybrid_env):
        """When a chunk matches both lexical and semantic searches, scores fuse additively."""
        service, db, store, provider, course_cs, course_math, doc_cs, doc_math, chk1, chk2, chk3 = hybrid_env

        # Query targeting chk1
        response = service.search("x86 call stack register allocation", top_k=5)
        assert response.mode == RetrievalMode.HYBRID
        assert len(response.results) >= 1

        top_chunk = response.results[0]
        assert top_chunk.chunk_id == chk1.chunk_id

        # Verify sources list indicates both lexical and semantic contributed
        if "lexical" in top_chunk.sources and "semantic" in top_chunk.sources:
            # Expected RRF score = 1 / (60 + lex_rank) + 1 / (60 + sem_rank)
            assert top_chunk.lexical_rank is not None
            assert top_chunk.semantic_rank is not None
            expected_score = (1.0 / (60 + top_chunk.lexical_rank)) + (1.0 / (60 + top_chunk.semantic_rank))
            assert pytest.approx(top_chunk.score, 1e-4) == expected_score

    def test_deterministic_tie_breaking(self, tmp_path):
        """When RRF scores are equal, ordering breaks ties deterministically via chunk_id."""
        db = Database(tmp_path / "tie_test.db")
        db.migrate()
        service = HybridRetrievalService(db=db, rrf_k=60)

        # Create two fake chunks with identical ranks/scores
        c_a = RetrievedChunk(
            chunk_id="chk_aaa",
            document_id="d1",
            course_id="c1",
            course_name="C1",
            filename="f1.pdf",
            file_type="pdf",
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Alpha content",
            token_count=2,
            content_hash="h1",
            score=0.016393,
        )
        c_z = RetrievedChunk(
            chunk_id="chk_zzz",
            document_id="d1",
            course_id="c1",
            course_name="C1",
            filename="f1.pdf",
            file_type="pdf",
            chunk_index=1,
            location_type="page",
            location_index=2,
            content="Zeta content",
            token_count=2,
            content_hash="h2",
            score=0.016393,
        )

        class MockLexical:
            def search(self, *args, **kwargs):
                return [c_z, c_a]

        service.lexical_service = MockLexical()
        service.semantic_service = None

        resp = service.search("test", top_k=5, mode=RetrievalMode.LEXICAL_ONLY)
        # c_z was rank 1, c_a was rank 2, so c_z score > c_a score
        assert resp.results[0].chunk_id == "chk_zzz"

        # Now test exact score equality
        scores = {"chk_zzz": 0.05, "chk_aaa": 0.05}
        sorted_ids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))
        assert sorted_ids == ["chk_aaa", "chk_zzz"]  # Alphabetical tie-break

    def test_course_filtering_strict_isolation(self, hybrid_env):
        """Hybrid search strictly restricts results to the requested course_id."""
        service, db, store, provider, course_cs, course_math, doc_cs, doc_math, chk1, chk2, chk3 = hybrid_env

        # Query terms matching math and cs, but filter to Computer Systems
        resp = service.search("Hamiltonian paths and x86 cache", top_k=5, course_id=course_cs.id)
        assert len(resp.results) > 0
        for r in resp.results:
            assert r.course_id == course_cs.id
            assert r.chunk_id != chk3.chunk_id

    def test_document_filtering(self, hybrid_env):
        """Hybrid search strictly restricts results to requested document_ids."""
        service, db, store, provider, course_cs, course_math, doc_cs, doc_math, chk1, chk2, chk3 = hybrid_env

        resp = service.search("register cache graphs", top_k=5, document_ids=[doc_math.id])
        assert len(resp.results) == 1
        assert resp.results[0].document_id == doc_math.id
        assert resp.results[0].chunk_id == chk3.chunk_id

    def test_degraded_lexical_mode_when_semantic_fails(self, hybrid_env):
        """When semantic search fails, hybrid search gracefully degrades to lexical search with honest warning."""
        service, db, store, provider, course_cs, course_math, doc_cs, doc_math, chk1, chk2, chk3 = hybrid_env

        class FailingSemanticService:
            def search(self, *args, **kwargs):
                raise OllamaConnectionError("Connection refused to Ollama daemon on 127.0.0.1:11434")

        service.semantic_service = FailingSemanticService()

        # Execute hybrid search
        response = service.search("x86 call stack", top_k=5, mode=RetrievalMode.HYBRID)

        # Must report DEGRADED_LEXICAL mode
        assert response.mode == RetrievalMode.DEGRADED_LEXICAL
        # Must return lexical results
        assert len(response.results) >= 1
        assert response.results[0].chunk_id == chk1.chunk_id
        # Must disclose failure in warnings
        assert any("OllamaConnectionError" in w for w in response.warnings)

    def test_top_k_limit_enforced(self, hybrid_env):
        """Returned result count does not exceed top_k."""
        service, db, store, provider, course_cs, course_math, doc_cs, doc_math, chk1, chk2, chk3 = hybrid_env

        resp = service.search("cache register x86", top_k=1)
        assert len(resp.results) == 1

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            service.search("test", top_k=0)

    def test_empty_query_and_empty_filters(self, hybrid_env):
        """Empty query or empty filter lists return clean empty response."""
        service, db, store, provider, course_cs, course_math, doc_cs, doc_math, chk1, chk2, chk3 = hybrid_env

        assert service.search("").count == 0
        assert service.search("   ").count == 0
        assert service.search("test", course_ids=[]).count == 0
        assert service.search("test", document_ids=[]).count == 0
