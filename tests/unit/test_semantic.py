"""Unit tests for SemanticSearchService using ChromaDB and SQLite validation."""

from __future__ import annotations

import pytest

from studyos.ingestion.models import DocumentChunk
from studyos.retrieval.embedding import FakeEmbeddingProvider
from studyos.retrieval.semantic import SemanticSearchService
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


@pytest.fixture
def semantic_setup(tmp_path):
    db = Database(tmp_path / "semantic_test.db")
    db.migrate()

    provider = FakeEmbeddingProvider(dimension=384)
    store = ChromaVectorStore(chroma_dir=tmp_path / "chroma_sem", provider=provider)

    c1 = db.create_course("Algorithms", "CS201")
    c2 = db.create_course("Data Mining", "DS301")

    d1 = db.create_document(course_id=c1.id, filename="graphs.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_d1")
    d2 = db.create_document(course_id=c2.id, filename="clustering.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_d2")

    chk1 = DocumentChunk.create(
        document_id=d1.id,
        course_id=c1.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="Dijkstra's shortest path algorithm with priority queues.",
        token_count=7,
    )
    chk2 = DocumentChunk.create(
        document_id=d2.id,
        course_id=c2.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="K-means clustering algorithm and centroid convergence.",
        token_count=7,
    )
    db.insert_chunks([chk1, chk2])
    store.sync(db)

    service = SemanticSearchService(db=db, vector_store=store, provider=provider)
    yield service, db, store, provider, c1, c2, d1, d2, chk1, chk2

    store.close()
    db.close()


class TestSemanticSearchService:
    def test_semantic_retrieval_and_sqlite_enrichment(self, semantic_setup):
        """Semantic search finds relevant chunks and enriches all metadata authoritatively from SQLite."""
        service, db, store, provider, c1, c2, d1, d2, chk1, chk2 = semantic_setup

        results = service.search("Dijkstra shortest path", top_k=5)
        assert len(results) >= 1
        top_match = results[0]

        # Verified metadata from SQLite
        assert top_match.chunk_id == chk1.chunk_id
        assert top_match.course_id == c1.id
        assert top_match.course_name == "Algorithms"
        assert top_match.filename == "graphs.pdf"
        assert top_match.file_type == "pdf"
        assert top_match.semantic_rank == 1
        assert "semantic" in top_match.sources
        assert top_match.breadcrumb == "[Source: graphs.pdf | Page: 1]"

    def test_semantic_course_filtering(self, semantic_setup):
        """Course filter strictly restricts semantic candidate matches to the selected course."""
        service, db, store, provider, c1, c2, d1, d2, chk1, chk2 = semantic_setup

        # Query relevant to clustering (c2), but filtered to c1 (Algorithms)
        results = service.search("clustering centroid", top_k=5, course_ids=[c1.id])
        # chk2 from Data Mining must be strictly excluded
        for r in results:
            assert r.course_id == c1.id
            assert r.chunk_id != chk2.chunk_id

    def test_semantic_document_filtering(self, semantic_setup):
        """Document filter strictly restricts semantic candidates to selected documents."""
        service, db, store, provider, c1, c2, d1, d2, chk1, chk2 = semantic_setup

        results = service.search("algorithm", top_k=5, document_ids=[d2.id])
        assert len(results) == 1
        assert results[0].document_id == d2.id
        assert results[0].chunk_id == chk2.chunk_id

    def test_stale_chroma_records_excluded_by_sqlite_validation(self, semantic_setup):
        """If Chroma contains a vector record for a deleted chunk, SQLite validation discards it."""
        service, db, store, provider, c1, c2, d1, d2, chk1, chk2 = semantic_setup

        # Delete document d1 directly in SQLite without syncing Chroma yet
        db.delete_document(d1.id)

        # Chroma still has chk1's vector, but SQLite validation must filter it out
        results = service.search("Dijkstra shortest path", top_k=5)
        result_ids = [r.chunk_id for r in results]
        assert chk1.chunk_id not in result_ids

    def test_empty_query_and_empty_filters(self, semantic_setup):
        """Whitespace query and empty filter lists return empty result list."""
        service, db, store, provider, c1, c2, d1, d2, chk1, chk2 = semantic_setup

        assert service.search("") == []
        assert service.search("   ") == []
        assert service.search("test", course_ids=[]) == []
        assert service.search("test", document_ids=[]) == []
        assert service.search("test", top_k=0) == []
