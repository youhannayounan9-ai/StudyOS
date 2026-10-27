"""Unit tests for LexicalSearchService using SQLite FTS5 BM25 scoring."""

from __future__ import annotations

import pytest

from studyos.ingestion.models import DocumentChunk
from studyos.retrieval.lexical import LexicalSearchService
from studyos.storage.db import Database


@pytest.fixture
def lexical_db(tmp_path) -> Database:
    db = Database(tmp_path / "lexical_test.db")
    db.migrate()

    # Create Course A and Course B
    c_a = db.create_course("Operating Systems", "CS301")
    c_b = db.create_course("Computer Networks", "CS302")

    # Documents for Course A
    d_a1 = db.create_document(course_id=c_a.id, filename="scheduling.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_a1")
    d_a2 = db.create_document(course_id=c_a.id, filename="memory.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_a2")

    # Documents for Course B
    d_b1 = db.create_document(course_id=c_b.id, filename="tcp.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_b1")

    # Chunks
    chk_a1 = DocumentChunk.create(
        document_id=d_a1.id,
        course_id=c_a.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="Preemptive round-robin scheduling and priority queue dispatching in operating systems kernel.",
        token_count=12,
    )
    chk_a2 = DocumentChunk.create(
        document_id=d_a2.id,
        course_id=c_a.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="Virtual memory paging, Translation Lookaside Buffer (TLB), and dirty bit tracking.",
        token_count=12,
    )
    chk_b1 = DocumentChunk.create(
        document_id=d_b1.id,
        course_id=c_b.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        content="Transmission Control Protocol (TCP) congestion control, sliding window, and packet retransmission scheduling.",
        token_count=14,
    )

    db.insert_chunks([chk_a1, chk_a2, chk_b1])
    yield db
    db.close()


class TestLexicalSearchService:
    def test_query_sanitization(self):
        """Sanitization extracts alphanumeric tokens and quotes them safely for FTS5."""
        assert LexicalSearchService.sanitize_query("") is None
        assert LexicalSearchService.sanitize_query("   ") is None
        assert LexicalSearchService.sanitize_query("!@#$%^&*()") is None
        assert LexicalSearchService.sanitize_query("Dijkstra's algorithm") == '"Dijkstra" "s" "algorithm"'
        assert LexicalSearchService.sanitize_query('unclosed "quote and colon:foo') == '"unclosed" "quote" "and" "colon" "foo"'
        assert LexicalSearchService.sanitize_query("операционные системы") == '"операционные" "системы"'

    def test_basic_search_and_ranking(self, lexical_db):
        """Relevant keyword queries match and return correctly populated RetrievedChunk objects."""
        service = LexicalSearchService(lexical_db)
        results = service.search("paging TLB", top_k=5)

        assert len(results) == 1
        chunk = results[0]
        assert "Virtual memory paging" in chunk.content
        assert chunk.course_name == "Operating Systems"
        assert chunk.filename == "memory.pdf"
        assert chunk.lexical_rank == 1
        assert "lexical" in chunk.sources
        assert chunk.breadcrumb == "[Source: memory.pdf | Page: 1]"

    def test_unmatched_query_returns_empty_list(self, lexical_db):
        """Queries with no matching terms in FTS5 return empty list without errors."""
        service = LexicalSearchService(lexical_db)
        results = service.search("quantum teleportation entanglement", top_k=5)
        assert results == []

    def test_malformed_and_special_character_input(self, lexical_db):
        """Punctuation, SQL injection, and wildcards do not trigger FTS5 syntax errors."""
        service = LexicalSearchService(lexical_db)

        # Quotes and booleans
        res1 = service.search('AND OR NOT " " * ^ %', top_k=5)
        assert isinstance(res1, list)

        # SQL injection attempt
        res2 = service.search("'; DROP TABLE chunks; --", top_k=5)
        assert isinstance(res2, list)

        # Whitespace
        assert service.search("     ") == []

    def test_course_filtering_strict_isolation(self, lexical_db):
        """Filtering by course_id strictly isolates matches to that course, preventing leaks."""
        service = LexicalSearchService(lexical_db)

        # Term 'scheduling' exists in Course A (kernel scheduling) and Course B (packet scheduling)
        results_all = service.search("scheduling", top_k=10)
        assert len(results_all) == 2

        # Filter strictly for Course A
        c_a = lexical_db.get_course_by_code("CS301")
        assert c_a is not None
        results_a = service.search("scheduling", top_k=10, course_ids=[c_a.id])
        assert len(results_a) == 1
        assert results_a[0].course_id == c_a.id
        assert "round-robin" in results_a[0].content

        # Filter strictly for Course B
        c_b = lexical_db.get_course_by_code("CS302")
        assert c_b is not None
        results_b = service.search("scheduling", top_k=10, course_ids=[c_b.id])
        assert len(results_b) == 1
        assert results_b[0].course_id == c_b.id
        assert "TCP" in results_b[0].content

    def test_document_and_combined_filtering(self, lexical_db):
        """Document filters and combined course/document filters work strictly."""
        service = LexicalSearchService(lexical_db)
        c_a = lexical_db.get_course_by_code("CS301")
        assert c_a is not None

        # Query matching multiple docs in Course A
        with lexical_db.get_connection() as conn:
            d_a1 = conn.execute("SELECT id FROM documents WHERE filename = 'scheduling.pdf'").fetchone()["id"]
            d_a2 = conn.execute("SELECT id FROM documents WHERE filename = 'memory.pdf'").fetchone()["id"]

        # Filter to document d_a1 specifically
        res_doc = service.search("operating systems", top_k=10, document_ids=[d_a1])
        assert len(res_doc) == 1
        assert res_doc[0].document_id == d_a1

        # Combined filter matching document d_a2 in Course A
        res_combined = service.search(
            "memory",
            top_k=10,
            course_ids=[c_a.id],
            document_ids=[d_a2],
        )
        assert len(res_combined) == 1
        assert res_combined[0].document_id == d_a2

        # Combined filter with mismatched document and course returns empty
        res_mismatch = service.search(
            "scheduling",
            top_k=10,
            course_ids=[c_a.id],
            document_ids=["doc_nonexistent"],
        )
        assert res_mismatch == []

    def test_empty_filter_lists_return_empty(self, lexical_db):
        """Passing an explicit empty list for course_ids or document_ids returns empty list."""
        service = LexicalSearchService(lexical_db)
        assert service.search("scheduling", course_ids=[]) == []
        assert service.search("scheduling", document_ids=[]) == []
        assert service.search("scheduling", top_k=0) == []
