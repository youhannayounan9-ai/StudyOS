"""Unit tests for SQLite FTS5 trigger-based synchronization and transaction rollback."""

from __future__ import annotations

import pytest

from studyos.ingestion.models import DocumentChunk
from studyos.storage.db import Database


@pytest.fixture
def fts_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    yield db
    db.close()


class TestFTS5Synchronization:
    def test_insert_trigger_populates_fts5(self, fts_db):
        """Inserting into chunks automatically populates chunks_fts without application-side FTS writes."""
        course = fts_db.create_course("Algorithms", "CS301")
        doc = fts_db.create_document(course.id, "graph.pdf", "pdf", 1000, "hash_g1")

        chunk = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Dijkstra algorithm computes shortest paths on non-negative weighted graphs.",
            token_count=12,
        )

        assert fts_db.count_fts_rows() == 0
        fts_db.insert_chunks([chunk])

        # FTS table must now contain the chunk
        assert fts_db.count_fts_rows() == 1

        # Search for keyword via FTS5 MATCH
        results = fts_db.query_fts("Dijkstra")
        assert len(results) == 1
        assert results[0]["chunk_id"] == chunk.chunk_id
        assert results[0]["document_id"] == doc.id
        assert results[0]["course_id"] == course.id

    def test_delete_trigger_removes_from_fts5(self, fts_db):
        """Deleting a single chunk row automatically removes it from chunks_fts via trigger chunks_ad."""
        course = fts_db.create_course("Operating Systems", "CS302")
        doc = fts_db.create_document(course.id, "sched.pdf", "pdf", 1000, "hash_s1")

        chunk = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Round Robin scheduling gives every process a fixed quantum.",
            token_count=10,
        )
        fts_db.insert_chunks([chunk])
        assert fts_db.count_fts_rows() == 1

        # Delete chunk directly from base table
        with fts_db.get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE id = ?;", (chunk.chunk_id,))

        assert fts_db.count_fts_rows() == 0
        assert len(fts_db.query_fts("Round Robin")) == 0

    def test_update_trigger_updates_content_and_attribution(self, fts_db):
        """
        Trigger chunks_au (AFTER UPDATE OF content, document_id, course_id ON chunks)
        refreshes chunks_fts when content or attribution changes.
        """
        course1 = fts_db.create_course("Course 1", "C1")
        course2 = fts_db.create_course("Course 2", "C2")
        doc1 = fts_db.create_document(course1.id, "doc1.pdf", "pdf", 1000, "h1")
        doc2 = fts_db.create_document(course2.id, "doc2.pdf", "pdf", 1000, "h2")

        chunk = DocumentChunk.create(
            document_id=doc1.id,
            course_id=course1.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Original content with keyword Quicksort.",
            token_count=6,
        )
        fts_db.insert_chunks([chunk])
        assert len(fts_db.query_fts("Quicksort")) == 1

        # 1. Update content
        with fts_db.get_connection() as conn:
            conn.execute(
                "UPDATE chunks SET content = ? WHERE id = ?;",
                ("Modified content replacing sorting with Heapsort algorithm.", chunk.chunk_id),
            )

        # FTS5 row count remains 1
        assert fts_db.count_fts_rows() == 1
        # Old keyword no longer matches
        assert len(fts_db.query_fts("Quicksort")) == 0
        # New keyword matches
        new_res = fts_db.query_fts("Heapsort")
        assert len(new_res) == 1
        assert new_res[0]["chunk_id"] == chunk.chunk_id

        # 2. Update attribution (course_id and document_id)
        with fts_db.get_connection() as conn:
            conn.execute(
                "UPDATE chunks SET course_id = ?, document_id = ? WHERE id = ?;",
                (course2.id, doc2.id, chunk.chunk_id),
            )

        # Query filtered by course1 must yield 0, while course2 yields 1
        assert len(fts_db.query_fts("Heapsort", course_id=course1.id)) == 0
        res_c2 = fts_db.query_fts("Heapsort", course_id=course2.id)
        assert len(res_c2) == 1
        assert res_c2[0]["document_id"] == doc2.id

    def test_cascade_delete_from_document_cleans_fts5(self, fts_db):
        """Deleting a document cascades to chunks, firing trigger chunks_ad on all chunks."""
        course = fts_db.create_course("Architecture", "CS304")
        doc = fts_db.create_document(course.id, "pipeline.pdf", "pdf", 1000, "hp1")

        chunks = [
            DocumentChunk.create(
                document_id=doc.id,
                course_id=course.id,
                chunk_index=i,
                location_type="page",
                location_index=i + 1,
                content=f"Instruction pipelining hazard stage {i}: branch prediction and forwarding.",
                token_count=10,
            )
            for i in range(5)
        ]
        fts_db.insert_chunks(chunks)
        assert fts_db.count_fts_rows() == 5

        # Delete document via repository method
        assert fts_db.delete_document(doc.id) is True
        assert fts_db.count_fts_rows() == 0
        assert len(fts_db.query_fts("pipelining")) == 0

    def test_cascade_delete_from_course_cleans_fts5(self, fts_db):
        """Deleting a course cascades to documents and chunks, firing triggers on all chunks."""
        course = fts_db.create_course("Theory", "CS305")
        doc = fts_db.create_document(course.id, "automata.pdf", "pdf", 1000, "ha1")
        chunks = [
            DocumentChunk.create(
                document_id=doc.id,
                course_id=course.id,
                chunk_index=i,
                location_type="page",
                location_index=i + 1,
                content=f"Turing machine halting problem proof step {i}.",
                token_count=8,
            )
            for i in range(3)
        ]
        fts_db.insert_chunks(chunks)
        assert fts_db.count_fts_rows() == 3

        # Delete course
        assert fts_db.delete_course(course.id) is True
        assert fts_db.count_fts_rows() == 0
        assert len(fts_db.query_fts("Turing")) == 0

    def test_transaction_rollback_reverts_fts5_and_base_table(self, fts_db):
        """
        Interrupted transaction or explicit rollback atomically reverts both chunks
        and chunks_fts virtual entries, preventing partial or corrupted index state.
        """
        course = fts_db.create_course("Security", "CS306")
        doc = fts_db.create_document(course.id, "crypto.pdf", "pdf", 1000, "hc1")

        chunk = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Diffie-Hellman key exchange discrete logarithm problem.",
            token_count=8,
        )

        with pytest.raises(RuntimeError, match="Simulated crash"):
            with fts_db.transaction() as conn:
                fts_db.insert_chunks([chunk], conn=conn)
                # Ensure it was visible inside the uncommitted transaction
                row = conn.execute("SELECT count(*) as cnt FROM chunks_fts;").fetchone()
                assert row["cnt"] == 1
                raise RuntimeError("Simulated crash during ingestion")

        # After rollback, both base table and FTS table must have 0 rows
        assert len(fts_db.get_chunks_for_document(doc.id)) == 0
        assert fts_db.count_fts_rows() == 0
        assert len(fts_db.query_fts("Diffie-Hellman")) == 0
