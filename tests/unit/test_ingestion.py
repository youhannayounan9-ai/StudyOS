"""Unit tests for IngestionService orchestration and business logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sqlite3
import pytest

from studyos.core.config import AppConfig
from studyos.ingestion.models import (
    CorruptFileError,
    IngestionError,
    UnsupportedFileTypeError,
)
from studyos.ingestion.service import IngestionService
from studyos.storage.db import Database
from tests.fixtures.synthetic.build_fixtures import generate_all_fixtures


@pytest.fixture
def service_setup(tmp_path):
    fixtures = generate_all_fixtures(tmp_path / "fixtures")
    db = Database(":memory:")
    db.migrate()
    config = AppConfig()
    service = IngestionService(db=db, config=config)
    course = db.create_course("Computer Science 101", "CS101")
    return service, db, course, fixtures


class TestIngestionService:
    def test_ingest_synthetic_pdf_success(self, service_setup):
        service, db, course, fixtures = service_setup
        result = service.ingest_file(fixtures["pdf"], course_id=course.id)

        assert result.status == "ready"
        assert result.element_count == 2
        assert result.chunk_count == 2
        assert result.document_id.startswith("doc_")

        # Verify database records
        doc = db.get_document(result.document_id)
        assert doc is not None
        assert doc.status == "ready"
        assert doc.total_pages_or_slides == 2
        assert doc.chunk_count == 2

        # Verify chunks persisted and indexed in FTS5
        chunks = db.get_chunks_for_document(result.document_id)
        assert len(chunks) == 2
        assert db.count_fts_rows() == 2

    def test_duplicate_detection_skips_reprocessing(self, service_setup):
        service, db, course, fixtures = service_setup
        pdf_path = fixtures["pdf"]

        # First ingestion
        r1 = service.ingest_file(pdf_path, course_id=course.id)
        assert r1.status == "ready"
        assert db.count_fts_rows() == 2

        # Second ingestion without force_reindex
        r2 = service.ingest_file(pdf_path, course_id=course.id, force_reindex=False)
        assert r2.status == "duplicate"
        assert r2.document_id == r1.document_id
        assert len(r2.warnings) == 1
        assert "already exists" in r2.warnings[0]

        # FTS5 row count must still be exactly 2 (no duplicate rows created)
        assert db.count_fts_rows() == 2

    def test_force_reindex_replaces_cleanly(self, service_setup):
        service, db, course, fixtures = service_setup
        pdf_path = fixtures["pdf"]

        r1 = service.ingest_file(pdf_path, course_id=course.id)
        assert r1.status == "ready"
        old_doc_id = r1.document_id

        # Re-index with force_reindex=True
        r2 = service.ingest_file(pdf_path, course_id=course.id, force_reindex=True)
        assert r2.status == "ready"
        assert r2.document_id != old_doc_id

        # Old document must be deleted, new document present
        assert db.get_document(old_doc_id) is None
        assert db.get_document(r2.document_id) is not None
        # FTS5 row count remains 2 (old deleted via trigger, new inserted via trigger)
        assert db.count_fts_rows() == 2

    def test_ingest_nonexistent_file_raises(self, service_setup):
        service, db, course, _ = service_setup
        with pytest.raises(FileNotFoundError):
            service.ingest_file("does_not_exist.pdf", course_id=course.id)

    def test_ingest_nonexistent_course_raises(self, service_setup):
        service, db, _, fixtures = service_setup
        with pytest.raises(IngestionError, match="Course not found"):
            service.ingest_file(fixtures["pdf"], course_id="crs_phantom")

    def test_atomic_rollback_on_persistence_failure(self, service_setup):
        """If an error occurs while persisting chunks, the transaction rolls back cleanly."""
        service, db, course, fixtures = service_setup

        with patch.object(db, "insert_chunks", side_effect=sqlite3.OperationalError("Disk I/O error")):
            with pytest.raises(sqlite3.OperationalError, match="Disk I/O error"):
                service.ingest_file(fixtures["pdf"], course_id=course.id)

        # Confirm zero orphaned documents, chunks, or FTS5 rows left behind
        assert len(db.list_documents(course.id)) == 0
        assert db.count_fts_rows() == 0

    def test_force_reindex_parse_failure_preserves_existing_document(self, service_setup):
        """
        Test A: When force-reindexing, if parsing fails, the previous valid document,
        chunks, and FTS5 records must remain completely intact.
        """
        service, db, course, fixtures = service_setup
        pdf_path = fixtures["pdf"]

        # 1. First ingest a valid document
        r1 = service.ingest_file(pdf_path, course_id=course.id)
        assert r1.status == "ready"
        old_doc_id = r1.document_id
        assert db.count_fts_rows() == 2

        # 2. Attempt force-reindex using a parser that raises a controlled parsing exception
        from studyos.ingestion.parsers import PDFParser

        with patch.object(PDFParser, "parse", side_effect=CorruptFileError("Simulated corrupted parser stream")):
            with pytest.raises(CorruptFileError, match="Simulated corrupted parser stream"):
                service.ingest_file(pdf_path, course_id=course.id, force_reindex=True)

        # 3. Verify that existing document, chunks, and FTS5 records remain intact
        existing_doc = db.get_document(old_doc_id)
        assert existing_doc is not None
        assert existing_doc.status == "ready"
        assert existing_doc.chunk_count == 2
        chunks = db.get_chunks_for_document(old_doc_id)
        assert len(chunks) == 2
        assert db.count_fts_rows() == 2
        assert len(db.list_documents(course.id)) == 1

    def test_force_reindex_persistence_failure_rolls_back_and_preserves_original(self, service_setup):
        """
        Test B: When force-reindexing, if a failure occurs after the previous document would
        otherwise have been deleted, verify the transaction rolls back, restores the original
        document, chunks, and FTS5 records, and leaves no partial replacement rows.
        """
        service, db, course, fixtures = service_setup
        pdf_path = fixtures["pdf"]

        # 1. Ingest valid document
        r1 = service.ingest_file(pdf_path, course_id=course.id)
        assert r1.status == "ready"
        old_doc_id = r1.document_id
        assert db.count_fts_rows() == 2

        # 2. Inject controlled failure inside the persistence block (e.g., when inserting chunks)
        with patch.object(db, "insert_chunks", side_effect=sqlite3.OperationalError("Simulated disk I/O failure")):
            with pytest.raises(sqlite3.OperationalError, match="Simulated disk I/O failure"):
                service.ingest_file(pdf_path, course_id=course.id, force_reindex=True)

        # 3. Verify rollback: original document, chunks, and FTS5 records restored
        doc = db.get_document(old_doc_id)
        assert doc is not None
        assert doc.status == "ready"
        chunks = db.get_chunks_for_document(old_doc_id)
        assert len(chunks) == 2
        assert db.count_fts_rows() == 2
        # Verify no partial replacement rows remain
        docs = db.list_documents(course.id)
        assert len(docs) == 1
        assert docs[0].id == old_doc_id

    def test_force_reindex_successful_replacement_single_document_and_fts(self, service_setup):
        """
        Test C: Successful force-reindex leaves exactly one current document for the
        intended course and hash, its chunks and FTS5 entries correspond to the replacement,
        and no stale entries remain.
        """
        service, db, course, fixtures = service_setup
        pdf_path = fixtures["pdf"]

        # 1. Initial ingestion
        r1 = service.ingest_file(pdf_path, course_id=course.id)
        assert r1.status == "ready"
        old_doc_id = r1.document_id
        old_chunks = db.get_chunks_for_document(old_doc_id)
        old_chunk_ids = {c.chunk_id for c in old_chunks}

        # 2. Force re-index
        r2 = service.ingest_file(pdf_path, course_id=course.id, force_reindex=True)
        assert r2.status == "ready"
        assert r2.document_id != old_doc_id

        # 3. Exactly one current document for course and hash
        docs = db.list_documents(course.id)
        assert len(docs) == 1
        assert docs[0].id == r2.document_id
        assert docs[0].sha256_hash == r2.sha256_hash

        # 4. Replacement chunks and FTS5 entries correspond to replacement
        new_chunks = db.get_chunks_for_document(r2.document_id)
        assert len(new_chunks) == 2
        new_chunk_ids = {c.chunk_id for c in new_chunks}
        assert old_chunk_ids.isdisjoint(new_chunk_ids)

        # FTS5 entries
        assert db.count_fts_rows() == 2
        with db.get_connection() as conn:
            rows = conn.execute("SELECT chunk_id, document_id FROM chunks_fts;").fetchall()
            fts_chunk_ids = {r["chunk_id"] for r in rows}
            fts_doc_ids = {r["document_id"] for r in rows}
            assert fts_chunk_ids == new_chunk_ids
            assert fts_doc_ids == {r2.document_id}
            # No stale entries remain
            assert old_doc_id not in fts_doc_ids
