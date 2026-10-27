"""Unit tests for SQLite storage layer and CRUD operations."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pytest

from studyos.ingestion.models import DocumentChunk
from studyos.storage.db import Database


@pytest.fixture
def mem_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    yield db
    db.close()


class TestDatabaseInitAndMigrations:
    def test_migrations_applied_and_idempotent(self, mem_db):
        # Migrations already ran in fixture
        with mem_db.get_connection() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                ).fetchall()
            ]
            assert "courses" in tables
            assert "documents" in tables
            assert "chunks" in tables
            assert "chunks_fts" in tables
            assert "schema_migrations" in tables

        # Re-running migrate() should be safe and idempotent
        mem_db.migrate()

    def test_file_database_wal_mode_and_persistence(self, tmp_path):
        db_file = tmp_path / "persistent.db"
        db = Database(db_file)
        db.migrate()

        with db.get_connection() as conn:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() == "wal"

        # Create a course and document
        course = db.create_course("Operating Systems", "CS201")
        doc = db.create_document(
            course_id=course.id,
            filename="lec1.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            sha256_hash="hash_os_1",
        )

        # Reconnect to the same file
        db2 = Database(db_file)
        fetched_course = db2.get_course(course.id)
        assert fetched_course is not None
        assert fetched_course.name == "Operating Systems"
        assert fetched_course.code == "CS201"

        fetched_doc = db2.get_document(doc.id)
        assert fetched_doc is not None
        assert fetched_doc.filename == "lec1.pdf"

    def test_upgrade_from_0001_repairs_inconsistent_chunks_and_syncs_fts(self, tmp_path):
        """Upgrading a database from migration 0001 to 0002 must repair mismatched chunks and keep FTS5 synchronized."""
        db_file = tmp_path / "upgrade_test.db"
        db = Database(db_file)

        # 1. Initialize schema with ONLY migration 0001
        migrations_dir = Path(__file__).resolve().parent.parent.parent / "studyos" / "storage" / "migrations"
        m1_sql = (migrations_dir / "0001_initial.sql").read_text(encoding="utf-8")
        with db.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.executescript(m1_sql)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (1);")

        # 2. Insert Course A and Course B
        course_a = db.create_course("Operating Systems", "CS301")
        course_b = db.create_course("Computer Networks", "CS302")

        # 3. Insert Document belonging to Course A
        doc_a = db.create_document(
            course_id=course_a.id,
            filename="kernel_scheduling.pdf",
            file_type="pdf",
            file_size_bytes=4096,
            sha256_hash="hash_kernel_123",
        )

        # 4. Insert an inconsistent chunk: belongs to course_b but references doc_a (Course A)
        inconsistent_chunk = DocumentChunk.create(
            document_id=doc_a.id,
            course_id=course_b.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Preemptive process scheduling and priority queues in kernel.",
            token_count=9,
        )
        # Also insert a consistent chunk to ensure it remains untouched
        consistent_chunk = DocumentChunk.create(
            document_id=doc_a.id,
            course_id=course_a.id,
            chunk_index=1,
            location_type="page",
            location_index=2,
            content="Virtual memory paging and TLB caching mechanism.",
            token_count=8,
        )
        db.insert_chunks([inconsistent_chunk, consistent_chunk])

        # 5. Confirm that the inconsistent chunk is indeed present before upgrading
        pre_chunks = db.get_chunks_for_document(doc_a.id)
        assert len(pre_chunks) == 2
        chk_map_pre = {c.chunk_id: c for c in pre_chunks}
        assert chk_map_pre[inconsistent_chunk.chunk_id].course_id == course_b.id
        assert chk_map_pre[inconsistent_chunk.chunk_id].course_id != doc_a.course_id
        assert chk_map_pre[consistent_chunk.chunk_id].course_id == course_a.id

        # In chunks_fts before upgrade, inconsistent chunk is indexed under course_b
        fts_b_pre = db.query_fts("scheduling", course_id=course_b.id)
        assert len(fts_b_pre) == 1
        assert fts_b_pre[0]["chunk_id"] == inconsistent_chunk.chunk_id
        assert fts_b_pre[0]["course_id"] == course_b.id

        fts_a_pre = db.query_fts("scheduling", course_id=course_a.id)
        assert len(fts_a_pre) == 0

        # Total FTS rows before upgrade is 2
        assert db.count_fts_rows() == 2

        # 6. Apply migration 0002 through the project's actual migration runner
        db.migrate()

        # Verify schema_migrations now records versions 1 and 2
        with db.get_connection() as conn:
            versions = [
                r["version"]
                for r in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version ASC;"
                ).fetchall()
            ]
            assert versions == [1, 2]

        # 7. Confirm the database no longer contains the inconsistency
        post_chunks = db.get_chunks_for_document(doc_a.id)
        assert len(post_chunks) == 2
        chk_map_post = {c.chunk_id: c for c in post_chunks}
        assert chk_map_post[inconsistent_chunk.chunk_id].course_id == course_a.id
        assert chk_map_post[inconsistent_chunk.chunk_id].course_id == doc_a.course_id
        assert chk_map_post[consistent_chunk.chunk_id].course_id == course_a.id

        # 8. Verify chunks_fts entry is synchronized and no orphaned rows exist
        fts_a_post = db.query_fts("scheduling", course_id=course_a.id)
        assert len(fts_a_post) == 1
        assert fts_a_post[0]["chunk_id"] == inconsistent_chunk.chunk_id
        assert fts_a_post[0]["course_id"] == course_a.id

        fts_b_post = db.query_fts("scheduling", course_id=course_b.id)
        assert len(fts_b_post) == 0

        assert db.count_fts_rows() == 2

        # 9. Verify the newly installed triggers protect against future violations
        new_bad_chunk = DocumentChunk.create(
            document_id=doc_a.id,
            course_id=course_b.id,
            chunk_index=2,
            location_type="page",
            location_index=3,
            content="Attempted future violation.",
            token_count=4,
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="Chunk course_id does not match parent document course_id",
        ):
            db.insert_chunks([new_bad_chunk])


class TestCourseOperations:
    def test_course_crud(self, mem_db):
        # Create
        c = mem_db.create_course("Computer Networks", "CS302", "Networking fundamentals")
        assert c.id.startswith("crs_")
        assert c.name == "Computer Networks"
        assert c.code == "CS302"

        # Get by id
        fetched = mem_db.get_course(c.id)
        assert fetched is not None
        assert fetched.name == "Computer Networks"

        # Get by code
        by_code = mem_db.get_course_by_code("CS302")
        assert by_code is not None
        assert by_code.id == c.id

        # List
        all_courses = mem_db.list_courses()
        assert len(all_courses) == 1
        assert all_courses[0].id == c.id

        # Delete
        assert mem_db.delete_course(c.id) is True
        assert mem_db.get_course(c.id) is None


class TestDocumentAndChunkOperations:
    def test_document_crud_and_unique_hash_constraint(self, mem_db):
        course = mem_db.create_course("Databases", "CS303")

        doc = mem_db.create_document(
            course_id=course.id,
            filename="db_intro.pdf",
            file_type="pdf",
            file_size_bytes=2048,
            sha256_hash="hash_db_intro",
            status="ready",
        )
        assert doc.id.startswith("doc_")

        # Get by hash
        by_hash = mem_db.get_document_by_hash(course.id, "hash_db_intro")
        assert by_hash is not None
        assert by_hash.id == doc.id

        # Unique constraint on (course_id, sha256_hash)
        with pytest.raises(sqlite3.IntegrityError):
            mem_db.create_document(
                course_id=course.id,
                filename="duplicate.pdf",
                file_type="pdf",
                file_size_bytes=2048,
                sha256_hash="hash_db_intro",
            )

    def test_foreign_key_enforcement(self, mem_db):
        """Inserting a document or chunk with non-existent foreign keys must fail."""
        with pytest.raises(sqlite3.IntegrityError):
            mem_db.create_document(
                course_id="crs_nonexistent",
                filename="test.pdf",
                file_type="pdf",
                file_size_bytes=100,
                sha256_hash="hash_orphan",
            )

        course = mem_db.create_course("AI", "CS401")
        orphan_chunk = DocumentChunk.create(
            document_id="doc_nonexistent",
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Orphan chunk",
            token_count=2,
        )
        with pytest.raises(sqlite3.IntegrityError):
            mem_db.insert_chunks([orphan_chunk])

    def test_cascade_deletion(self, mem_db):
        """Deleting a course cascades deletion to its documents and chunks."""
        course = mem_db.create_course("Data Structures", "CS202")
        doc = mem_db.create_document(
            course_id=course.id,
            filename="trees.pdf",
            file_type="pdf",
            file_size_bytes=5000,
            sha256_hash="hash_trees",
        )

        chk1 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="AVL Trees",
            token_count=2,
        )
        chk2 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=1,
            location_type="page",
            location_index=2,
            content="Red-Black Trees",
            token_count=3,
        )
        mem_db.insert_chunks([chk1, chk2])

        assert len(mem_db.get_chunks_for_document(doc.id)) == 2

        # Delete course
        assert mem_db.delete_course(course.id) is True
        assert mem_db.get_document(doc.id) is None
        assert len(mem_db.get_chunks_for_document(doc.id)) == 0

    def test_chunk_course_id_must_match_parent_document_course_id_on_insert(self, mem_db):
        """Inserting a chunk with a course_id different from its parent document must fail."""
        course_a = mem_db.create_course("Course A", "CA101")
        course_b = mem_db.create_course("Course B", "CB201")

        doc_a = mem_db.create_document(
            course_id=course_a.id,
            filename="doc_a.pdf",
            file_type="pdf",
            file_size_bytes=1000,
            sha256_hash="hash_doc_a",
        )

        # Attempt to insert a chunk pointing to doc_a (Course A) but with course_b (Course B)
        mismatched_chunk = DocumentChunk.create(
            document_id=doc_a.id,
            course_id=course_b.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Mismatched Course Chunk",
            token_count=3,
        )

        with pytest.raises(sqlite3.IntegrityError, match="Chunk course_id does not match parent document course_id"):
            mem_db.insert_chunks([mismatched_chunk])

        # Verify no orphaned/mismatched chunks exist
        assert len(mem_db.get_chunks_for_document(doc_a.id)) == 0

    def test_raw_sql_insert_and_updates_enforce_course_id_invariant(self, mem_db):
        """Database triggers must reject mismatched course_id even when invoked via raw SQL outside service methods."""
        course_a = mem_db.create_course("Course A", "CA102")
        course_b = mem_db.create_course("Course B", "CB202")

        doc_a = mem_db.create_document(
            course_id=course_a.id,
            filename="doc_a.pdf",
            file_type="pdf",
            file_size_bytes=1000,
            sha256_hash="hash_doc_a_raw",
        )
        doc_b = mem_db.create_document(
            course_id=course_b.id,
            filename="doc_b.pdf",
            file_type="pdf",
            file_size_bytes=1000,
            sha256_hash="hash_doc_b_raw",
        )

        with mem_db.get_connection() as conn:
            # 1. Raw SQL INSERT with mismatched course_id
            with pytest.raises(sqlite3.IntegrityError, match="Chunk course_id does not match parent document course_id"):
                conn.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, course_id, chunk_index, location_type,
                        location_index, location_title, content, token_count, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    ("chk_raw_bad", doc_a.id, course_b.id, 0, "page", 1, None, "Raw bad", 2, "hash_raw"),
                )

            # Insert a valid chunk
            conn.execute(
                """
                INSERT INTO chunks (
                    id, document_id, course_id, chunk_index, location_type,
                    location_index, location_title, content, token_count, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                ("chk_raw_good", doc_a.id, course_a.id, 0, "page", 1, None, "Raw good", 2, "hash_good"),
            )

            # 2. Raw SQL UPDATE of chunk's course_id to mismatched course_b
            with pytest.raises(sqlite3.IntegrityError, match="Chunk course_id does not match parent document course_id"):
                conn.execute("UPDATE chunks SET course_id = ? WHERE id = ?;", (course_b.id, "chk_raw_good"))

            # 3. Raw SQL UPDATE of chunk's document_id to doc_b (which belongs to course_b, mismatched with course_a)
            with pytest.raises(sqlite3.IntegrityError, match="Chunk course_id does not match parent document course_id"):
                conn.execute("UPDATE chunks SET document_id = ? WHERE id = ?;", (doc_b.id, "chk_raw_good"))

            # 4. Raw SQL UPDATE of doc_a's course_id to course_b when child chunk exists with course_a
            with pytest.raises(sqlite3.IntegrityError, match="Cannot update document course_id"):
                conn.execute("UPDATE documents SET course_id = ? WHERE id = ?;", (course_b.id, doc_a.id))
