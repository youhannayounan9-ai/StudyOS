"""SQLite storage layer, connection management, migrations, and CRUD operations."""

from __future__ import annotations

import contextlib
from pathlib import Path
import sqlite3
import uuid
from typing import Generator

from studyos.ingestion.models import Course, Document, DocumentChunk


class Database:
    """Manages SQLite connections, schema migrations, and relational persistence."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path) if isinstance(db_path, Path) else db_path
        self._mem_conn: sqlite3.Connection | None = None
        self._ensure_parent_dir()
        self._verify_fts5_support()

    def _ensure_parent_dir() -> None:
        pass

    def _ensure_parent_dir(self) -> None:
        if self.db_path != ":memory:":
            p = Path(self.db_path)
            p.parent.mkdir(parents=True, exist_ok=True)

    def _verify_fts5_support(self) -> None:
        """Verify that the current SQLite build includes the FTS5 extension."""
        with sqlite3.connect(":memory:") as test_conn:
            try:
                test_conn.execute("CREATE VIRTUAL TABLE temp._fts5_check USING fts5(x);")
                test_conn.execute("DROP TABLE temp._fts5_check;")
            except sqlite3.OperationalError as e:
                raise RuntimeError(
                    f"SQLite FTS5 extension is not available in this Python build: {e}"
                ) from e

    def connect(self) -> sqlite3.Connection:
        """Create or retrieve a configured SQLite connection with foreign keys and WAL mode."""
        if self.db_path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
                self._mem_conn.execute("PRAGMA foreign_keys = ON;")
            return self._mem_conn

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        # Enable WAL mode for high concurrency and crash durability on disk
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @contextlib.contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an active connection with transaction handling."""
        if self.db_path == ":memory:":
            conn = self.connect()
            with conn:
                yield conn
        else:
            conn = self.connect()
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an atomic transaction block."""
        with self.get_connection() as conn:
            yield conn

    def close(self) -> None:
        """Close persistent in-memory connection if active."""
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None

    def migrate(self) -> None:
        """Apply all pending SQL migrations in deterministic numerical order.

        Each migration is executed inside an explicit transaction block. If an error occurs
        during migration execution, the transaction is immediately rolled back, ensuring
        zero partially applied schemas or database corruption.
        """
        migrations_dir = Path(__file__).parent / "migrations"
        if not migrations_dir.is_dir():
            return

        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            applied_rows = conn.execute("SELECT version FROM schema_migrations;").fetchall()
            applied_versions = {row["version"] for row in applied_rows}

            migration_files = sorted(migrations_dir.glob("*.sql"))
            for mf in migration_files:
                try:
                    version = int(mf.stem.split("_")[0])
                except (ValueError, IndexError):
                    continue

                if version not in applied_versions:
                    sql_script = mf.read_text(encoding="utf-8")
                    tx_script = (
                        f"BEGIN TRANSACTION;\n"
                        f"{sql_script}\n"
                        f"INSERT INTO schema_migrations (version) VALUES ({version});\n"
                        f"COMMIT;\n"
                    )
                    try:
                        conn.executescript(tx_script)
                    except Exception as e:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"Migration {version:04d} ({mf.name}) failed and was rolled back: {e}"
                        ) from e

    # -----------------------------------------------------------------------
    # Course Operations
    # -----------------------------------------------------------------------

    def create_course(
        self,
        name: str,
        code: str | None = None,
        description: str | None = None,
        course_id: str | None = None,
    ) -> Course:
        cid = course_id or f"crs_{uuid.uuid4().hex[:12]}"
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO courses (id, name, code, description)
                VALUES (?, ?, ?, ?);
                """,
                (cid, name, code, description),
            )
        return Course(id=cid, name=name, code=code, description=description)

    def get_course(self, course_id: str) -> Course | None:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT id, name, code, description, created_at, updated_at FROM courses WHERE id = ?;",
                (course_id,),
            ).fetchone()
            if not row:
                return None
            return Course(
                id=row["id"],
                name=row["name"],
                code=row["code"],
                description=row["description"],
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )

    def get_course_by_code(self, code: str) -> Course | None:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT id, name, code, description, created_at, updated_at FROM courses WHERE code = ?;",
                (code,),
            ).fetchone()
            if not row:
                return None
            return Course(
                id=row["id"],
                name=row["name"],
                code=row["code"],
                description=row["description"],
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )

    def list_courses(self) -> list[Course]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, name, code, description, created_at, updated_at FROM courses ORDER BY name ASC;"
            ).fetchall()
            return [
                Course(
                    id=r["id"],
                    name=r["name"],
                    code=r["code"],
                    description=r["description"],
                    created_at=str(r["created_at"]),
                    updated_at=str(r["updated_at"]),
                )
                for r in rows
            ]

    def delete_course(self, course_id: str) -> bool:
        """Delete course. Triggers CASCADE delete on documents and chunks, firing FTS5 triggers."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM courses WHERE id = ?;", (course_id,))
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Document Operations
    # -----------------------------------------------------------------------

    def create_document(
        self,
        course_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        sha256_hash: str,
        status: str = "pending",
        document_id: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> Document:
        did = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        query = """
            INSERT INTO documents (
                id, course_id, filename, file_type, file_size_bytes, sha256_hash, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        params = (did, course_id, filename, file_type, file_size_bytes, sha256_hash, status)
        if conn is not None:
            conn.execute(query, params)
        else:
            with self.get_connection() as c:
                c.execute(query, params)

        return Document(
            id=did,
            course_id=course_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            sha256_hash=sha256_hash,
            status=status,
            total_pages_or_slides=0,
            chunk_count=0,
        )

    def update_document_status(
        self,
        document_id: str,
        status: str,
        total_pages_or_slides: int = 0,
        chunk_count: int = 0,
        error_message: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        query = """
            UPDATE documents
            SET status = ?, total_pages_or_slides = ?, chunk_count = ?, error_message = ?
            WHERE id = ?;
        """
        params = (status, total_pages_or_slides, chunk_count, error_message, document_id)
        if conn is not None:
            conn.execute(query, params)
        else:
            with self.get_connection() as c:
                c.execute(query, params)

    def get_document(self, document_id: str) -> Document | None:
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, course_id, filename, file_type, file_size_bytes, sha256_hash,
                       status, total_pages_or_slides, chunk_count, error_message, created_at
                FROM documents WHERE id = ?;
                """,
                (document_id,),
            ).fetchone()
            if not row:
                return None
            return Document(
                id=row["id"],
                course_id=row["course_id"],
                filename=row["filename"],
                file_type=row["file_type"],
                file_size_bytes=row["file_size_bytes"],
                sha256_hash=row["sha256_hash"],
                status=row["status"],
                total_pages_or_slides=row["total_pages_or_slides"],
                chunk_count=row["chunk_count"],
                error_message=row["error_message"],
                created_at=str(row["created_at"]),
            )

    def get_document_by_hash(self, course_id: str, sha256_hash: str) -> Document | None:
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, course_id, filename, file_type, file_size_bytes, sha256_hash,
                       status, total_pages_or_slides, chunk_count, error_message, created_at
                FROM documents WHERE course_id = ? AND sha256_hash = ?;
                """,
                (course_id, sha256_hash),
            ).fetchone()
            if not row:
                return None
            return Document(
                id=row["id"],
                course_id=row["course_id"],
                filename=row["filename"],
                file_type=row["file_type"],
                file_size_bytes=row["file_size_bytes"],
                sha256_hash=row["sha256_hash"],
                status=row["status"],
                total_pages_or_slides=row["total_pages_or_slides"],
                chunk_count=row["chunk_count"],
                error_message=row["error_message"],
                created_at=str(row["created_at"]),
            )

    def list_documents(self, course_id: str) -> list[Document]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, course_id, filename, file_type, file_size_bytes, sha256_hash,
                       status, total_pages_or_slides, chunk_count, error_message, created_at
                FROM documents WHERE course_id = ? ORDER BY created_at ASC;
                """,
                (course_id,),
            ).fetchall()
            return [
                Document(
                    id=r["id"],
                    course_id=r["course_id"],
                    filename=r["filename"],
                    file_type=r["file_type"],
                    file_size_bytes=r["file_size_bytes"],
                    sha256_hash=r["sha256_hash"],
                    status=r["status"],
                    total_pages_or_slides=r["total_pages_or_slides"],
                    chunk_count=r["chunk_count"],
                    error_message=r["error_message"],
                    created_at=str(r["created_at"]),
                )
                for r in rows
            ]

    def delete_document(
        self,
        document_id: str,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        """Delete document. Triggers CASCADE delete on chunks, which fires chunks_ad FTS5 trigger."""
        query = "DELETE FROM documents WHERE id = ?;"
        if conn is not None:
            cursor = conn.execute(query, (document_id,))
            return cursor.rowcount > 0
        with self.get_connection() as c:
            cursor = c.execute(query, (document_id,))
            return cursor.rowcount > 0

    # -----------------------------------------------------------------------
    # Chunk Operations
    # -----------------------------------------------------------------------

    def insert_chunks(
        self,
        chunks: list[DocumentChunk],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Insert chunks into the chunks table. Triggers automatically populate chunks_fts."""
        if not chunks:
            return

        query = """
            INSERT INTO chunks (
                id, document_id, course_id, chunk_index, location_type,
                location_index, location_title, content, token_count, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        rows = [
            (
                c.chunk_id,
                c.document_id,
                c.course_id,
                c.chunk_index,
                c.location_type,
                c.location_index,
                c.location_title,
                c.content,
                c.token_count,
                c.content_hash,
            )
            for c in chunks
        ]

        if conn is not None:
            conn.executemany(query, rows)
        else:
            with self.get_connection() as c:
                c.executemany(query, rows)

    def get_chunks_for_document(self, document_id: str) -> list[DocumentChunk]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, course_id, chunk_index, location_type,
                       location_index, location_title, content, token_count, content_hash
                FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC;
                """,
                (document_id,),
            ).fetchall()
            return [
                DocumentChunk(
                    chunk_id=r["id"],
                    document_id=r["document_id"],
                    course_id=r["course_id"],
                    chunk_index=r["chunk_index"],
                    location_type=r["location_type"],
                    location_index=r["location_index"],
                    location_title=r["location_title"],
                    content=r["content"],
                    token_count=r["token_count"],
                    content_hash=r["content_hash"],
                )
                for r in rows
            ]

    # -----------------------------------------------------------------------
    # FTS5 Query Operations (Verification & Testing)
    # -----------------------------------------------------------------------

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize raw user query into safe FTS5 token phrase format."""
        import re
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return '""'
        return " ".join(f'"{t}"' for t in tokens)

    def query_fts(
        self,
        query: str,
        course_id: str | None = None,
        document_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Perform BM25 full-text keyword search via chunks_fts."""
        safe_query = self._sanitize_fts_query(query)
        sql = """
            SELECT chunk_id, document_id, course_id, content, bm25(chunks_fts) as rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
        """
        params: list[str | int] = [safe_query]

        if course_id:
            sql += " AND course_id = ?"
            params.append(course_id)

        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            sql += f" AND document_id IN ({placeholders})"
            params.extend(document_ids)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def count_fts_rows(self) -> int:
        """Count total rows currently indexed in chunks_fts."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT count(*) as cnt FROM chunks_fts;").fetchone()
            return row["cnt"] if row else 0
