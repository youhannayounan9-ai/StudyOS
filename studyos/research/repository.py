"""SQLite persistence repository for external research records and sources."""

from __future__ import annotations

import logging
import sqlite3
import uuid

from studyos.research.models import (
    ExternalSource,
    ResearchRecord,
    ResearchStatus,
    SourceType,
)
from studyos.storage.db import Database

logger = logging.getLogger(__name__)


class ResearchRepository:
    """Manages transactional persistence for external research queries and validated sources."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save_research_record(self, record: ResearchRecord) -> None:
        """Persist a research record and its associated sources in an atomic transaction."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO external_research_records (id, query, course_id, summary, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary = excluded.summary,
                    status = excluded.status,
                    error_message = excluded.error_message;
                """,
                (
                    record.id,
                    record.query,
                    record.course_id,
                    record.summary,
                    record.status.value,
                    record.error_message,
                ),
            )

            for src in record.sources:
                conn.execute(
                    """
                    INSERT INTO external_research_sources (
                        id, research_id, citation_index, url, title, publisher,
                        published_date, retrieved_at, snippet, source_type, is_valid
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(research_id, citation_index) DO UPDATE SET
                        url = excluded.url,
                        title = excluded.title,
                        publisher = excluded.publisher,
                        published_date = excluded.published_date,
                        snippet = excluded.snippet,
                        source_type = excluded.source_type,
                        is_valid = excluded.is_valid;
                    """,
                    (
                        src.id or f"src_{uuid.uuid4().hex[:12]}",
                        record.id,
                        src.citation_index,
                        src.url,
                        src.title,
                        src.publisher,
                        src.published_date,
                        src.retrieved_at,
                        src.snippet,
                        src.source_type.value,
                        1 if src.is_valid else 0,
                    ),
                )

    def get_research_record(self, record_id: str) -> ResearchRecord | None:
        """Retrieve a research record with all its sources by ID."""
        with self.db.get_connection() as conn:
            r_row = conn.execute(
                "SELECT * FROM external_research_records WHERE id = ?;",
                (record_id,),
            ).fetchone()
            if not r_row:
                return None

            s_rows = conn.execute(
                """
                SELECT * FROM external_research_sources
                WHERE research_id = ?
                ORDER BY citation_index ASC;
                """,
                (record_id,),
            ).fetchall()

            sources = [self._row_to_source(s) for s in s_rows]
            return ResearchRecord(
                id=r_row["id"],
                query=r_row["query"],
                course_id=r_row["course_id"],
                summary=r_row["summary"],
                status=ResearchStatus(r_row["status"]),
                error_message=r_row["error_message"],
                created_at=str(r_row["created_at"]) if r_row["created_at"] else None,
                sources=sources,
            )

    def list_research_records(
        self,
        course_id: str | None = None,
        limit: int = 20,
    ) -> list[ResearchRecord]:
        """List past research queries."""
        with self.db.get_connection() as conn:
            if course_id:
                rows = conn.execute(
                    """
                    SELECT * FROM external_research_records
                    WHERE course_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?;
                    """,
                    (course_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM external_research_records
                    ORDER BY created_at DESC
                    LIMIT ?;
                    """,
                    (limit,),
                ).fetchall()

            records = []
            for r in rows:
                s_rows = conn.execute(
                    "SELECT * FROM external_research_sources WHERE research_id = ? ORDER BY citation_index ASC;",
                    (r["id"],),
                ).fetchall()
                records.append(
                    ResearchRecord(
                        id=r["id"],
                        query=r["query"],
                        course_id=r["course_id"],
                        summary=r["summary"],
                        status=ResearchStatus(r["status"]),
                        error_message=r["error_message"],
                        created_at=str(r["created_at"]) if r["created_at"] else None,
                        sources=[self._row_to_source(s) for s in s_rows],
                    )
                )
            return records

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> ExternalSource:
        return ExternalSource(
            id=row["id"],
            research_id=row["research_id"],
            citation_index=int(row["citation_index"]),
            url=row["url"],
            title=row["title"],
            publisher=row["publisher"],
            published_date=row["published_date"],
            retrieved_at=str(row["retrieved_at"]) if row["retrieved_at"] else "",
            snippet=row["snippet"] or "",
            source_type=SourceType(row["source_type"]),
            is_valid=bool(row["is_valid"]),
        )
