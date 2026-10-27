"""Authoritative lexical retrieval service using SQLite FTS5 BM25 scoring."""

from __future__ import annotations

import re
from typing import Any

from studyos.retrieval.models import RetrievedChunk
from studyos.storage.db import Database


class LexicalSearchService:
    """Performs safe, filtered full-text lexical search using SQLite FTS5."""

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def sanitize_query(query: str) -> str | None:
        """Sanitize raw query string into a safe, valid FTS5 MATCH expression.

        Extracts unicode alphanumeric tokens and formats them as quoted phrases.
        Returns None if no searchable tokens are present.
        """
        if not query or not query.strip():
            return None

        # Extract alphanumeric words (unicode-aware)
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return None

        # Format each token safely in double quotes to avoid FTS5 syntax errors
        return " ".join(f'"{token}"' for token in tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        course_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform filtered BM25 lexical search joined to authoritative SQLite records."""
        if top_k <= 0:
            return []

        # Explicit empty filter lists match nothing
        if (course_ids is not None and len(course_ids) == 0) or (
            document_ids is not None and len(document_ids) == 0
        ):
            return []

        safe_fts_query = self.sanitize_query(query)
        if safe_fts_query is None:
            return []

        # Construct parameterized SQL query with pre-filtering
        sql_parts = [
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.course_id,
                crs.name AS course_name,
                d.filename,
                d.file_type,
                c.chunk_index,
                c.location_type,
                c.location_index,
                c.location_title,
                c.content,
                c.token_count,
                c.content_hash,
                bm25(chunks_fts) AS lexical_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            JOIN documents d ON d.id = c.document_id
            JOIN courses crs ON crs.id = c.course_id
            WHERE chunks_fts MATCH ?
            """
        ]
        params: list[Any] = [safe_fts_query]

        if course_ids:
            placeholders = ",".join("?" for _ in course_ids)
            sql_parts.append(f"AND c.course_id IN ({placeholders})")
            params.extend(course_ids)

        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            sql_parts.append(f"AND c.document_id IN ({placeholders})")
            params.extend(document_ids)

        sql_parts.append("ORDER BY bm25(chunks_fts) ASC LIMIT ?;")
        params.append(top_k)

        full_sql = "\n".join(sql_parts)

        with self.db.get_connection() as conn:
            rows = conn.execute(full_sql, params).fetchall()

        results: list[RetrievedChunk] = []
        for idx, row in enumerate(rows, start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    course_id=row["course_id"],
                    course_name=row["course_name"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    chunk_index=row["chunk_index"],
                    location_type=row["location_type"],
                    location_index=row["location_index"],
                    location_title=row["location_title"],
                    content=row["content"],
                    token_count=row["token_count"],
                    content_hash=row["content_hash"],
                    score=float(row["lexical_score"]),
                    lexical_rank=idx,
                    sources=["lexical"],
                )
            )

        return results
