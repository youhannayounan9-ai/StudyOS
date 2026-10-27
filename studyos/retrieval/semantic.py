"""Authoritative semantic retrieval service using ChromaDB vectors and SQLite validation."""

from __future__ import annotations

from typing import Any

from studyos.retrieval.embedding import EmbeddingProvider
from studyos.retrieval.models import RetrievedChunk
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


class SemanticSearchService:
    """Performs dense vector semantic search backed by SQLite authoritative metadata."""

    def __init__(
        self,
        db: Database,
        vector_store: ChromaVectorStore,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self.db = db
        self.vector_store = vector_store
        self.provider = provider or vector_store.provider

    def search(
        self,
        query: str,
        top_k: int = 10,
        course_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform semantic search, validating results and resolving all metadata via SQLite."""
        if top_k <= 0 or not query or not query.strip():
            return []

        # Explicit empty filter lists match nothing
        if (course_ids is not None and len(course_ids) == 0) or (
            document_ids is not None and len(document_ids) == 0
        ):
            return []

        if self.provider is None:
            raise RuntimeError("Cannot execute semantic search without an EmbeddingProvider.")

        # 1. Embed query
        query_vector = self.provider.embed_query(query)

        # 2. Query candidates from ChromaDB
        # Fetch up to top_k * 2 candidates to compensate for any stale vector records
        fetch_k = max(top_k * 2, 20)
        candidates = self.vector_store.search_candidates(
            query_embedding=query_vector,
            top_k=fetch_k,
            course_ids=course_ids,
            document_ids=document_ids,
        )

        if not candidates:
            return []

        # Map chunk_id to (semantic_rank, distance)
        chunk_ranks: dict[str, tuple[int, float]] = {
            cid: (rank, dist) for rank, (cid, dist) in enumerate(candidates, start=1)
        }
        candidate_ids = list(chunk_ranks.keys())

        # 3. Authoritative SQLite Validation and Metadata Enrichment
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
                c.content_hash
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN courses crs ON crs.id = c.course_id
            WHERE c.id IN ({})
            """.format(",".join("?" for _ in candidate_ids))
        ]
        params: list[Any] = list(candidate_ids)

        if course_ids:
            placeholders = ",".join("?" for _ in course_ids)
            sql_parts.append(f"AND c.course_id IN ({placeholders})")
            params.extend(course_ids)

        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            sql_parts.append(f"AND c.document_id IN ({placeholders})")
            params.extend(document_ids)

        full_sql = "\n".join(sql_parts)

        with self.db.get_connection() as conn:
            rows = conn.execute(full_sql, params).fetchall()

        # Build RetrievedChunk objects, ordering by the original Chroma semantic rank
        results: list[RetrievedChunk] = []
        for row in rows:
            cid = row["chunk_id"]
            rank, dist = chunk_ranks.get(cid, (len(candidates) + 1, 1.0))
            # Convert L2 distance into similarity score: 1.0 / (1.0 + dist)
            similarity = 1.0 / (1.0 + max(0.0, dist))

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
                    score=similarity,
                    semantic_rank=rank,
                    sources=["semantic"],
                )
            )

        # Sort by semantic rank ascending (best match first)
        results.sort(key=lambda x: x.semantic_rank if x.semantic_rank is not None else 999999)
        return results[:top_k]
