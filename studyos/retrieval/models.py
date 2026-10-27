"""Typed domain models for retrieval results, modes, and vector synchronization."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    """Execution mode of the retrieval query."""

    HYBRID = "hybrid"
    LEXICAL_ONLY = "lexical_only"
    SEMANTIC_ONLY = "semantic_only"
    DEGRADED_LEXICAL = "degraded_lexical"


class RetrievedChunk(BaseModel):
    """An authoritative chunk retrieved and scored through lexical, semantic, or hybrid search.

    All document, course, location, and textual data is populated directly from SQLite
    to guarantee metadata integrity and prevent stale or spoofed vector-store records.
    """

    chunk_id: str
    document_id: str
    course_id: str
    course_name: str
    filename: str
    file_type: str
    chunk_index: int
    location_type: str
    location_index: int
    location_title: str | None = None
    content: str
    token_count: int
    content_hash: str
    score: float = Field(description="Combined RRF score or component relevance score")
    lexical_rank: int | None = Field(default=None, description="1-based lexical candidate rank if matched")
    semantic_rank: int | None = Field(default=None, description="1-based semantic candidate rank if matched")
    sources: list[str] = Field(
        default_factory=list,
        description="Search modalities that contributed to this result (e.g. ['lexical', 'semantic'])",
    )

    @property
    def breadcrumb(self) -> str:
        """Formatted citation breadcrumb for evidence display."""
        loc = f"{self.location_type.capitalize()}: {self.location_index}"
        if self.location_title:
            loc += f" - {self.location_title}"
        return f"[Source: {self.filename} | {loc}]"


class RetrievalResponse(BaseModel):
    """Standardized response contract for all retrieval queries."""

    query: str
    results: list[RetrievedChunk] = Field(default_factory=list)
    mode: RetrievalMode
    warnings: list[str] = Field(default_factory=list)
    total_candidates: int = Field(default=0, description="Total distinct candidates evaluated before top-k slicing")
    course_ids_filter: list[str] | None = None
    document_ids_filter: list[str] | None = None

    @property
    def count(self) -> int:
        """Number of returned results."""
        return len(self.results)


class SyncStats(BaseModel):
    """Diagnostic statistics produced by vector index synchronization."""

    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    total_chunks: int = 0
    duration_seconds: float = 0.0

    @property
    def has_changes(self) -> bool:
        """Return True if any vectors were added, updated, or deleted."""
        return (self.added + self.updated + self.deleted) > 0
