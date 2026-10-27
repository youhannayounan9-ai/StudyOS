"""Typed models for StudyOS evidence-grounded tutoring service and citations."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, field_validator

from studyos.retrieval.models import RetrievalMode, RetrievedChunk


class ResponseStyle(str, Enum):
    """Pedagogical explanation style."""

    BALANCED = "balanced"
    CONCISE = "concise"
    DETAILED = "detailed"


class TutorRequest(BaseModel):
    """Validated request contract for the AI study tutor."""

    query: str = Field(description="Student's study question or conceptual inquiry")
    course_id: str | None = Field(default=None, description="Optional primary course ID or code")
    course_ids: list[str] | None = Field(default=None, description="Optional multi-course filter")
    document_ids: list[str] | None = Field(default=None, description="Optional document filter")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of candidate evidence chunks")
    style: ResponseStyle = Field(
        default=ResponseStyle.BALANCED,
        description="Explanation depth (balanced, concise, or detailed)",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace.")
        if len(cleaned) > 2000:
            raise ValueError(f"Query string exceeds maximum permitted length of 2000 characters ({len(cleaned)}).")
        return cleaned


class EvidencePassage(BaseModel):
    """A single retrieved chunk bound to an application-assigned evidence marker."""

    marker: str = Field(description="Temporary request-scoped evidence marker, e.g. 'E1'")
    chunk: RetrievedChunk = Field(description="Authoritative SQLite-backed chunk record")

    @property
    def formatted_marker(self) -> str:
        """Standard bracketed marker representation: [E1]."""
        return f"[{self.marker}]"


class ValidatedCitation(BaseModel):
    """A model citation that has been validated against active retrieval evidence."""

    marker: str = Field(description="Normalized evidence marker, e.g. 'E1'")
    chunk_id: str
    document_id: str
    course_id: str
    course_name: str
    filename: str
    file_type: str
    location_type: str
    location_index: int
    location_title: str | None = None
    excerpt: str = Field(description="Short factual excerpt from the authoritative chunk")

    @property
    def citation_label(self) -> str:
        """Human-readable citation string formatted from trusted SQLite metadata."""
        loc = f"{self.location_type.lower()} {self.location_index}"
        if self.location_title:
            loc += f" ({self.location_title})"
        return f"[{self.marker}] {self.filename} — {loc}"


class TutorResponse(BaseModel):
    """Standardized response contract for AI tutor answers."""

    query: str
    answer: str
    citations: list[ValidatedCitation] = Field(default_factory=list)
    unsupported: bool = Field(
        default=False,
        description="True if the tutor abstained because retrieved evidence was missing or insufficient",
    )
    retrieval_mode: RetrievalMode = Field(description="Retrieval modality actually used")
    warnings: list[str] = Field(default_factory=list)
    raw_markers_found: list[str] = Field(
        default_factory=list,
        description="All evidence markers extracted from model output",
    )
    rejected_markers: list[str] = Field(
        default_factory=list,
        description="Markers cited by the model that did not match active request evidence",
    )
    evidence_count: int = Field(default=0, description="Number of evidence chunks provided to tutor prompt")

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0
