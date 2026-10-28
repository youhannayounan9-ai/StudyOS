"""Domain models for StudyOS external research."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class ResearchStatus(str, Enum):
    """Execution status of an external research query."""

    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


class SourceType(str, Enum):
    """Classification of external research sources."""

    WEB_PAGE = "web_page"
    ENCYCLOPEDIA = "encyclopedia"
    SCHOLARLY_ARTICLE = "scholarly_article"
    DOCUMENTATION = "documentation"


class ExternalSource(BaseModel):
    """An external web or academic source fetched and validated for research."""

    id: str = Field(description="Unique source ID")
    research_id: str | None = None
    citation_index: int = Field(ge=1, description="Application-assigned citation index [EXT-N]")
    url: str = Field(description="Original verified external URL")
    title: str = Field(description="Document or article title")
    publisher: str | None = Field(default=None, description="Publisher or domain name")
    published_date: str | None = Field(default=None, description="Publication date if reliably present")
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    snippet: str = Field(description="Concise excerpt or text summary from source")
    source_type: SourceType = SourceType.WEB_PAGE
    is_valid: bool = True

    @property
    def citation_marker(self) -> str:
        return f"[EXT-{self.citation_index}]"


class ResearchResult(BaseModel):
    """Raw item returned from an external search provider."""

    url: str
    title: str
    snippet: str
    publisher: str | None = None
    published_date: str | None = None


class ResearchRecord(BaseModel):
    """Persisted external research operation record."""

    id: str
    query: str
    course_id: str | None = None
    summary: str | None = None
    status: ResearchStatus = ResearchStatus.SUCCESS
    error_message: str | None = None
    created_at: str | None = None
    sources: list[ExternalSource] = Field(default_factory=list)


class ResearchSummary(BaseModel):
    """Grounded research summary containing validated citations."""

    research_id: str
    query: str
    course_id: str | None = None
    summary_text: str
    sources: list[ExternalSource]
    valid_citations_used: list[int] = Field(default_factory=list)
    invalid_citations_rejected: list[str] = Field(default_factory=list)
    status: ResearchStatus = ResearchStatus.SUCCESS
    disclaimer: str = (
        "EXTERNAL RESEARCH DISCLAIMER: The information below was retrieved from external web sources, "
        "not your authoritative course materials. It is provided for supplementary research only and "
        "has not altered your course mastery or concept definitions."
    )
