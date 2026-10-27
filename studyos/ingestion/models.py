"""Document and ingestion data models for StudyOS."""

from __future__ import annotations

import hashlib
import uuid
from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Base exception for document ingestion errors."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when the document file format is unsupported."""
    pass


class FileTooLargeError(IngestionError):
    """Raised when a document exceeds the maximum permitted size limit."""
    pass


class EmptyFileError(IngestionError):
    """Raised when an uploaded file contains zero bytes."""
    pass


class CorruptFileError(IngestionError):
    """Raised when a document file cannot be decoded or parsed due to corruption."""
    pass


class EncryptedFileError(IngestionError):
    """Raised when a document is password-protected or encrypted."""
    pass


class ExtractionEmptyError(IngestionError):
    """Raised when a document contains no extractable text (e.g. scanned image-only PDF)."""
    pass


class DuplicateDocumentError(IngestionError):
    """Raised when a document with identical hash already exists in the course."""
    pass


# ---------------------------------------------------------------------------
# Normalized Document Models
# ---------------------------------------------------------------------------

class PageElement(BaseModel):
    """A discrete structural element from a parsed document (page, slide, or section)."""

    text: str = Field(description="Extracted clean text content for this element")
    location_type: Literal["page", "slide", "section"] = Field(
        description="Structural boundary type"
    )
    location_index: int = Field(
        ge=1,
        description="1-based index (page number, slide number, or section sequence)",
    )
    location_title: str | None = Field(
        default=None,
        description="Optional title or heading associated with this element",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Extraction warnings encountered during parsing of this element",
    )


class ParsedDocument(BaseModel):
    """The normalized output of a format-specific document parser."""

    filename: str
    file_type: Literal["pdf", "docx", "pptx", "txt"]
    sha256_hash: str
    file_size_bytes: int
    elements: list[PageElement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_text_length(self) -> int:
        return sum(len(el.text) for el in self.elements)


class DocumentChunk(BaseModel):
    """A bounded semantic chunk suitable for retrieval and FTS indexing."""

    chunk_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:12]}")
    document_id: str
    course_id: str
    chunk_index: int = Field(ge=0, description="0-based sequence index within document")
    location_type: Literal["page", "slide", "section"]
    location_index: int = Field(ge=1)
    location_title: str | None = None
    content: str = Field(description="Full text of the chunk including breadcrumb header")
    token_count: int = Field(ge=1, description="Estimated token count")
    content_hash: str = Field(description="SHA-256 hash of the chunk content")

    @classmethod
    def create(
        cls,
        document_id: str,
        course_id: str,
        chunk_index: int,
        location_type: Literal["page", "slide", "section"],
        location_index: int,
        content: str,
        token_count: int,
        location_title: str | None = None,
        chunk_id: str | None = None,
    ) -> DocumentChunk:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        actual_id = chunk_id or f"chk_{uuid.uuid4().hex[:12]}"
        return cls(
            chunk_id=actual_id,
            document_id=document_id,
            course_id=course_id,
            chunk_index=chunk_index,
            location_type=location_type,
            location_index=location_index,
            location_title=location_title,
            content=content,
            token_count=max(1, token_count),
            content_hash=content_hash,
        )


class Course(BaseModel):
    """Relational entity representing an academic course."""

    id: str
    name: str
    code: str | None = None
    description: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class Document(BaseModel):
    """Relational entity representing an ingested document."""

    id: str
    course_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    sha256_hash: str
    status: Literal["pending", "processing", "ready", "failed"]
    total_pages_or_slides: int = 0
    chunk_count: int = 0
    error_message: str | None = None
    created_at: str | None = None


class IngestionResult(BaseModel):
    """Structured report returned by the ingestion service."""

    document_id: str
    status: Literal["ready", "failed", "duplicate"]
    sha256_hash: str
    element_count: int
    chunk_count: int
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
