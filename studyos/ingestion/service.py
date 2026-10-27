"""Orchestration service for document validation, parsing, chunking, and persistence."""

from __future__ import annotations

import logging
from pathlib import Path
import uuid

from studyos.chunking.chunker import StructureAwareChunker
from studyos.core.config import AppConfig, get_default_config
from studyos.ingestion.models import (
    DuplicateDocumentError,
    EmptyFileError,
    FileTooLargeError,
    IngestionError,
    IngestionResult,
)
from studyos.ingestion.parsers import get_parser_for_file
from studyos.storage.db import Database

logger = logging.getLogger("studyos.ingestion")


class IngestionService:
    """
    Coordinates the document ingestion pipeline.

    Steps:
    1. Validation of existence, size, and format.
    2. SHA-256 hash calculation and duplicate detection.
    3. Parsing into normalized PageElement structures.
    4. Structure-aware chunking with breadcrumb header injection.
    5. Atomic SQLite transactional persistence and trigger-based FTS5 sync.
    """

    def __init__(
        self,
        db: Database,
        config: AppConfig | None = None,
        chunker: StructureAwareChunker | None = None,
    ) -> None:
        self.db = db
        self.config = config or get_default_config()
        self.chunker = chunker or StructureAwareChunker(
            target_chunk_size=self.config.chunk_size_tokens,
            chunk_overlap=self.config.chunk_overlap_tokens,
            min_chunk_size=self.config.min_chunk_size_tokens,
        )

    def ingest_file(
        self,
        file_path: Path | str,
        course_id: str,
        force_reindex: bool = False,
    ) -> IngestionResult:
        """
        Ingest an individual document file into a course.

        Args:
            file_path: Path to the target document.
            course_id: Primary key of the course to associate the document with.
            force_reindex: If True and document already exists in course, replace it.

        Returns:
            IngestionResult with status, counts, warnings, and document ID.

        Raises:
            IngestionError or subclass if validation, parsing, or persistence fails.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found: {path}")

        # Ensure target course exists
        course = self.db.get_course(course_id)
        if not course:
            raise IngestionError(f"Course not found with id: {course_id}")

        # Select parser and compute file hash/size
        parser = get_parser_for_file(
            path, max_file_size_bytes=self.config.max_file_size_bytes
        )
        file_size, sha256_hash = parser.validate_file(path)

        # Duplicate detection check
        existing_doc = self.db.get_document_by_hash(
            course_id=course_id, sha256_hash=sha256_hash
        )
        if existing_doc and not force_reindex:
            logger.info(
                "Duplicate file detected (SHA-256: %s) for course %s",
                sha256_hash[:8],
                course_id,
            )
            return IngestionResult(
                document_id=existing_doc.id,
                status="duplicate",
                sha256_hash=sha256_hash,
                element_count=existing_doc.total_pages_or_slides,
                chunk_count=existing_doc.chunk_count,
                warnings=[
                    f"Document with matching SHA-256 hash already exists in this course: '{existing_doc.filename}'"
                ],
            )

        # 1. Parse document outside any transaction (if parsing fails, previous doc is untouched)
        parsed_doc = parser.parse(path)

        # 2. Pre-generate document ID and chunk document outside transaction
        new_doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        chunks = self.chunker.chunk_document(
            parsed_doc=parsed_doc,
            course_id=course_id,
            document_id=new_doc_id,
        )

        # 3. Persist document metadata and chunks atomically within a single transaction
        with self.db.transaction() as conn:
            # If force_reindex and previous document exists, delete it within this transaction
            if existing_doc and force_reindex:
                logger.info(
                    "Force reindex requested: atomically deleting previous document %s",
                    existing_doc.id,
                )
                self.db.delete_document(existing_doc.id, conn=conn)

            # Insert replacement document
            doc = self.db.create_document(
                course_id=course_id,
                filename=parsed_doc.filename,
                file_type=parsed_doc.file_type,
                file_size_bytes=parsed_doc.file_size_bytes,
                sha256_hash=parsed_doc.sha256_hash,
                status="processing",
                document_id=new_doc_id,
                conn=conn,
            )

            # Insert all replacement chunks (triggers automatically populate chunks_fts)
            self.db.insert_chunks(chunks, conn=conn)

            # Mark replacement document as ready
            self.db.update_document_status(
                document_id=doc.id,
                status="ready",
                total_pages_or_slides=len(parsed_doc.elements),
                chunk_count=len(chunks),
                conn=conn,
            )

        logger.info(
            "Successfully ingested document %s (%d elements, %d chunks)",
            doc.id,
            len(parsed_doc.elements),
            len(chunks),
        )

        return IngestionResult(
            document_id=doc.id,
            status="ready",
            sha256_hash=sha256_hash,
            element_count=len(parsed_doc.elements),
            chunk_count=len(chunks),
            warnings=parsed_doc.warnings,
        )
