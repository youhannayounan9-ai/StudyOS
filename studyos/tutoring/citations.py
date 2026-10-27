"""Citation extraction, set-membership validation, and trusted metadata resolution."""

from __future__ import annotations

import re
from typing import Mapping
from pydantic import BaseModel, Field

from studyos.retrieval.models import RetrievedChunk
from studyos.tutoring.models import ValidatedCitation


# Regular expression matching bracketed evidence markers such as [E1], [E2], case-insensitively
EVIDENCE_MARKER_REGEX = re.compile(r"\[(E\d+)\]", re.IGNORECASE)


def extract_excerpt(content: str, max_length: int = 200) -> str:
    """Extract a short, clean excerpt from chunk text for citation preview."""
    clean = " ".join(content.strip().split())
    if len(clean) <= max_length:
        return clean
    # Attempt to cut at a sentence or word boundary
    truncated = clean[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length // 2:
        return truncated[: last_period + 1]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space] + "..."
    return truncated + "..."


class CitationValidationResult(BaseModel):
    """Result of citation token extraction and verification against active evidence."""

    valid_citations: list[ValidatedCitation] = Field(default_factory=list)
    raw_markers: list[str] = Field(default_factory=list)
    rejected_markers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if at least one valid citation was matched and no ungrounded markers were cited."""
        return len(self.valid_citations) > 0 and len(self.rejected_markers) == 0


class CitationValidator:
    """Validates model citation markers against active request evidence and resolves trusted metadata."""

    @classmethod
    def validate_citations(
        cls,
        text: str,
        evidence_map: Mapping[str, RetrievedChunk],
    ) -> CitationValidationResult:
        """Parse all citation markers in the generated text and validate against active evidence.

        Args:
            text: The raw assistant output text.
            evidence_map: Mapping of normalized marker ('E1') to authoritative RetrievedChunk.

        Returns:
            CitationValidationResult with verified citations, rejected markers, and warnings.
        """
        raw_matches = EVIDENCE_MARKER_REGEX.findall(text)
        normalized_markers = [m.upper() for m in raw_matches]

        seen_markers: set[str] = set()
        valid_citations: list[ValidatedCitation] = []
        rejected_markers: list[str] = []
        warnings: list[str] = []

        for marker in normalized_markers:
            if marker in seen_markers:
                continue
            seen_markers.add(marker)

            if marker in evidence_map:
                chunk = evidence_map[marker]
                valid_citations.append(
                    ValidatedCitation(
                        marker=marker,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        course_id=chunk.course_id,
                        course_name=chunk.course_name,
                        filename=chunk.filename,
                        file_type=chunk.file_type,
                        location_type=chunk.location_type,
                        location_index=chunk.location_index,
                        location_title=chunk.location_title,
                        excerpt=extract_excerpt(chunk.content),
                    )
                )
            else:
                rejected_markers.append(marker)
                warnings.append(
                    f"Model cited unverified marker '[{marker}]' which does not exist in the active evidence set. "
                    "This citation has been rejected to prevent citation spoofing."
                )

        return CitationValidationResult(
            valid_citations=valid_citations,
            raw_markers=normalized_markers,
            rejected_markers=rejected_markers,
            warnings=warnings,
        )
