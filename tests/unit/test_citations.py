"""Unit tests for citation token extraction, set-membership validation, and trusted metadata resolution."""

from __future__ import annotations

import pytest

from studyos.retrieval.models import RetrievedChunk
from studyos.tutoring.citations import (
    CitationValidator,
    extract_excerpt,
)


def _make_sample_chunk(
    chunk_id: str,
    doc_id: str = "doc_123",
    course_id: str = "crs_algo",
    course_name: str = "Algorithms",
    filename: str = "lecture_03.pdf",
    location_type: str = "page",
    location_index: int = 14,
    location_title: str | None = None,
    content: str = "Dijkstra's algorithm solves single-source shortest paths on non-negative weighted graphs.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        course_id=course_id,
        course_name=course_name,
        filename=filename,
        file_type="pdf",
        chunk_index=0,
        location_type=location_type,
        location_index=location_index,
        location_title=location_title,
        content=content,
        token_count=len(content.split()),
        content_hash="abc123hash",
        score=0.85,
    )


def test_validate_citations_multiple_valid_markers():
    """Valid markers in text must resolve to authoritative SQLite chunk metadata."""
    chunk1 = _make_sample_chunk("chk_1", filename="algo_w3.pdf", location_type="page", location_index=12)
    chunk2 = _make_sample_chunk(
        "chk_2",
        filename="sorting.pptx",
        location_type="slide",
        location_index=7,
        location_title="Bellman-Ford",
    )

    evidence_map = {"E1": chunk1, "E2": chunk2}

    text = "Dijkstra handles non-negative weights [E1]. Meanwhile, Bellman-Ford can handle negative weights [E2]."
    result = CitationValidator.validate_citations(text, evidence_map)

    assert result.is_valid is True
    assert len(result.valid_citations) == 2
    assert len(result.rejected_markers) == 0

    c1 = result.valid_citations[0]
    assert c1.marker == "E1"
    assert c1.chunk_id == "chk_1"
    assert c1.filename == "algo_w3.pdf"
    assert c1.citation_label == "[E1] algo_w3.pdf — page 12"

    c2 = result.valid_citations[1]
    assert c2.marker == "E2"
    assert c2.chunk_id == "chk_2"
    assert c2.filename == "sorting.pptx"
    assert c2.citation_label == "[E2] sorting.pptx — slide 7 (Bellman-Ford)"


def test_validate_citations_deduplication():
    """Repeated markers in the generated text must be deduplicated while preserving order."""
    chunk1 = _make_sample_chunk("chk_1", filename="algo_w3.pdf")
    evidence_map = {"E1": chunk1}

    text = "Point one [E1]. Point two is also supported [E1], and point three [e1]."
    result = CitationValidator.validate_citations(text, evidence_map)

    assert len(result.valid_citations) == 1
    assert result.valid_citations[0].marker == "E1"
    assert len(result.raw_markers) == 3


def test_validate_citations_rejects_invented_markers():
    """Invented markers (e.g. [E999], [E5] when only E1 exists) must be rejected with warnings."""
    chunk1 = _make_sample_chunk("chk_1")
    evidence_map = {"E1": chunk1}

    text = "Fact from evidence [E1]. But here is an invented claim [E999] and another [E42]."
    result = CitationValidator.validate_citations(text, evidence_map)

    assert len(result.valid_citations) == 1
    assert result.valid_citations[0].marker == "E1"
    assert result.rejected_markers == ["E999", "E42"]
    assert len(result.warnings) == 2
    assert any("[E999]" in w for w in result.warnings)
    assert any("[E42]" in w for w in result.warnings)
    assert result.is_valid is False  # Contains rejected markers


def test_validate_citations_no_markers_in_text():
    """Text without any citation markers returns an empty valid citations list."""
    chunk1 = _make_sample_chunk("chk_1")
    evidence_map = {"E1": chunk1}

    text = "This is a statement with no bracketed citation markers at all."
    result = CitationValidator.validate_citations(text, evidence_map)

    assert len(result.valid_citations) == 0
    assert len(result.raw_markers) == 0
    assert len(result.rejected_markers) == 0
    assert result.is_valid is False


def test_model_cannot_fabricate_metadata():
    """Model output mentioning false filenames or false page numbers does not overwrite trusted SQLite metadata."""
    chunk1 = _make_sample_chunk(
        "chk_real",
        filename="real_course_notes.pdf",
        location_type="page",
        location_index=5,
    )
    evidence_map = {"E1": chunk1}

    # Model hallucinates that E1 is from "fake_handbook.pdf page 999"
    text = "According to fake_handbook.pdf on page 999 [E1], Dijkstra is greedy."
    result = CitationValidator.validate_citations(text, evidence_map)

    assert len(result.valid_citations) == 1
    citation = result.valid_citations[0]
    # Authoritative metadata MUST come from chunk1, NEVER model text
    assert citation.filename == "real_course_notes.pdf"
    assert citation.location_index == 5
    assert citation.citation_label == "[E1] real_course_notes.pdf — page 5"


def test_request_marker_isolation():
    """Citations from request A cannot resolve against evidence assigned to request B."""
    chunk_req_a = _make_sample_chunk("chk_a", filename="course_a.pdf")
    chunk_req_b = _make_sample_chunk("chk_b", filename="course_b.pdf")

    map_req_a = {"E1": chunk_req_a}
    map_req_b = {"E1": chunk_req_b}

    text = "Claim supported by [E1]."

    res_a = CitationValidator.validate_citations(text, map_req_a)
    res_b = CitationValidator.validate_citations(text, map_req_b)

    assert res_a.valid_citations[0].filename == "course_a.pdf"
    assert res_b.valid_citations[0].filename == "course_b.pdf"


def test_extract_excerpt_handling():
    """Excerpt extraction trims whitespace and caps at reasonable preview length."""
    short_text = "Short text."
    assert extract_excerpt(short_text, max_length=50) == "Short text."

    long_text = "This is the first sentence. " + ("Extra explanatory text that goes on and on. " * 10)
    excerpt = extract_excerpt(long_text, max_length=60)
    assert len(excerpt) <= 65
    assert excerpt.endswith(".") or excerpt.endswith("...")
