"""Unit tests for TutorService orchestration, evidence grounding, and abstention."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from studyos.core.config import AppConfig
from studyos.inference.fake import FakeChatProvider
from studyos.inference.protocol import ChatTimeoutError, OllamaChatConnectionError
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievalMode, RetrievalResponse, RetrievedChunk
from studyos.tutoring.models import ResponseStyle, TutorRequest, TutorResponse
from studyos.tutoring.service import TutorService


def _make_mock_chunk(
    chunk_id: str = "chk_01",
    doc_id: str = "doc_01",
    course_id: str = "crs_algo",
    course_name: str = "Algorithms",
    filename: str = "graphs.pdf",
    location_type: str = "page",
    location_index: int = 4,
    content: str = "Dijkstra's algorithm finds the shortest path between nodes in a graph.",
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
        content=content,
        token_count=len(content.split()),
        content_hash="h123",
        score=0.035,
    )


def test_tutor_answers_with_valid_evidence():
    """When retrieval returns evidence, tutor invokes chat provider and validates citations."""
    mock_retrieval = MagicMock(spec=HybridRetrievalService)
    chunk = _make_mock_chunk()
    mock_retrieval.search.return_value = RetrievalResponse(
        query="What is Dijkstra?",
        results=[chunk],
        mode=RetrievalMode.HYBRID,
        warnings=[],
        total_candidates=1,
    )

    fake_chat = FakeChatProvider(
        default_response="Dijkstra's algorithm computes single-source shortest paths [E1]."
    )

    service = TutorService(retrieval_service=mock_retrieval, chat_provider=fake_chat)

    req = TutorRequest(query="What is Dijkstra?", course_id="crs_algo")
    resp = service.answer(req)

    assert resp.unsupported is False
    assert len(resp.citations) == 1
    assert resp.citations[0].marker == "E1"
    assert resp.citations[0].filename == "graphs.pdf"
    assert resp.citations[0].location_index == 4
    assert resp.retrieval_mode == RetrievalMode.HYBRID
    assert resp.evidence_count == 1
    assert "Dijkstra's algorithm" in resp.answer


def test_tutor_abstains_immediately_on_empty_retrieval():
    """When retrieval finds 0 matching chunks, tutor abstains without calling the LLM."""
    mock_retrieval = MagicMock(spec=HybridRetrievalService)
    mock_retrieval.search.return_value = RetrievalResponse(
        query="Quantum computing entanglement",
        results=[],
        mode=RetrievalMode.HYBRID,
        warnings=[],
        total_candidates=0,
    )

    fake_chat = FakeChatProvider()
    service = TutorService(retrieval_service=mock_retrieval, chat_provider=fake_chat)

    req = TutorRequest(query="Quantum computing entanglement")
    resp = service.answer(req)

    assert resp.unsupported is True
    assert len(resp.citations) == 0
    assert "The uploaded materials do not provide enough evidence" in resp.answer
    # CRITICAL: Chat provider must NOT have been called!
    assert len(fake_chat.requests) == 0


def test_tutor_propagates_course_and_document_filters():
    """Course and document filters in TutorRequest are passed directly to HybridRetrievalService."""
    mock_retrieval = MagicMock(spec=HybridRetrievalService)
    mock_retrieval.search.return_value = RetrievalResponse(
        query="test", results=[], mode=RetrievalMode.HYBRID
    )
    fake_chat = FakeChatProvider()
    service = TutorService(retrieval_service=mock_retrieval, chat_provider=fake_chat)

    req = TutorRequest(
        query="Filter test",
        course_id="crs_1",
        course_ids=["crs_1", "crs_2"],
        document_ids=["doc_99"],
        top_k=3,
    )
    service.answer(req)

    mock_retrieval.search.assert_called_once_with(
        query="Filter test",
        top_k=3,
        course_id="crs_1",
        course_ids=["crs_1", "crs_2"],
        document_ids=["doc_99"],
    )


def test_tutor_preserves_degraded_lexical_retrieval_mode():
    """When retrieval runs in degraded mode, tutor records the warning and mode accurately."""
    mock_retrieval = MagicMock(spec=HybridRetrievalService)
    chunk = _make_mock_chunk()
    mock_retrieval.search.return_value = RetrievalResponse(
        query="test query",
        results=[chunk],
        mode=RetrievalMode.DEGRADED_LEXICAL,
        warnings=["Ollama is offline"],
        total_candidates=1,
    )

    fake_chat = FakeChatProvider(default_response="Explanation grounded in [E1].")
    service = TutorService(retrieval_service=mock_retrieval, chat_provider=fake_chat)

    req = TutorRequest(query="test query")
    resp = service.answer(req)

    assert resp.retrieval_mode == RetrievalMode.DEGRADED_LEXICAL
    assert any("DEGRADED_LEXICAL" in w for w in resp.warnings)
    assert any("Ollama is offline" in w for w in resp.warnings)
    assert len(resp.citations) == 1


def test_tutor_bounded_repair_on_uncited_response():
    """When model generates factual answer without citations, tutor triggers bounded repair."""
    mock_retrieval = MagicMock(spec=HybridRetrievalService)
    chunk = _make_mock_chunk()
    mock_retrieval.search.return_value = RetrievalResponse(
        query="What is Dijkstra?",
        results=[chunk],
        mode=RetrievalMode.HYBRID,
        warnings=[],
        total_candidates=1,
    )

    fake_chat = FakeChatProvider()
    # 1st call: Answer with NO markers
    fake_chat.queue_response("Dijkstra's algorithm solves shortest paths.")
    # 2nd call (repair): Rewritten answer with [E1]
    fake_chat.queue_response("Dijkstra's algorithm solves shortest paths [E1].")

    service = TutorService(retrieval_service=mock_retrieval, chat_provider=fake_chat)

    req = TutorRequest(query="What is Dijkstra?")
    resp = service.answer(req)

    # Must have called chat twice (initial + 1 repair)
    assert len(fake_chat.requests) == 2
    assert resp.unsupported is False
    assert len(resp.citations) == 1
    assert resp.citations[0].marker == "E1"


def test_tutor_bounded_repair_failure_reports_warning():
    """If repair attempt still produces no citations, tutor flags it as ungrounded/unsupported."""
    mock_retrieval = MagicMock(spec=HybridRetrievalService)
    chunk = _make_mock_chunk()
    mock_retrieval.search.return_value = RetrievalResponse(
        query="What is Dijkstra?",
        results=[chunk],
        mode=RetrievalMode.HYBRID,
        warnings=[],
        total_candidates=1,
    )

    fake_chat = FakeChatProvider(
        default_response="Factual claim without any citations ever."
    )

    service = TutorService(retrieval_service=mock_retrieval, chat_provider=fake_chat)

    req = TutorRequest(query="What is Dijkstra?")
    resp = service.answer(req)

    assert len(fake_chat.requests) == 2  # Attempted repair once
    assert resp.unsupported is True
    assert len(resp.citations) == 0
    assert any("without valid evidence citations" in w for w in resp.warnings)


def test_tutor_request_validation():
    """Empty query or oversized query raises ValueError."""
    with pytest.raises(ValueError, match="empty or whitespace"):
        TutorRequest(query="   ")

    with pytest.raises(ValueError, match="exceeds maximum permitted length"):
        TutorRequest(query="x" * 2001)

    with pytest.raises(ValueError):
        TutorRequest(query="valid", top_k=0)


def test_tutor_chat_provider_exception_propagation():
    """Connection errors from chat provider are raised cleanly as typed exceptions."""
    mock_retrieval = MagicMock(spec=HybridRetrievalService)
    chunk = _make_mock_chunk()
    mock_retrieval.search.return_value = RetrievalResponse(
        query="What is Dijkstra?",
        results=[chunk],
        mode=RetrievalMode.HYBRID,
        warnings=[],
        total_candidates=1,
    )

    fake_chat = FakeChatProvider()
    fake_chat.set_error(OllamaChatConnectionError("Connection lost to daemon"))

    service = TutorService(retrieval_service=mock_retrieval, chat_provider=fake_chat)

    req = TutorRequest(query="What is Dijkstra?")
    with pytest.raises(OllamaChatConnectionError, match="Connection lost to daemon"):
        service.answer(req)
