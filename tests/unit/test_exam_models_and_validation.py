"""Unit tests for assessment models, request parameter validation, and question schema validation."""

from __future__ import annotations

import json
import pytest

from studyos.assessment.models import (
    Difficulty,
    ExamRequest,
    MCQOption,
    QuestionType,
)
from studyos.assessment.validator import ExamValidationError, ExamValidator
from studyos.retrieval.models import RetrievedChunk


def _make_dummy_chunk(chunk_id: str = "chk_01") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc_01",
        course_id="crs_01",
        course_name="Algorithms",
        filename="lecture01.pdf",
        file_type="pdf",
        chunk_index=0,
        location_type="page",
        location_index=5,
        content="Dijkstra's algorithm solves single-source shortest path with non-negative edge weights.",
        token_count=12,
        content_hash="hash1",
        score=0.9,
    )


def test_exam_request_validation():
    """Verify parameter boundaries on ExamRequest."""
    # Valid request
    req = ExamRequest(course_id="crs_algo", question_count=5, difficulty=Difficulty.HARD)
    assert req.course_id == "crs_algo"
    assert req.question_count == 5

    # Empty course_id
    with pytest.raises(ValueError, match="course_id cannot be empty"):
        ExamRequest(course_id="   ")

    # Invalid question counts (ge=1, le=20)
    with pytest.raises(ValueError):
        ExamRequest(course_id="crs_algo", question_count=0)

    with pytest.raises(ValueError):
        ExamRequest(course_id="crs_algo", question_count=25)


def test_validate_valid_mcq_payload():
    """Valid MCQ payload parses and populates all fields with active evidence markers."""
    evidence_map = {"E1": _make_dummy_chunk("chk_01")}
    raw_payload = json.dumps(
        {
            "questions": [
                {
                    "type": "mcq",
                    "question": "What is the primary constraint of Dijkstra's algorithm?",
                    "difficulty": "medium",
                    "options": [
                        {"id": "A", "text": "Requires negative edges"},
                        {"id": "B", "text": "Requires non-negative edge weights"},
                        {"id": "C", "text": "Only works on DAGs"},
                        {"id": "D", "text": "Requires all-pairs matrices"}
                    ],
                    "correct_answer": "B",
                    "explanation": "Dijkstra requires non-negative edge weights [E1].",
                    "evidence_markers": ["E1"]
                }
            ]
        }
    )

    questions = ExamValidator.validate_raw_payload(raw_payload, "ex_1", evidence_map)
    assert len(questions) == 1
    q = questions[0]
    assert q.question_type == QuestionType.MCQ
    assert q.correct_answer == "B"
    assert len(q.options) == 4
    assert len(q.sources) == 1
    assert q.sources[0].evidence_marker == "E1"
    assert q.sources[0].chunk_id == "chk_01"


def test_validate_mcq_missing_correct_answer_in_options():
    """MCQ where correct_answer does not match any choice ID is rejected."""
    evidence_map = {"E1": _make_dummy_chunk("chk_01")}
    raw_payload = json.dumps(
        {
            "questions": [
                {
                    "type": "mcq",
                    "question": "Valid question text here?",
                    "options": [
                        {"id": "A", "text": "Option A"},
                        {"id": "B", "text": "Option B"}
                    ],
                    "correct_answer": "Z",  # Not in options!
                    "explanation": "Supported [E1]",
                    "evidence_markers": ["E1"]
                }
            ]
        }
    )

    with pytest.raises(ExamValidationError, match="does not match any choice ID"):
        ExamValidator.validate_raw_payload(raw_payload, "ex_1", evidence_map)


def test_validate_mcq_duplicate_choice_texts_rejected():
    """MCQ where distractors have duplicate texts is rejected."""
    evidence_map = {"E1": _make_dummy_chunk("chk_01")}
    raw_payload = json.dumps(
        {
            "questions": [
                {
                    "type": "mcq",
                    "question": "Valid question text here?",
                    "options": [
                        {"id": "A", "text": "Identical choice"},
                        {"id": "B", "text": "Identical choice"}
                    ],
                    "correct_answer": "A",
                    "explanation": "Supported [E1]",
                    "evidence_markers": ["E1"]
                }
            ]
        }
    )

    with pytest.raises(ExamValidationError, match="Duplicate option text"):
        ExamValidator.validate_raw_payload(raw_payload, "ex_1", evidence_map)


def test_validate_true_false_question():
    """True/False question normalizes correct answer to 'True' or 'False'."""
    evidence_map = {"E1": _make_dummy_chunk("chk_01")}
    raw_payload = json.dumps(
        {
            "questions": [
                {
                    "type": "true_false",
                    "question": "Dijkstra's algorithm can handle negative cycle detection.",
                    "difficulty": "easy",
                    "correct_answer": "false",  # lowercase
                    "explanation": "Bellman-Ford is needed for negative cycle detection [E1].",
                    "evidence_markers": ["E1"]
                }
            ]
        }
    )

    questions = ExamValidator.validate_raw_payload(raw_payload, "ex_1", evidence_map)
    assert len(questions) == 1
    q = questions[0]
    assert q.question_type == QuestionType.TRUE_FALSE
    assert q.correct_answer == "False"
    assert len(q.options) == 2


def test_validate_short_answer_question():
    """Short answer question requires reference answer and structured rubric."""
    evidence_map = {"E1": _make_dummy_chunk("chk_01")}
    raw_payload = json.dumps(
        {
            "questions": [
                {
                    "type": "short_answer",
                    "question": "Explain why Dijkstra fails with negative weights.",
                    "difficulty": "hard",
                    "correct_answer": "Greedy choice assumption fails because visited vertices may be updated via negative paths.",
                    "rubric": {
                        "criteria": ["Mentions greedy choice", "Explains path weight reduction"],
                        "key_points": ["greedy", "negative edge"],
                        "max_score": 2.0
                    },
                    "explanation": "Greedy vertex finalization is invalid [E1].",
                    "evidence_markers": ["E1"]
                }
            ]
        }
    )

    questions = ExamValidator.validate_raw_payload(raw_payload, "ex_1", evidence_map)
    assert len(questions) == 1
    q = questions[0]
    assert q.question_type == QuestionType.SHORT_ANSWER
    assert q.rubric is not None
    assert len(q.rubric.criteria) == 2
    assert q.rubric.max_score == 2.0


def test_validate_rejects_empty_question_text():
    """Questions with empty or trivially short text are rejected."""
    evidence_map = {"E1": _make_dummy_chunk("chk_01")}
    raw_payload = json.dumps(
        {
            "questions": [
                {
                    "type": "true_false",
                    "question": "Why?",  # Too short!
                    "correct_answer": "True",
                    "evidence_markers": ["E1"]
                }
            ]
        }
    )

    with pytest.raises(ExamValidationError, match="too short"):
        ExamValidator.validate_raw_payload(raw_payload, "ex_1", evidence_map)


def test_validate_deduplicates_repeated_questions():
    """If the model generates duplicate questions, duplicates are discarded."""
    evidence_map = {"E1": _make_dummy_chunk("chk_01")}
    raw_payload = json.dumps(
        {
            "questions": [
                {
                    "type": "true_false",
                    "question": "Dijkstra uses a priority queue.",
                    "correct_answer": "True",
                    "evidence_markers": ["E1"]
                },
                {
                    "type": "true_false",
                    "question": "Dijkstra uses a priority queue.",  # Duplicate!
                    "correct_answer": "True",
                    "evidence_markers": ["E1"]
                }
            ]
        }
    )

    questions = ExamValidator.validate_raw_payload(raw_payload, "ex_1", evidence_map)
    assert len(questions) == 1
