"""Unit tests for concept name normalization and question-to-concept mapping."""

from studyos.assessment.models import Difficulty, ExamQuestion, QuestionSourceRef, QuestionType
from studyos.inference.fake import FakeChatProvider
from studyos.learner.concept_mapper import ConceptMapper, normalize_concept_name


def test_normalize_concept_name():
    assert normalize_concept_name("  Dynamic Programming  ") == "dynamic programming"
    assert normalize_concept_name("Binary Search Trees...") == "binary search trees"
    assert normalize_concept_name("Graphs / Adjacency   Matrix") == "graphs / adjacency matrix"
    assert normalize_concept_name("") == "unclassified"
    assert normalize_concept_name("   ---   ") == "unclassified"


def test_extract_concept_labels_from_sources():
    q = ExamQuestion(
        id="q1",
        exam_id="ex1",
        question_index=1,
        question_type=QuestionType.MCQ,
        difficulty=Difficulty.MEDIUM,
        question_text="How does Dijkstra relaxation work?",
        correct_answer="B",
        explanation="Explanation [E1].",
        sources=[
            QuestionSourceRef(
                chunk_id="chk1",
                evidence_marker="E1",
                location_title="Dijkstra Shortest Path",
                excerpt="Dijkstra algorithm iteratively relaxes edges...",
            ),
            QuestionSourceRef(
                chunk_id="chk2",
                evidence_marker="E2",
                location_title="Slide 14",  # Generic title should be filtered out
                excerpt="Generic slide content...",
            ),
        ],
    )

    labels = ConceptMapper.extract_concept_labels_from_question(q, exam_topic=None)
    assert "Dijkstra Shortest Path" in labels
    assert "Slide 14" not in labels


def test_extract_concept_labels_fallback_to_exam_topic():
    q = ExamQuestion(
        id="q2",
        exam_id="ex1",
        question_index=2,
        question_type=QuestionType.TRUE_FALSE,
        difficulty=Difficulty.EASY,
        question_text="Is BFS complete?",
        correct_answer="True",
        explanation="BFS is complete on finite graphs.",
        sources=[],
    )
    labels = ConceptMapper.extract_concept_labels_from_question(q, exam_topic="Graph Traversal")
    assert labels == ["Graph Traversal"]


def test_extract_concept_labels_fallback_to_unclassified():
    q = ExamQuestion(
        id="q3",
        exam_id="ex1",
        question_index=3,
        question_type=QuestionType.SHORT_ANSWER,
        difficulty=Difficulty.HARD,
        question_text="Explain something.",
        correct_answer="Answer",
        explanation="Exp",
        sources=[],
    )
    labels = ConceptMapper.extract_concept_labels_from_question(q, exam_topic=None)
    assert labels == ["Unclassified"]


def test_propose_concepts_with_llm():
    fake_chat = FakeChatProvider(
        model_name="fake-llm",
        default_response="Shortest Path Algorithms, Edge Relaxation",
    )
    mapper = ConceptMapper(chat_provider=fake_chat)
    sources = [
        QuestionSourceRef(
            chunk_id="chk1",
            evidence_marker="E1",
            excerpt="Dijkstra algorithm performs edge relaxation on non-negative graphs.",
        )
    ]
    proposed = mapper.propose_concepts_with_llm(
        question_text="How does Dijkstra work?",
        sources=sources,
        course_name="CS101",
    )
    assert "Shortest Path Algorithms" in proposed
    assert "Edge Relaxation" in proposed


def test_propose_concepts_llm_failure_fails_safe():
    fake_chat = FakeChatProvider(model_name="fake-llm")
    fake_chat.set_error(RuntimeError("Simulated LLM error"))
    mapper = ConceptMapper(chat_provider=fake_chat)
    sources = [
        QuestionSourceRef(
            chunk_id="chk1",
            evidence_marker="E1",
            excerpt="Dijkstra algorithm...",
        )
    ]
    # Should not raise; fails open to empty list
    proposed = mapper.propose_concepts_with_llm("question", sources, "CS101")
    assert proposed == []
