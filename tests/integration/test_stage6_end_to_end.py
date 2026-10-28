"""Comprehensive Stage 6 end-to-end workflow integration test.

Tests the complete StudyOS learning loop:
1. Synthetic material ingestion -> Chunks stored in SQLite.
2. Direct concept discovery from chunks (independent of exams) -> Concept catalog with chunk provenance.
3. Grounded concept relationship graph (prerequisite and related) -> DAG validation & cycle rejection.
4. Graded assessment performance -> Authoritative learner events.
5. Advanced forgetting-aware review scheduling (local-fsrs-v1).
6. Transparent adaptive study plan -> Prioritization, time budgeting, and prerequisite checking.
7. Explicit opt-in external research with fake search provider -> Grounded summary and [EXT-N] citations.
"""

from datetime import datetime, timedelta, timezone
import pytest

from studyos.assessment.models import Difficulty, ExamResult, QuestionType
from studyos.inference.fake import FakeChatProvider
from studyos.learner.discovery import ConceptDiscoveryService
from studyos.learner.graph import ConceptGraph, CycleDetectedError
from studyos.learner.models import (
    Concept,
    ConceptRelationship,
    ConceptReviewStatus,
    MasteryCategory,
    RelationshipProvenance,
    RelationshipStatus,
    RelationshipType,
    ReviewRating,
    ReviewStatus,
)
from studyos.learner.planner import AdaptivePlannerService, PlanActionType
from studyos.learner.repository import LearnerRepository
from studyos.learner.service import LearnerService
from studyos.research import (
    ExternalResearchService,
    FakeSearchProvider,
    ResearchRecord,
    ResearchResult,
    ResearchStatus,
)
from studyos.storage.db import Database


@pytest.fixture
def stage6_env(tmp_path):
    db_file = tmp_path / "stage6_e2e.db"
    db = Database(db_file)
    db.migrate()

    # Seed course and document chunks
    with db.connect() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('course_ds101', 'Data Structures & Algorithms')")
        conn.execute(
            """
            INSERT INTO documents (id, course_id, filename, file_type, file_size_bytes, sha256_hash, status, chunk_count)
            VALUES ('doc_trees', 'course_ds101', 'trees_and_heaps.pdf', 'pdf', 1024, 'hash123', 'completed', 3)
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (id, course_id, document_id, chunk_index, content, token_count, content_hash, location_type, location_index, location_title)
            VALUES
            ('chk_1', 'course_ds101', 'doc_trees', 0, 'Binary Search Trees (BST) maintain sorted order where left child is smaller and right child is greater.', 25, 'h1', 'heading', 1, '1. Binary Search Trees'),
            ('chk_2', 'course_ds101', 'doc_trees', 1, 'Heaps and Priority Queues: A binary heap is a complete binary tree that satisfies the heap property.', 28, 'h2', 'heading', 2, '2. Binary Heaps'),
            ('chk_3', 'course_ds101', 'doc_trees', 2, 'Graph Algorithms: Graphs consist of vertices and edges. Pathfinding algorithms like Dijkstra rely on heaps.', 30, 'h3', 'heading', 3, '3. Graph Algorithms');
            """
        )

    return db


def test_stage6_end_to_end_learning_cycle(stage6_env):
    db = stage6_env
    repo = LearnerRepository(db)

    # -------------------------------------------------------------------------
    # STEP 1: Direct Concept Discovery from Raw Ingested Materials
    # -------------------------------------------------------------------------
    discovery_svc = ConceptDiscoveryService(db=db, repository=repo, chat_provider=None)
    disc_result = discovery_svc.discover_concepts(course_id="course_ds101", auto_accept=True)

    assert disc_result.status == "completed"
    assert disc_result.chunks_scanned == 3
    assert disc_result.total_discovered >= 2

    concepts = repo.list_concepts(course_id="course_ds101")
    assert len(concepts) >= 2

    # Verify chunk provenance was saved
    cpt_bst = next((c for c in concepts if "binary search" in c.name.lower() or "tree" in c.name.lower()), concepts[0])
    chunks = repo.get_chunks_for_concept(cpt_bst.id)
    assert len(chunks) >= 1
    assert chunks[0]["is_live"] is True

    # -------------------------------------------------------------------------
    # STEP 2: Grounded Concept Graph & Cycle Prevention
    # -------------------------------------------------------------------------
    cpt_heap = repo.get_or_create_concept(
        course_id="course_ds101",
        name="Binary Heaps",
        description="Priority queue tree representation.",
    )
    cpt_graph_algo = repo.get_or_create_concept(
        course_id="course_ds101",
        name="Dijkstra Algorithm",
        description="Single-source shortest path using priority queues.",
    )

    # Add prerequisite: Binary Heaps -> Dijkstra Algorithm
    rel1 = repo.add_concept_relationship(
        course_id="course_ds101",
        source_concept_id=cpt_heap.id,
        target_concept_id=cpt_graph_algo.id,
        relationship_type=RelationshipType.PREREQUISITE,
        source_evidence="Dijkstra relies on heaps for efficient minimum extraction.",
        review_status=RelationshipStatus.ACCEPTED,
    )
    assert rel1.id is not None
    assert rel1.relationship_type == RelationshipType.PREREQUISITE

    # Attempt to introduce a cycle: Dijkstra Algorithm -> Binary Heaps
    with pytest.raises(CycleDetectedError) as exc_info:
        repo.add_concept_relationship(
            course_id="course_ds101",
            source_concept_id=cpt_graph_algo.id,
            target_concept_id=cpt_heap.id,
            relationship_type=RelationshipType.PREREQUISITE,
            review_status=RelationshipStatus.ACCEPTED,
        )
    assert "Cycle detected" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # STEP 3: Assessment Performance & Learner State
    # -------------------------------------------------------------------------
    learner_svc = LearnerService(db=db, repository=repo)

    # Record graded event for Binary Heaps (e.g. scored 1.0 / 1.0)
    repo.record_event(
        learner_id="learner_default",
        course_id="course_ds101",
        event_type="exam_question_graded",
        concept_id=cpt_heap.id,
        score_achieved=1.0,
        score_max=1.0,
        is_correct=True,
    )
    # Recompute mastery
    m_heap = repo.recompute_mastery_for_concept("learner_default", cpt_heap.id)
    assert m_heap.observation_count == 1
    assert m_heap.mastery_category == MasteryCategory.NEEDS_EVIDENCE  # 1 observation < 3

    # -------------------------------------------------------------------------
    # STEP 4: Advanced Forgetting-Aware Review Scheduling (local-fsrs-v1)
    # -------------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    task = repo.create_or_update_review_task(
        learner_id="learner_default",
        concept_id=cpt_heap.id,
        course_id="course_ds101",
        due_date=(now - timedelta(days=1)).isoformat(),  # Overdue
    )
    assert task.is_overdue is True

    # Complete review with rating "good" using FSRS policy
    completed_task = repo.complete_review_task(task_id=task.id, rating=ReviewRating.GOOD, policy="local-fsrs-v1")
    assert completed_task.status == ReviewStatus.COMPLETED

    # Verify updated mastery with FSRS fields
    m_updated = repo.get_concept_mastery("learner_default", cpt_heap.id)
    assert m_updated.scheduler_policy == "local-fsrs-v1"
    assert m_updated.stability >= 1.0
    assert m_updated.interval_days >= 1.0
    assert m_updated.retrievability is not None

    # -------------------------------------------------------------------------
    # STEP 5: Transparent Adaptive Study Planner
    # -------------------------------------------------------------------------
    planner = AdaptivePlannerService(db=db, repository=repo)

    # Plan targeting Dijkstra Algorithm (whose prerequisite Binary Heaps only has 1 observation)
    plan = planner.generate_plan(
        course_id="course_ds101",
        target_concept_id=cpt_graph_algo.id,
        time_budget_minutes=30,
    )
    assert plan.target_concept_id == cpt_graph_algo.id
    assert plan.total_actions >= 1

    # Prerequisite Binary Heaps should be recommended before Dijkstra
    prereq_items = [item for item in plan.items if item.action_type == PlanActionType.UNMET_PREREQUISITE]
    assert len(prereq_items) == 1
    assert prereq_items[0].concept_id == cpt_heap.id
    assert "Required prerequisite" in prereq_items[0].reason

    # -------------------------------------------------------------------------
    # STEP 6: Explicit Opt-In External Research with Citations
    # -------------------------------------------------------------------------
    fake_provider = FakeSearchProvider(
        [
            ResearchResult(
                url="https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm",
                title="Dijkstra's Algorithm",
                snippet="An algorithm for finding shortest paths between nodes in a weighted graph.",
                publisher="Wikipedia",
                published_date="2023-11-01",
            ),
            ResearchResult(
                url="https://algs4.cs.princeton.edu/44sp/",
                title="Shortest Paths - Algorithms 4th Edition",
                snippet="Priority-first search implementation of Dijkstra's algorithm using indexed binary heaps.",
                publisher="Princeton University",
                published_date="2022-04-15",
            ),
        ]
    )

    research_svc = ExternalResearchService(
        db=db,
        provider=fake_provider,
        chat_client=None,  # Deterministic synthesis
    )

    # Opt-in is strictly enforced
    with pytest.raises(PermissionError):
        research_svc.research("Dijkstra indexed heap optimization", explicit_opt_in=False)

    summary = research_svc.research(
        query="Dijkstra indexed heap optimization",
        course_id="course_ds101",
        explicit_opt_in=True,
    )

    assert summary.status == ResearchStatus.SUCCESS
    assert len(summary.sources) == 2
    assert summary.sources[0].citation_marker == "[EXT-1]"
    assert summary.sources[1].citation_marker == "[EXT-2]"
    assert "[EXT-1]" in summary.summary_text
    assert "[EXT-2]" in summary.summary_text
    assert "DISCLAIMER" in summary.disclaimer

    # Confirm course mastery was NOT modified by external research
    m_check = repo.get_concept_mastery("learner_default", cpt_heap.id)
    assert m_check.observation_count == m_updated.observation_count
    assert m_check.mastery_score == m_updated.mastery_score
