"""Unit tests for the Adaptive Planner Service."""

from datetime import datetime, timedelta, timezone
import pytest

from studyos.learner.graph import ConceptGraph
from studyos.learner.models import (
    Concept,
    ConceptMastery,
    ConceptProgressSummary,
    ConceptRelationship,
    MasteryCategory,
    RelationshipStatus,
    RelationshipType,
    ReviewStatus,
    ReviewTask,
)
from studyos.learner.planner import (
    AdaptivePlannerService,
    PlanActionType,
)
from studyos.learner.repository import LearnerRepository
from studyos.storage.db import Database


@pytest.fixture
def planner_setup(tmp_path):
    """Sets up a database with sample concepts, reviews, and learner state."""
    db_path = tmp_path / "planner_test.db"
    db = Database(db_path)
    db.migrate()

    repo = LearnerRepository(db)
    service = AdaptivePlannerService(db, repo)

    with db.connect() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('course_cs101', 'CS 101')")

    return db, repo, service


def test_empty_course_plan(planner_setup):
    db, repo, service = planner_setup
    plan = service.generate_plan(course_id="course_cs101")
    assert plan.items == []
    assert plan.total_actions == 0
    assert plan.allocated_minutes == 0


def test_prioritizes_overdue_and_due_reviews(planner_setup):
    db, repo, service = planner_setup
    now = datetime.now(timezone.utc)

    # Concept A: Overdue review
    c_a = repo.get_or_create_concept("course_cs101", "Arrays")
    repo.upsert_concept_mastery(
        ConceptMastery(
            id="cpm_a",
            learner_id="learner_default",
            concept_id=c_a.id,
            course_id="course_cs101",
            mastery_score=0.8,
            mastery_category=MasteryCategory.STRONG,
            observation_count=5,
            correct_count=4,
        )
    )
    repo.create_or_update_review_task(
        learner_id="learner_default",
        concept_id=c_a.id,
        course_id="course_cs101",
        due_date=(now - timedelta(days=2)).isoformat(),
    )

    # Concept B: Due today review
    c_b = repo.get_or_create_concept("course_cs101", "Linked Lists")
    repo.upsert_concept_mastery(
        ConceptMastery(
            id="cpm_b",
            learner_id="learner_default",
            concept_id=c_b.id,
            course_id="course_cs101",
            mastery_score=0.75,
            mastery_category=MasteryCategory.STRONG,
            observation_count=4,
            correct_count=3,
        )
    )
    repo.create_or_update_review_task(
        learner_id="learner_default",
        concept_id=c_b.id,
        course_id="course_cs101",
        due_date=now.isoformat(),
    )

    # Concept C: Weak concept
    c_c = repo.get_or_create_concept("course_cs101", "Recursion")
    repo.upsert_concept_mastery(
        ConceptMastery(
            id="cpm_c",
            learner_id="learner_default",
            concept_id=c_c.id,
            course_id="course_cs101",
            mastery_score=0.25,
            mastery_category=MasteryCategory.WEAK,
            observation_count=4,
            correct_count=1,
        )
    )

    # Concept D: Unassessed concept
    c_d = repo.get_or_create_concept("course_cs101", "Dynamic Programming")

    plan = service.generate_plan(course_id="course_cs101")
    assert len(plan.items) == 4

    # Priority check:
    # 1. Overdue review (Arrays)
    # 2. Due review (Linked Lists)
    # 3. Weak remediation (Recursion)
    # 4. Diagnostic (Dynamic Programming)
    assert plan.items[0].concept_id == c_a.id
    assert plan.items[0].action_type == PlanActionType.OVERDUE_REVIEW
    assert plan.items[0].priority == 1

    assert plan.items[1].concept_id == c_b.id
    assert plan.items[1].action_type == PlanActionType.DUE_REVIEW
    assert plan.items[1].priority == 3

    assert plan.items[2].concept_id == c_c.id
    assert plan.items[2].action_type == PlanActionType.WEAK_REMEDIATION
    assert plan.items[2].priority == 4

    assert plan.items[3].concept_id == c_d.id
    assert plan.items[3].action_type == PlanActionType.UNASSESSED_DIAGNOSTIC
    assert plan.items[3].priority == 7


def test_target_concept_unmet_prerequisites(planner_setup):
    db, repo, service = planner_setup

    # Concepts: Trees -> Graphs -> Pathfinding
    c_trees = repo.get_or_create_concept("course_cs101", "Trees")
    c_graphs = repo.get_or_create_concept("course_cs101", "Graphs")
    c_path = repo.get_or_create_concept("course_cs101", "Pathfinding")

    # Graph: Trees is prerequisite for Graphs; Graphs is prerequisite for Pathfinding
    repo.add_concept_relationship(
        course_id="course_cs101",
        source_concept_id=c_trees.id,
        target_concept_id=c_graphs.id,
        relationship_type=RelationshipType.PREREQUISITE,
        review_status=RelationshipStatus.ACCEPTED,
    )
    repo.add_concept_relationship(
        course_id="course_cs101",
        source_concept_id=c_graphs.id,
        target_concept_id=c_path.id,
        relationship_type=RelationshipType.PREREQUISITE,
        review_status=RelationshipStatus.ACCEPTED,
    )

    # Pathfinding is our target concept. Both Trees and Graphs are unassessed.
    plan = service.generate_plan(course_id="course_cs101", target_concept_id=c_path.id)

    # Prerequisite items should precede target
    prereq_actions = [item for item in plan.items if item.action_type == PlanActionType.UNMET_PREREQUISITE]
    assert len(prereq_actions) == 2
    prereq_ids = [item.concept_id for item in prereq_actions]
    assert c_trees.id in prereq_ids
    assert c_graphs.id in prereq_ids


def test_time_budget_enforcement(planner_setup):
    db, repo, service = planner_setup

    # Create 3 concepts
    c1 = repo.get_or_create_concept("course_cs101", "Basics 1")
    c2 = repo.get_or_create_concept("course_cs101", "Basics 2")
    c3 = repo.get_or_create_concept("course_cs101", "Basics 3")

    # Diagnostic is 10 min each.
    # Budget of 20 min should allow exactly 2 items.
    plan = service.generate_plan(course_id="course_cs101", time_budget_minutes=20)
    assert len(plan.items) == 2
    assert plan.allocated_minutes == 20
    assert plan.time_budget_minutes == 20
