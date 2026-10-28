"""Unit tests for course-scoped concept relationship graph, cycle detection, and DAG validation."""

import pytest
from studyos.learner.graph import ConceptGraph, CycleDetectedError
from studyos.learner.models import (
    ConceptRelationship,
    RelationshipProvenance,
    RelationshipStatus,
    RelationshipType,
)
from studyos.learner.repository import LearnerRepository
from studyos.storage.db import Database


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_graph.db"
    db = Database(db_file)
    db.migrate()
    return db


def test_valid_same_course_prerequisite_and_related_edges(test_db):
    """Test valid prerequisite and related relationships within the same course."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Data Structures", "CS201")

    c_arrays = repo.get_or_create_concept(crs.id, "Arrays")
    c_lists = repo.get_or_create_concept(crs.id, "Linked Lists")
    c_stacks = repo.get_or_create_concept(crs.id, "Stacks")

    # Arrays is a prerequisite for Stacks
    rel_pre = repo.add_concept_relationship(
        course_id=crs.id,
        source_concept_id=c_arrays.id,
        target_concept_id=c_stacks.id,
        relationship_type=RelationshipType.PREREQUISITE,
        provenance=RelationshipProvenance.MANUAL,
        source_evidence="Stacks can be implemented via fixed arrays.",
    )
    assert rel_pre.relationship_type == RelationshipType.PREREQUISITE
    assert rel_pre.review_status == RelationshipStatus.ACCEPTED
    assert rel_pre.source_concept_id == c_arrays.id
    assert rel_pre.target_concept_id == c_stacks.id

    # Arrays and Linked Lists are related
    rel_rel = repo.add_concept_relationship(
        course_id=crs.id,
        source_concept_id=c_arrays.id,
        target_concept_id=c_lists.id,
        relationship_type=RelationshipType.RELATED,
    )
    assert rel_rel.relationship_type == RelationshipType.RELATED

    # List relationships
    all_rels = repo.list_concept_relationships(crs.id)
    assert len(all_rels) == 2

    # Graph retrieval
    graph = repo.get_concept_graph(crs.id)
    assert graph.get_prerequisite_ids(c_stacks.id) == [c_arrays.id]
    assert c_lists.id in graph.get_related_ids(c_arrays.id)
    assert c_arrays.id in graph.get_related_ids(c_lists.id)


def test_cross_course_edge_rejected(test_db):
    """Prerequisites across different courses must be rejected."""
    repo = LearnerRepository(test_db)
    crs1 = test_db.create_course("Calculus", "MATH101")
    crs2 = test_db.create_course("Physics", "PHYS101")

    c1 = repo.get_or_create_concept(crs1.id, "Derivatives")
    c2 = repo.get_or_create_concept(crs2.id, "Velocity")

    with pytest.raises(ValueError, match="Cross-course relationship rejected"):
        repo.add_concept_relationship(
            course_id=crs1.id,
            source_concept_id=c1.id,
            target_concept_id=c2.id,
            relationship_type=RelationshipType.PREREQUISITE,
        )


def test_self_edge_rejected(test_db):
    """Self-referencing relationship edge (A -> A) must be rejected."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Algorithms", "CS301")
    c = repo.get_or_create_concept(crs.id, "Recursion")

    with pytest.raises(ValueError, match="Self-referencing relationship is not allowed"):
        repo.add_concept_relationship(
            course_id=crs.id,
            source_concept_id=c.id,
            target_concept_id=c.id,
            relationship_type=RelationshipType.PREREQUISITE,
        )


def test_direct_and_multi_hop_cycle_detection(test_db):
    """Cycle detection must prevent circular dependencies in prerequisites."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Compilers", "CS401")

    c_lex = repo.get_or_create_concept(crs.id, "Lexing")
    c_parse = repo.get_or_create_concept(crs.id, "Parsing")
    c_ast = repo.get_or_create_concept(crs.id, "AST Generation")

    # 1. Lexing -> Parsing
    repo.add_concept_relationship(
        course_id=crs.id,
        source_concept_id=c_lex.id,
        target_concept_id=c_parse.id,
        relationship_type=RelationshipType.PREREQUISITE,
    )

    # Direct 2-node cycle: Parsing -> Lexing must be rejected!
    with pytest.raises(CycleDetectedError, match="Cycle detected"):
        repo.add_concept_relationship(
            course_id=crs.id,
            source_concept_id=c_parse.id,
            target_concept_id=c_lex.id,
            relationship_type=RelationshipType.PREREQUISITE,
            review_status=RelationshipStatus.ACCEPTED,
        )

    # 2. Parsing -> AST
    repo.add_concept_relationship(
        course_id=crs.id,
        source_concept_id=c_parse.id,
        target_concept_id=c_ast.id,
        relationship_type=RelationshipType.PREREQUISITE,
    )

    # 3-node cycle: AST -> Lexing must be rejected!
    with pytest.raises(CycleDetectedError, match="Cycle detected"):
        repo.add_concept_relationship(
            course_id=crs.id,
            source_concept_id=c_ast.id,
            target_concept_id=c_lex.id,
            relationship_type=RelationshipType.PREREQUISITE,
            review_status=RelationshipStatus.ACCEPTED,
        )

    # Topological prerequisite order for AST: [Lexing, Parsing]
    graph = repo.get_concept_graph(crs.id)
    prereqs = graph.get_prerequisite_ids(c_ast.id)
    assert prereqs == [c_lex.id, c_parse.id]


def test_proposed_relationship_lifecycle_and_cycle_prevention(test_db):
    """A proposed relationship does not trigger cycles until accepted, and cannot be accepted if cyclical."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Databases", "CS501")

    c_bplus = repo.get_or_create_concept(crs.id, "B+ Trees")
    c_index = repo.get_or_create_concept(crs.id, "Indexing")

    # Accepted: Indexing -> B+ Trees
    repo.add_concept_relationship(
        course_id=crs.id,
        source_concept_id=c_index.id,
        target_concept_id=c_bplus.id,
        relationship_type=RelationshipType.PREREQUISITE,
        review_status=RelationshipStatus.ACCEPTED,
    )

    # Proposed opposite edge: B+ Trees -> Indexing
    prop = repo.add_concept_relationship(
        course_id=crs.id,
        source_concept_id=c_bplus.id,
        target_concept_id=c_index.id,
        relationship_type=RelationshipType.PREREQUISITE,
        review_status=RelationshipStatus.PROPOSED,
    )
    assert prop.review_status == RelationshipStatus.PROPOSED

    # Accepting this proposed edge must fail cycle check!
    with pytest.raises(CycleDetectedError, match="creates a cycle"):
        repo.update_relationship_status(prop.id, RelationshipStatus.ACCEPTED)

    # Rejecting it succeeds
    rejected = repo.update_relationship_status(prop.id, RelationshipStatus.REJECTED)
    assert rejected.review_status == RelationshipStatus.REJECTED


def test_delete_relationship(test_db):
    """Deleting a relationship updates graph traversal immediately."""
    repo = LearnerRepository(test_db)
    crs = test_db.create_course("Operating Systems", "CS601")

    c1 = repo.get_or_create_concept(crs.id, "Threads")
    c2 = repo.get_or_create_concept(crs.id, "Semaphores")

    rel = repo.add_concept_relationship(
        course_id=crs.id,
        source_concept_id=c1.id,
        target_concept_id=c2.id,
        relationship_type=RelationshipType.PREREQUISITE,
    )
    graph_before = repo.get_concept_graph(crs.id)
    assert graph_before.get_prerequisite_ids(c2.id) == [c1.id]

    deleted = repo.delete_concept_relationship(rel.id)
    assert deleted is True

    graph_after = repo.get_concept_graph(crs.id)
    assert graph_after.get_prerequisite_ids(c2.id) == []
