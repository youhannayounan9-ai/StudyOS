"""Course-scoped concept relationship graph, DAG cycle validation, and traversal algorithms."""

from __future__ import annotations

from collections import deque
import logging
from typing import Sequence

from studyos.learner.models import (
    Concept,
    ConceptRelationship,
    RelationshipStatus,
    RelationshipType,
)

logger = logging.getLogger(__name__)


class CycleDetectedError(ValueError):
    """Raised when adding or accepting a prerequisite edge would create a cycle in the DAG."""
    pass


class ConceptGraph:
    """Manages in-memory graph representation and topological operations for course concepts."""

    def __init__(self, course_id: str, relationships: Sequence[ConceptRelationship]) -> None:
        self.course_id = course_id
        # Adjacency list: source_id -> list of target_ids for accepted prerequisites
        self.adj: dict[str, list[str]] = {}
        # Reverse adjacency list: target_id -> list of source_ids (prerequisites of target)
        self.rev_adj: dict[str, list[str]] = {}
        # Undirected related adjacency
        self.related_adj: dict[str, list[str]] = {}

        for rel in relationships:
            if rel.course_id != course_id:
                continue

            if rel.relationship_type == RelationshipType.PREREQUISITE:
                if rel.review_status == RelationshipStatus.ACCEPTED:
                    self.adj.setdefault(rel.source_concept_id, []).append(rel.target_concept_id)
                    self.rev_adj.setdefault(rel.target_concept_id, []).append(rel.source_concept_id)
            elif rel.relationship_type == RelationshipType.RELATED:
                if rel.review_status == RelationshipStatus.ACCEPTED:
                    self.related_adj.setdefault(rel.source_concept_id, []).append(rel.target_concept_id)
                    self.related_adj.setdefault(rel.target_concept_id, []).append(rel.source_concept_id)

    def would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Return True if adding directed edge source_id -> target_id would create a cycle.

        A cycle would be created if there is already a directed path from target_id to source_id.
        """
        if source_id == target_id:
            return True

        # BFS from target_id looking for source_id
        visited: set[str] = set()
        queue: deque[str] = deque([target_id])

        while queue:
            curr = queue.popleft()
            if curr == source_id:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            for neighbor in self.adj.get(curr, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        return False

    def get_prerequisite_ids(self, concept_id: str, recursive: bool = True) -> list[str]:
        """Return all prerequisite concept IDs for concept_id in topological prerequisite order.

        Prerequisites that must be learned earliest appear first in the returned list.
        """
        if not recursive:
            return list(dict.fromkeys(self.rev_adj.get(concept_id, [])))

        # Find all ancestors (prerequisites)
        visited: set[str] = set()
        ancestors: set[str] = set()
        queue: deque[str] = deque([concept_id])

        while queue:
            curr = queue.popleft()
            for prereq in self.rev_adj.get(curr, []):
                if prereq not in ancestors:
                    ancestors.add(prereq)
                    queue.append(prereq)

        if not ancestors:
            return []

        # Topological sort of ancestors subgraph:
        # In-degree based on edges between ancestors
        in_degree: dict[str, int] = {node: 0 for node in ancestors}
        for u in ancestors:
            for v in self.adj.get(u, []):
                if v in ancestors:
                    in_degree[v] += 1

        # Queue nodes with in_degree == 0 (no prerequisites among ancestors)
        kahn_queue: deque[str] = deque([node for node in ancestors if in_degree[node] == 0])
        ordered: list[str] = []

        while kahn_queue:
            u = kahn_queue.popleft()
            ordered.append(u)
            for v in self.adj.get(u, []):
                if v in ancestors:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        kahn_queue.append(v)

        # In case of remaining nodes (should not happen if acyclic)
        for node in ancestors:
            if node not in ordered:
                ordered.append(node)

        return ordered

    def get_dependent_ids(self, concept_id: str) -> list[str]:
        """Return all concepts that directly depend on concept_id as a prerequisite."""
        return list(dict.fromkeys(self.adj.get(concept_id, [])))

    def get_related_ids(self, concept_id: str) -> list[str]:
        """Return all concepts directly related to concept_id."""
        return list(dict.fromkeys(self.related_adj.get(concept_id, [])))
