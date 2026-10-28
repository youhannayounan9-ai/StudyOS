"""Adaptive study planning service for StudyOS.

Generates transparent, deterministic study plans based on actual learner state,
concept relationship DAGs, forgetting-aware review schedules, and optional time budgets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any
from pydantic import BaseModel, Field

from studyos.learner.graph import ConceptGraph
from studyos.learner.models import (
    ConceptMastery,
    ConceptProgressSummary,
    MasteryCategory,
    ReviewStatus,
)
from studyos.learner.repository import LearnerRepository
from studyos.learner.scheduler import calculate_retrievability
from studyos.storage.db import Database

logger = logging.getLogger(__name__)


class PlanActionType(str, Enum):
    """Types of prioritized study actions."""

    OVERDUE_REVIEW = "overdue_review"
    UNMET_PREREQUISITE = "unmet_prerequisite"
    DUE_REVIEW = "due_review"
    WEAK_REMEDIATION = "weak_remediation"
    TARGET_STUDY = "target_study"
    GATHER_EVIDENCE = "gather_evidence"
    UNASSESSED_DIAGNOSTIC = "unassessed_diagnostic"
    MODERATE_REINFORCEMENT = "moderate_reinforcement"


class PlannedStudyItem(BaseModel):
    """An actionable, ranked study item with realistic duration and diagnostic reason."""

    concept_id: str
    concept_name: str
    course_id: str
    action_type: PlanActionType
    action_title: str
    priority: int = Field(description="1 (highest urgency) to 8 (lowest)")
    estimated_minutes: int = Field(description="Documented estimated duration in minutes")
    reason: str = Field(description="Transparent diagnostic explanation")
    mastery_score: float
    mastery_category: MasteryCategory
    observation_count: int
    retrievability: float | None = None
    unmet_prerequisites: list[str] = Field(default_factory=list)
    supporting_evidence: dict[str, Any] = Field(default_factory=dict)


class AdaptiveStudyPlan(BaseModel):
    """A personalized, time-budgeted study plan for today."""

    learner_id: str
    course_id: str | None = None
    target_concept_id: str | None = None
    target_concept_name: str | None = None
    time_budget_minutes: int | None = None
    allocated_minutes: int = 0
    items: list[PlannedStudyItem] = Field(default_factory=list)
    total_actions: int = 0
    generated_at: str


class AdaptivePlannerService:
    """Deterministic, transparent study planner operating 100% offline."""

    # Documented standard action durations
    DURATION_OVERDUE_REVIEW = 5
    DURATION_DUE_REVIEW = 5
    DURATION_PREREQUISITE_STUDY = 15
    DURATION_WEAK_PRACTICE = 15
    DURATION_TARGET_STUDY = 20
    DURATION_DIAGNOSTIC_QUIZ = 10
    DURATION_GATHER_EVIDENCE = 10
    DURATION_REINFORCE = 10

    def __init__(self, db: Database, repository: LearnerRepository | None = None) -> None:
        self.db = db
        self.repository = repository or LearnerRepository(db)

    def generate_plan(
        self,
        learner_id: str = "learner_default",
        course_id: str | None = None,
        target_concept_id: str | None = None,
        time_budget_minutes: int | None = None,
        max_items: int = 10,
    ) -> AdaptiveStudyPlan:
        """Generate a deterministic, prioritized study plan."""
        now = datetime.now(timezone.utc)

        # 1. Resolve scope
        if target_concept_id:
            target_cpt = self.repository.get_concept(target_concept_id)
            if not target_cpt:
                raise ValueError(f"Target concept '{target_concept_id}' not found.")
            target_name = target_cpt.name
            course_id = target_cpt.course_id
        else:
            target_name = None

        # 2. Load concepts in scope
        concepts = self.repository.list_concepts(course_id=course_id)
        if not concepts:
            return AdaptiveStudyPlan(
                learner_id=learner_id,
                course_id=course_id,
                target_concept_id=target_concept_id,
                target_concept_name=target_name,
                time_budget_minutes=time_budget_minutes,
                allocated_minutes=0,
                items=[],
                total_actions=0,
                generated_at=now.isoformat(),
            )

        cpt_map = {c.id: c for c in concepts}

        # 3. Load mastery and active scheduled reviews
        mastery_list = self.repository.list_concept_mastery(learner_id=learner_id, course_id=course_id)
        mastery_map = {m.concept_id: m for m in mastery_list}

        reviews_list = self.repository.list_review_tasks(
            learner_id=learner_id,
            course_id=course_id,
            status=ReviewStatus.SCHEDULED,
        )
        reviews_map = {r.concept_id: r for r in reviews_list}

        # 4. Load concept graphs per course
        graphs: dict[str, ConceptGraph] = {}
        for c in concepts:
            if c.course_id not in graphs:
                graphs[c.course_id] = self.repository.get_concept_graph(c.course_id)

        # 5. Build candidate actions
        candidates: list[PlannedStudyItem] = []

        # (a) If target concept specified, inspect prerequisites first
        target_unmet_ids: list[str] = []
        if target_concept_id:
            t_graph = graphs.get(course_id, ConceptGraph(course_id, []))
            prereq_ids = t_graph.get_prerequisite_ids(target_concept_id, recursive=True)

            for pid in prereq_ids:
                p_cpt = cpt_map.get(pid) or self.repository.get_concept(pid)
                if not p_cpt:
                    continue
                p_m = mastery_map.get(pid) or self.repository.get_concept_mastery(learner_id, pid)
                cat = p_m.mastery_category if p_m else MasteryCategory.UNASSESSED
                score = p_m.mastery_score if p_m else 0.0
                obs = p_m.observation_count if p_m else 0

                # Prerequisite is unmet if unassessed or weak
                if cat in (MasteryCategory.UNASSESSED, MasteryCategory.NEEDS_EVIDENCE, MasteryCategory.WEAK):
                    target_unmet_ids.append(pid)
                    candidates.append(
                        PlannedStudyItem(
                            concept_id=pid,
                            concept_name=p_cpt.name,
                            course_id=p_cpt.course_id,
                            action_type=PlanActionType.UNMET_PREREQUISITE,
                            action_title=f"Study Prerequisite: {p_cpt.name}",
                            priority=2,
                            estimated_minutes=self.DURATION_PREREQUISITE_STUDY,
                            reason=(
                                f"Required prerequisite for target '{target_name}'. "
                                f"Current status is '{cat.value}' ({score:.2f} mastery across {obs} attempts). "
                                f"Address this foundational concept before advancing."
                            ),
                            mastery_score=score,
                            mastery_category=cat,
                            observation_count=obs,
                            supporting_evidence={
                                "target_concept_id": target_concept_id,
                                "target_concept_name": target_name,
                                "prerequisite_of": target_concept_id,
                            },
                        )
                    )

        # (b) Process all concepts for reviews, weakness, evidence, and diagnostics
        for cpt in concepts:
            # Skip if already added as prerequisite of target
            if any(item.concept_id == cpt.id for item in candidates):
                continue

            m = mastery_map.get(cpt.id)
            score = m.mastery_score if m else 0.0
            cat = m.mastery_category if m else MasteryCategory.UNASSESSED
            obs = m.observation_count if m else 0
            stability = m.stability if m else 1.0
            ret = m.retrievability if m else 1.0

            rt = reviews_map.get(cpt.id)

            # Priority 1: Overdue reviews
            if rt and rt.is_overdue:
                candidates.append(
                    PlannedStudyItem(
                        concept_id=cpt.id,
                        concept_name=cpt.name,
                        course_id=cpt.course_id,
                        action_type=PlanActionType.OVERDUE_REVIEW,
                        action_title=f"Overdue Review: {cpt.name}",
                        priority=1,
                        estimated_minutes=self.DURATION_OVERDUE_REVIEW,
                        reason=f"Spaced review was due on {rt.due_date[:10]}. Immediate recall practice prevents memory decay.",
                        mastery_score=score,
                        mastery_category=cat,
                        observation_count=obs,
                        retrievability=ret,
                        supporting_evidence={"due_date": rt.due_date, "is_overdue": True},
                    )
                )
                continue

            # Priority 3: Scheduled reviews due today
            if rt and rt.is_due:
                candidates.append(
                    PlannedStudyItem(
                        concept_id=cpt.id,
                        concept_name=cpt.name,
                        course_id=cpt.course_id,
                        action_type=PlanActionType.DUE_REVIEW,
                        action_title=f"Due Review: {cpt.name}",
                        priority=3,
                        estimated_minutes=self.DURATION_DUE_REVIEW,
                        reason=f"Review is due today for scheduled spaced reinforcement.",
                        mastery_score=score,
                        mastery_category=cat,
                        observation_count=obs,
                        retrievability=ret,
                        supporting_evidence={"due_date": rt.due_date, "is_due": True},
                    )
                )
                continue

            # Priority 4: Weak concepts with sufficient observations
            if cat == MasteryCategory.WEAK:
                candidates.append(
                    PlannedStudyItem(
                        concept_id=cpt.id,
                        concept_name=cpt.name,
                        course_id=cpt.course_id,
                        action_type=PlanActionType.WEAK_REMEDIATION,
                        action_title=f"Remediate Weak Concept: {cpt.name}",
                        priority=4,
                        estimated_minutes=self.DURATION_WEAK_PRACTICE,
                        reason=(
                            f"Low accuracy on validated attempts ({m.correct_count if m else 0}/{obs} correct, "
                            f"mastery: {score:.2f}). Practice recommended."
                        ),
                        mastery_score=score,
                        mastery_category=cat,
                        observation_count=obs,
                        supporting_evidence={"mastery_score": score, "correct": m.correct_count if m else 0},
                    )
                )
                continue

            # Priority 5: Target concept if prerequisites are met
            if target_concept_id and cpt.id == target_concept_id:
                if not target_unmet_ids:
                    candidates.append(
                        PlannedStudyItem(
                            concept_id=cpt.id,
                            concept_name=cpt.name,
                            course_id=cpt.course_id,
                            action_type=PlanActionType.TARGET_STUDY,
                            action_title=f"Study Target: {cpt.name}",
                            priority=5,
                            estimated_minutes=self.DURATION_TARGET_STUDY,
                            reason=f"All prerequisites for target concept '{cpt.name}' are satisfied. Ready for focused study.",
                            mastery_score=score,
                            mastery_category=cat,
                            observation_count=obs,
                            supporting_evidence={"prerequisites_cleared": True},
                        )
                    )
                continue

            # Priority 6: Concepts needing more evidence (1 <= n < 3)
            if cat == MasteryCategory.NEEDS_EVIDENCE:
                candidates.append(
                    PlannedStudyItem(
                        concept_id=cpt.id,
                        concept_name=cpt.name,
                        course_id=cpt.course_id,
                        action_type=PlanActionType.GATHER_EVIDENCE,
                        action_title=f"Gather Evidence: {cpt.name}",
                        priority=6,
                        estimated_minutes=self.DURATION_GATHER_EVIDENCE,
                        reason=f"Only {obs} graded observation(s) recorded. Take a quiz to establish confident mastery.",
                        mastery_score=score,
                        mastery_category=cat,
                        observation_count=obs,
                        supporting_evidence={"observation_count": obs},
                    )
                )
                continue

            # Priority 7: Unassessed concepts
            if cat == MasteryCategory.UNASSESSED:
                candidates.append(
                    PlannedStudyItem(
                        concept_id=cpt.id,
                        concept_name=cpt.name,
                        course_id=cpt.course_id,
                        action_type=PlanActionType.UNASSESSED_DIAGNOSTIC,
                        action_title=f"Diagnostic Quiz: {cpt.name}",
                        priority=7,
                        estimated_minutes=self.DURATION_DIAGNOSTIC_QUIZ,
                        reason="Concept has not yet been assessed in any exam. Take an initial diagnostic quiz.",
                        mastery_score=0.0,
                        mastery_category=cat,
                        observation_count=0,
                        supporting_evidence={"unassessed": True},
                    )
                )
                continue

            # Priority 8: Moderate progress concepts
            if cat == MasteryCategory.MODERATE:
                candidates.append(
                    PlannedStudyItem(
                        concept_id=cpt.id,
                        concept_name=cpt.name,
                        course_id=cpt.course_id,
                        action_type=PlanActionType.MODERATE_REINFORCEMENT,
                        action_title=f"Reinforce Progress: {cpt.name}",
                        priority=8,
                        estimated_minutes=self.DURATION_REINFORCE,
                        reason=f"Moderate mastery ({score:.2f}). Additional practice will consolidate understanding.",
                        mastery_score=score,
                        mastery_category=cat,
                        observation_count=obs,
                        supporting_evidence={"mastery_score": score},
                    )
                )

        # 6. Sort deterministically by (priority, observation_count)
        candidates.sort(key=lambda item: (item.priority, item.observation_count))

        # 7. Apply time budget and max_items bounds
        selected_items: list[PlannedStudyItem] = []
        allocated_time = 0

        for item in candidates:
            if len(selected_items) >= max_items:
                break
            if time_budget_minutes is not None:
                if allocated_time + item.estimated_minutes > time_budget_minutes:
                    # Check if there is another smaller item that fits
                    continue
            selected_items.append(item)
            allocated_time += item.estimated_minutes

        return AdaptiveStudyPlan(
            learner_id=learner_id,
            course_id=course_id,
            target_concept_id=target_concept_id,
            target_concept_name=target_name,
            time_budget_minutes=time_budget_minutes,
            allocated_minutes=allocated_time,
            items=selected_items,
            total_actions=len(selected_items),
            generated_at=now.isoformat(),
        )
