"""Authoritative SQLite repository for StudyOS learner profile, concepts, mastery, and review tasks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from typing import Sequence
import uuid

from studyos.learner.graph import ConceptGraph, CycleDetectedError
from studyos.learner.mastery import calculate_concept_mastery
from studyos.learner.models import (
    Concept,
    ConceptMastery,
    ConceptProvenance,
    ConceptRelationship,
    ConceptReviewStatus,
    LearnerEvent,
    LearnerEventType,
    LearnerProfile,
    MasteryCategory,
    RelationshipProvenance,
    RelationshipStatus,
    RelationshipType,
    ReviewRating,
    ReviewStatus,
    ReviewTask,
)
from studyos.learner.scheduler import (
    calculate_retrievability,
    compute_fsrs_schedule,
    compute_next_schedule,
    is_review_due,
    is_review_overdue,
    map_grade_to_rating,
)
from studyos.storage.db import Database

logger = logging.getLogger(__name__)


class LearnerRepository:
    """Manages transactional persistence for learner state, concept catalog, and review queues."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------------------------------------------------------------------------
    # Learner Profile Operations
    # ---------------------------------------------------------------------------

    def get_or_create_profile(
        self,
        learner_id: str = "learner_default",
        name: str = "Local Learner",
    ) -> LearnerProfile:
        """Retrieve existing learner profile or create default profile idempotently."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT id, name, created_at, updated_at FROM learner_profile WHERE id = ?;",
                (learner_id,),
            ).fetchone()
            if row:
                return LearnerProfile(
                    id=row["id"],
                    name=row["name"],
                    created_at=str(row["created_at"]) if row["created_at"] else None,
                    updated_at=str(row["updated_at"]) if row["updated_at"] else None,
                )

            conn.execute(
                "INSERT INTO learner_profile (id, name) VALUES (?, ?);",
                (learner_id, name),
            )
            return LearnerProfile(id=learner_id, name=name)

    def get_profile(self, learner_id: str) -> LearnerProfile | None:
        """Retrieve profile by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT id, name, created_at, updated_at FROM learner_profile WHERE id = ?;",
                (learner_id,),
            ).fetchone()
            if not row:
                return None
            return LearnerProfile(
                id=row["id"],
                name=row["name"],
                created_at=str(row["created_at"]) if row["created_at"] else None,
                updated_at=str(row["updated_at"]) if row["updated_at"] else None,
            )

    # ---------------------------------------------------------------------------
    # Concept Catalog Operations
    # ---------------------------------------------------------------------------

    def get_or_create_concept(
        self,
        course_id: str,
        name: str,
        description: str | None = None,
        review_status: ConceptReviewStatus | str = ConceptReviewStatus.ACCEPTED,
        provenance: ConceptProvenance | str = ConceptProvenance.ASSESSMENT_DERIVED,
        learning_objectives: list[str] | None = None,
    ) -> Concept:
        """Retrieve existing concept within course or insert new concept (unique on course_id, normalized_name)."""
        from studyos.learner.concept_mapper import normalize_concept_name

        norm = normalize_concept_name(name)
        status_val = ConceptReviewStatus(review_status).value
        prov_val = ConceptProvenance(provenance).value
        objs_json = json.dumps(learning_objectives) if learning_objectives else None

        with self.db.transaction() as conn:
            # Verify course exists
            course = conn.execute("SELECT id FROM courses WHERE id = ?;", (course_id,)).fetchone()
            if not course:
                raise ValueError(f"Course '{course_id}' does not exist.")

            row = conn.execute(
                """
                SELECT * FROM concepts
                WHERE course_id = ? AND normalized_name = ?;
                """,
                (course_id, norm),
            ).fetchone()
            if row:
                return self._row_to_concept(row)

            cid = f"cpt_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO concepts (id, course_id, name, normalized_name, description, review_status, provenance, learning_objectives)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (cid, course_id, name.strip(), norm, description, status_val, prov_val, objs_json),
            )
            created_row = conn.execute("SELECT * FROM concepts WHERE id = ?;", (cid,)).fetchone()
            return self._row_to_concept(created_row)

    def get_concept(self, concept_id: str) -> Concept | None:
        """Retrieve concept by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM concepts WHERE id = ?;",
                (concept_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_concept(row)

    def list_concepts(
        self,
        course_id: str | None = None,
        status: ConceptReviewStatus | str | None = None,
    ) -> list[Concept]:
        """List concepts, optionally filtered by course and review status."""
        query = "SELECT * FROM concepts WHERE 1=1"
        params: list[Any] = []
        if course_id:
            query += " AND course_id = ?"
            params.append(course_id)
        if status:
            query += " AND review_status = ?"
            params.append(ConceptReviewStatus(status).value)
        query += " ORDER BY name ASC;"

        with self.db.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_concept(r) for r in rows]

    @staticmethod
    def _row_to_concept(row) -> Concept:
        learning_objs = []
        if "learning_objectives" in row.keys() and row["learning_objectives"]:
            try:
                learning_objs = json.loads(row["learning_objectives"])
            except Exception:
                learning_objs = []
        rev_status = (
            ConceptReviewStatus(row["review_status"])
            if ("review_status" in row.keys() and row["review_status"])
            else ConceptReviewStatus.ACCEPTED
        )
        prov = (
            ConceptProvenance(row["provenance"])
            if ("provenance" in row.keys() and row["provenance"])
            else ConceptProvenance.ASSESSMENT_DERIVED
        )

        return Concept(
            id=row["id"],
            course_id=row["course_id"],
            name=row["name"],
            normalized_name=row["normalized_name"],
            description=row["description"],
            review_status=rev_status,
            provenance=prov,
            learning_objectives=learning_objs,
            created_at=str(row["created_at"]) if row["created_at"] else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] else None,
        )

    def map_question_to_concept(self, question_id: str, concept_id: str) -> None:
        """Link an assessment question to a concept."""
        with self.db.transaction() as conn:
            # Verify concept and question belong to same course
            q_row = conn.execute(
                """
                SELECT eq.id, e.course_id
                FROM exam_questions eq
                JOIN exams e ON eq.exam_id = e.id
                WHERE eq.id = ?;
                """,
                (question_id,),
            ).fetchone()
            if not q_row:
                raise ValueError(f"Question '{question_id}' not found.")

            c_row = conn.execute("SELECT course_id FROM concepts WHERE id = ?;", (concept_id,)).fetchone()
            if not c_row:
                raise ValueError(f"Concept '{concept_id}' not found.")

            if q_row["course_id"] != c_row["course_id"]:
                raise ValueError(
                    f"Cross-course mapping rejected: question course '{q_row['course_id']}' != concept course '{c_row['course_id']}'."
                )

            conn.execute(
                """
                INSERT OR IGNORE INTO question_concepts (question_id, concept_id)
                VALUES (?, ?);
                """,
                (question_id, concept_id),
            )

    def get_concepts_for_question(self, question_id: str) -> list[Concept]:
        """Retrieve concepts mapped to a question."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.course_id, c.name, c.normalized_name, c.description, c.created_at, c.updated_at
                FROM question_concepts qc
                JOIN concepts c ON qc.concept_id = c.id
                WHERE qc.question_id = ?
                ORDER BY c.name ASC;
                """,
                (question_id,),
            ).fetchall()
            return [
                Concept(
                    id=r["id"],
                    course_id=r["course_id"],
                    name=r["name"],
                    normalized_name=r["normalized_name"],
                    description=r["description"],
                    created_at=str(r["created_at"]) if r["created_at"] else None,
                    updated_at=str(r["updated_at"]) if r["updated_at"] else None,
                )
                for r in rows
            ]

    def map_concept_to_chunk(self, concept_id: str, chunk_id: str) -> None:
        """Associate a source chunk with a concept (same course enforced)."""
        with self.db.transaction() as conn:
            c_row = conn.execute("SELECT course_id FROM concepts WHERE id = ?;", (concept_id,)).fetchone()
            if not c_row:
                raise ValueError(f"Concept '{concept_id}' not found.")

            chk_row = conn.execute("SELECT course_id FROM chunks WHERE id = ?;", (chunk_id,)).fetchone()
            if not chk_row:
                raise ValueError(f"Chunk '{chunk_id}' not found.")

            if c_row["course_id"] != chk_row["course_id"]:
                raise ValueError(
                    f"Cross-course mapping rejected: concept course '{c_row['course_id']}' != chunk course '{chk_row['course_id']}'."
                )

            conn.execute(
                "INSERT OR IGNORE INTO concept_chunks (concept_id, chunk_id) VALUES (?, ?);",
                (concept_id, chunk_id),
            )

    def get_chunks_for_concept(self, concept_id: str) -> list[dict]:
        """Retrieve chunks mapped to a concept, distinguishing live vs stale."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT cc.chunk_id, c.document_id, c.content, c.location_type, c.location_index, c.location_title
                FROM concept_chunks cc
                LEFT JOIN chunks c ON cc.chunk_id = c.id
                WHERE cc.concept_id = ?;
                """,
                (concept_id,),
            ).fetchall()
            res = []
            for r in rows:
                is_live = r["chunk_id"] is not None and r["content"] is not None
                res.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "document_id": r["document_id"],
                        "location_title": r["location_title"],
                        "location_index": r["location_index"],
                        "location_type": r["location_type"],
                        "excerpt": (r["content"][:200] if r["content"] else None),
                        "is_live": is_live,
                        "is_stale": not is_live,
                    }
                )
            return res

    # ---------------------------------------------------------------------------
    # Learner Events Operations
    # ---------------------------------------------------------------------------

    def has_event_for_attempt_question(self, attempt_id: str, question_id: str) -> bool:
        """Check if a graded event has already been recorded for this attempt and question."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM learner_events
                WHERE reference_id = ? AND question_id = ?;
                """,
                (attempt_id, question_id),
            ).fetchone()
            return row is not None

    def record_event(
        self,
        learner_id: str,
        course_id: str,
        event_type: str,
        reference_id: str | None = None,
        concept_id: str | None = None,
        question_id: str | None = None,
        score_achieved: float | None = None,
        score_max: float | None = None,
        is_correct: bool | None = None,
        details: dict | None = None,
    ) -> LearnerEvent:
        """Record an activity event atomically."""
        eid = f"evt_{uuid.uuid4().hex[:12]}"
        corr_val = 1 if is_correct is True else (0 if is_correct is False else None)
        det_json = json.dumps(details) if details else None

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO learner_events (
                    id, learner_id, course_id, concept_id, event_type, reference_id,
                    question_id, score_achieved, score_max, is_correct, details_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (
                    eid,
                    learner_id,
                    course_id,
                    concept_id,
                    event_type,
                    reference_id,
                    question_id,
                    score_achieved,
                    score_max,
                    corr_val,
                    det_json,
                ),
            )
        return LearnerEvent(
            id=eid,
            learner_id=learner_id,
            course_id=course_id,
            concept_id=concept_id,
            event_type=event_type,
            reference_id=reference_id,
            question_id=question_id,
            score_achieved=score_achieved,
            score_max=score_max,
            is_correct=is_correct,
            details_json=det_json,
        )

    def update_or_record_finalized_grade(
        self,
        learner_id: str,
        course_id: str,
        attempt_id: str,
        question_id: str,
        concept_id: str,
        awarded_score: float,
        max_score: float,
        is_correct: bool,
    ) -> None:
        """Update existing pending event to finalized grade or insert finalized event without duplicating."""
        corr_val = 1 if is_correct else 0
        with self.db.transaction() as conn:
            existing = conn.execute(
                """
                SELECT id FROM learner_events
                WHERE reference_id = ? AND question_id = ?;
                """,
                (attempt_id, question_id),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE learner_events
                    SET event_type = ?, score_achieved = ?, score_max = ?, is_correct = ?, concept_id = ?
                    WHERE id = ?;
                    """,
                    (
                        LearnerEventType.MANUAL_GRADE_FINALIZED.value,
                        awarded_score,
                        max_score,
                        corr_val,
                        concept_id,
                        existing["id"],
                    ),
                )
            else:
                eid = f"evt_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO learner_events (
                        id, learner_id, course_id, concept_id, event_type, reference_id,
                        question_id, score_achieved, score_max, is_correct, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                    """,
                    (
                        eid,
                        learner_id,
                        course_id,
                        concept_id,
                        LearnerEventType.MANUAL_GRADE_FINALIZED.value,
                        attempt_id,
                        question_id,
                        awarded_score,
                        max_score,
                        corr_val,
                    ),
                )

    def list_events(
        self,
        learner_id: str,
        course_id: str | None = None,
        limit: int = 50,
        since: str | None = None,
        until: str | None = None,
    ) -> list[LearnerEvent]:
        """List chronological learner activity history."""
        sql = "SELECT * FROM learner_events WHERE learner_id = ?"
        params: list[str | int] = [learner_id]

        if course_id:
            sql += " AND course_id = ?"
            params.append(course_id)
        if since:
            sql += " AND created_at >= ?"
            params.append(since)
        if until:
            sql += " AND created_at <= ?"
            params.append(until)

        sql += " ORDER BY created_at DESC LIMIT ?;"
        params.append(limit)

        with self.db.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                LearnerEvent(
                    id=r["id"],
                    learner_id=r["learner_id"],
                    course_id=r["course_id"],
                    concept_id=r["concept_id"],
                    event_type=r["event_type"],
                    reference_id=r["reference_id"],
                    question_id=r["question_id"],
                    score_achieved=r["score_achieved"],
                    score_max=r["score_max"],
                    is_correct=(bool(r["is_correct"]) if r["is_correct"] is not None else None),
                    details_json=r["details_json"],
                    created_at=str(r["created_at"]) if r["created_at"] else None,
                )
                for r in rows
            ]

    # ---------------------------------------------------------------------------
    # Concept Mastery Operations
    # ---------------------------------------------------------------------------

    def get_concept_mastery(self, learner_id: str, concept_id: str) -> ConceptMastery | None:
        """Retrieve concept mastery record."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT cm.*, c.name as concept_name
                FROM concept_mastery cm
                JOIN concepts c ON cm.concept_id = c.id
                WHERE cm.learner_id = ? AND cm.concept_id = ?;
                """,
                (learner_id, concept_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_concept_mastery(row)

    def list_concept_mastery(
        self,
        learner_id: str,
        course_id: str | None = None,
    ) -> list[ConceptMastery]:
        """List all concept mastery records for a learner."""
        with self.db.get_connection() as conn:
            if course_id:
                rows = conn.execute(
                    """
                    SELECT cm.*, c.name as concept_name
                    FROM concept_mastery cm
                    JOIN concepts c ON cm.concept_id = c.id
                    WHERE cm.learner_id = ? AND cm.course_id = ?
                    ORDER BY cm.mastery_score ASC, cm.observation_count ASC;
                    """,
                    (learner_id, course_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT cm.*, c.name as concept_name
                    FROM concept_mastery cm
                    JOIN concepts c ON cm.concept_id = c.id
                    WHERE cm.learner_id = ?
                    ORDER BY cm.course_id, cm.mastery_score ASC;
                    """,
                    (learner_id,),
                ).fetchall()
            return [self._row_to_concept_mastery(r) for r in rows]

    def upsert_concept_mastery(
        self,
        mastery: ConceptMastery,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Upsert concept mastery state atomically."""
        mid = mastery.id or f"cpm_{uuid.uuid4().hex[:12]}"
        sql = """
            INSERT INTO concept_mastery (
                id, learner_id, concept_id, course_id, mastery_score, mastery_category,
                observation_count, correct_count, incorrect_count, uncertainty, last_assessed_at,
                next_review_due, interval_days, ease_factor, repetition_count,
                stability, difficulty, retrievability, scheduler_policy, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                mastery_score = excluded.mastery_score,
                mastery_category = excluded.mastery_category,
                observation_count = excluded.observation_count,
                correct_count = excluded.correct_count,
                incorrect_count = excluded.incorrect_count,
                uncertainty = excluded.uncertainty,
                last_assessed_at = excluded.last_assessed_at,
                next_review_due = excluded.next_review_due,
                interval_days = excluded.interval_days,
                ease_factor = excluded.ease_factor,
                repetition_count = excluded.repetition_count,
                stability = excluded.stability,
                difficulty = excluded.difficulty,
                retrievability = excluded.retrievability,
                scheduler_policy = excluded.scheduler_policy,
                updated_at = CURRENT_TIMESTAMP;
        """
        params = (
            mid,
            mastery.learner_id,
            mastery.concept_id,
            mastery.course_id,
            mastery.mastery_score,
            mastery.mastery_category.value,
            mastery.observation_count,
            mastery.correct_count,
            mastery.incorrect_count,
            mastery.uncertainty,
            mastery.last_assessed_at,
            mastery.next_review_due,
            mastery.interval_days,
            mastery.ease_factor,
            mastery.repetition_count,
            mastery.stability,
            mastery.difficulty,
            mastery.retrievability,
            mastery.scheduler_policy,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.db.transaction() as c:
                c.execute(sql, params)

    def recompute_mastery_for_concept(
        self,
        learner_id: str,
        concept_id: str,
        course_id: str | None = None,
    ) -> ConceptMastery:
        """Recompute mastery from authoritative graded events using normalized per-observation scores."""
        with self.db.get_connection() as conn:
            # Get concept and course
            c_row = conn.execute(
                "SELECT id, course_id, name FROM concepts WHERE id = ?;",
                (concept_id,),
            ).fetchone()
            if not c_row:
                raise ValueError(f"Concept '{concept_id}' not found.")
            actual_course_id = c_row["course_id"]
            concept_name = c_row["name"]

            # Query all learner activity events for this concept
            events = conn.execute(
                """
                SELECT id, score_achieved, score_max, is_correct, event_type, created_at
                FROM learner_events
                WHERE learner_id = ? AND concept_id = ?
                ORDER BY created_at ASC;
                """,
                (learner_id, concept_id),
            ).fetchall()

            # Retrieve existing mastery row to retain scheduling intervals
            existing = conn.execute(
                "SELECT * FROM concept_mastery WHERE learner_id = ? AND concept_id = ?;",
                (learner_id, concept_id),
            ).fetchone()

        valid_p_scores: list[float] = []
        last_assessed = None

        for e in events:
            # 1. Exclude pending manual review or unreviewed events
            if e["event_type"] == "pending_manual_review":
                continue
            if e["is_correct"] is None and e["score_achieved"] is None:
                continue

            # Extract scores
            raw_earned = e["score_achieved"]
            raw_possible = e["score_max"]

            # Fallback for synthetic events that specified boolean is_correct without scores
            if raw_earned is None or raw_possible is None:
                if e["is_correct"] is not None:
                    raw_earned = 1.0 if e["is_correct"] else 0.0
                    raw_possible = 1.0
                else:
                    continue

            # Validate finite numerical values
            try:
                s_earned = float(raw_earned)
                s_possible = float(raw_possible)
            except (TypeError, ValueError):
                continue

            if not (math.isfinite(s_earned) and math.isfinite(s_possible)):
                continue

            # score_possible must be strictly positive
            if s_possible <= 0.0:
                continue

            # score_earned must lie in [0, score_possible]
            if s_earned < 0.0 or s_earned > s_possible:
                continue

            p_i = s_earned / s_possible
            valid_p_scores.append(p_i)
            if e["created_at"]:
                last_assessed = str(e["created_at"])

        obs_count = len(valid_p_scores)
        k_successes = sum(valid_p_scores)

        estimate = calculate_concept_mastery(
            observation_count=obs_count,
            success_credits=k_successes,
        )

        interval = float(existing["interval_days"]) if existing else 1.0
        ease = float(existing["ease_factor"]) if existing else 2.5
        reps = int(existing["repetition_count"]) if existing else 0
        due = str(existing["next_review_due"]) if (existing and existing["next_review_due"]) else None

        stability = (
            float(existing["stability"])
            if (existing and "stability" in existing.keys() and existing["stability"] is not None)
            else interval
        )
        difficulty = (
            float(existing["difficulty"])
            if (existing and "difficulty" in existing.keys() and existing["difficulty"] is not None)
            else 5.0
        )
        retrievability = (
            float(existing["retrievability"])
            if (existing and "retrievability" in existing.keys() and existing["retrievability"] is not None)
            else 1.0
        )
        policy = (
            str(existing["scheduler_policy"])
            if (existing and "scheduler_policy" in existing.keys() and existing["scheduler_policy"] is not None)
            else "local-fsrs-v1"
        )

        mastery_obj = ConceptMastery(
            id=existing["id"] if existing else f"cpm_{uuid.uuid4().hex[:12]}",
            learner_id=learner_id,
            concept_id=concept_id,
            course_id=actual_course_id,
            concept_name=concept_name,
            mastery_score=estimate.score,
            mastery_category=estimate.category,
            observation_count=estimate.observation_count,
            correct_count=round(k_successes, 3),
            incorrect_count=round(obs_count - k_successes, 3),
            uncertainty=estimate.uncertainty,
            has_sufficient_evidence=estimate.has_sufficient_evidence,
            explanation_reason=estimate.reason,
            last_assessed_at=last_assessed,
            next_review_due=due,
            interval_days=interval,
            ease_factor=ease,
            repetition_count=reps,
            stability=stability,
            difficulty=difficulty,
            retrievability=retrievability,
            scheduler_policy=policy,
        )
        self.upsert_concept_mastery(mastery_obj)
        return mastery_obj

    # ---------------------------------------------------------------------------
    # Review Task Queue Operations
    # ---------------------------------------------------------------------------

    def create_or_update_review_task(
        self,
        learner_id: str,
        concept_id: str,
        course_id: str,
        due_date: str,
        scheduler_policy: str = "local-fsrs-v1",
        scheduled_interval: float | None = None,
        estimated_retention: float | None = None,
    ) -> ReviewTask:
        """Schedule or reschedule a review task for a concept."""
        with self.db.transaction() as conn:
            # Check for existing scheduled task
            row = conn.execute(
                """
                SELECT id FROM review_tasks
                WHERE learner_id = ? AND concept_id = ? AND status = ?;
                """,
                (learner_id, concept_id, ReviewStatus.SCHEDULED.value),
            ).fetchone()

            if row:
                tid = row["id"]
                conn.execute(
                    """
                    UPDATE review_tasks
                    SET due_date = ?,
                        scheduler_policy = ?,
                        scheduled_interval = COALESCE(?, scheduled_interval),
                        estimated_retention = COALESCE(?, estimated_retention)
                    WHERE id = ?;
                    """,
                    (due_date, scheduler_policy, scheduled_interval, estimated_retention, tid),
                )
            else:
                tid = f"rvt_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO review_tasks (
                        id, learner_id, concept_id, course_id, due_date, status,
                        scheduler_policy, scheduled_interval, estimated_retention
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        tid,
                        learner_id,
                        concept_id,
                        course_id,
                        due_date,
                        ReviewStatus.SCHEDULED.value,
                        scheduler_policy,
                        scheduled_interval or 1.0,
                        estimated_retention or 1.0,
                    ),
                )

        task = self.get_review_task(tid)
        if not task:
            raise RuntimeError(f"Failed to load created review task '{tid}'.")
        return task

    def get_review_task(self, task_id: str) -> ReviewTask | None:
        """Retrieve review task by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT rt.*, c.name as concept_name
                FROM review_tasks rt
                JOIN concepts c ON rt.concept_id = c.id
                WHERE rt.id = ?;
                """,
                (task_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_review_task(row)

    def list_review_tasks(
        self,
        learner_id: str,
        course_id: str | None = None,
        status: ReviewStatus | None = None,
    ) -> list[ReviewTask]:
        """List review tasks."""
        sql = """
            SELECT rt.*, c.name as concept_name
            FROM review_tasks rt
            JOIN concepts c ON rt.concept_id = c.id
            WHERE rt.learner_id = ?
        """
        params: list[str] = [learner_id]
        if course_id:
            sql += " AND rt.course_id = ?"
            params.append(course_id)
        if status:
            sql += " AND rt.status = ?"
            params.append(status.value)

        sql += " ORDER BY rt.due_date ASC;"

        with self.db.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_review_task(r) for r in rows]

    def complete_review_task(
        self,
        task_id: str,
        rating: ReviewRating,
        policy: str | None = None,
    ) -> ReviewTask:
        """Mark a review task completed, update concept schedule, and queue next review."""
        task = self.get_review_task(task_id)
        if not task:
            raise ValueError(f"Review task '{task_id}' not found.")

        # Get concept mastery state to update schedule
        mastery = self.get_concept_mastery(task.learner_id, task.concept_id)
        current_interval = mastery.interval_days if mastery else 1.0
        current_ease = mastery.ease_factor if mastery else 2.5
        reps = mastery.repetition_count if mastery else 0
        current_stability = mastery.stability if mastery else 1.0
        current_difficulty = mastery.difficulty if mastery else 5.0
        selected_policy = policy or (mastery.scheduler_policy if mastery else "local-fsrs-v1")
        last_rev = None
        if mastery and mastery.last_assessed_at:
            try:
                last_rev = datetime.fromisoformat(mastery.last_assessed_at)
            except Exception:
                pass

        schedule = compute_next_schedule(
            current_interval=current_interval,
            current_ease=current_ease,
            repetition_count=reps,
            rating_or_correctness=rating,
            policy=selected_policy,
            stability=current_stability,
            difficulty=current_difficulty,
            last_reviewed_at=last_rev,
        )

        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE review_tasks
                SET status = ?, rating = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (ReviewStatus.COMPLETED.value, rating.value, task_id),
            )

            # Record event
            eid = f"evt_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO learner_events (
                    id, learner_id, course_id, concept_id, event_type, reference_id,
                    is_correct, details_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (
                    eid,
                    task.learner_id,
                    task.course_id,
                    task.concept_id,
                    LearnerEventType.REVIEW_COMPLETED.value,
                    task_id,
                    1 if rating in (ReviewRating.GOOD, ReviewRating.EASY) else 0,
                    json.dumps({"rating": rating.value, "new_interval": schedule.interval_days}),
                ),
            )

            # Update concept mastery schedule
            due_iso = schedule.next_due_date.isoformat()
            if mastery:
                mastery.interval_days = schedule.interval_days
                mastery.ease_factor = schedule.ease_factor
                mastery.repetition_count = schedule.repetition_count
                mastery.stability = schedule.stability
                mastery.difficulty = schedule.difficulty
                mastery.retrievability = schedule.estimated_retention
                mastery.scheduler_policy = schedule.scheduler_policy
                mastery.next_review_due = due_iso
                self.upsert_concept_mastery(mastery, conn=conn)

            # Create next scheduled review task
            next_tid = f"rvt_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO review_tasks (
                    id, learner_id, concept_id, course_id, due_date, status,
                    scheduler_policy, scheduled_interval, estimated_retention
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    next_tid,
                    task.learner_id,
                    task.concept_id,
                    task.course_id,
                    due_iso,
                    ReviewStatus.SCHEDULED.value,
                    schedule.scheduler_policy,
                    schedule.interval_days,
                    schedule.estimated_retention,
                ),
            )

        updated_task = self.get_review_task(task_id)
        if not updated_task:
            raise RuntimeError(f"Failed to load completed review task '{task_id}'.")
        return updated_task

    # ---------------------------------------------------------------------------
    # Concept Relationships (Graph) Operations
    # ---------------------------------------------------------------------------

    def add_concept_relationship(
        self,
        course_id: str,
        source_concept_id: str,
        target_concept_id: str,
        relationship_type: RelationshipType | str,
        provenance: RelationshipProvenance | str = RelationshipProvenance.MANUAL,
        source_evidence: str | None = None,
        confidence: float = 1.0,
        review_status: RelationshipStatus | str = RelationshipStatus.ACCEPTED,
    ) -> ConceptRelationship:
        """Create or update a directed relationship between two concepts in the same course.

        Enforces:
        - Self-referencing edges are strictly rejected.
        - Source and target concepts must both exist and belong to course_id.
        - Accepted prerequisite edges must be strictly acyclic (raises CycleDetectedError).
        """
        if source_concept_id == target_concept_id:
            raise ValueError("Self-referencing relationship is not allowed.")

        rel_type = RelationshipType(relationship_type)
        status = RelationshipStatus(review_status)
        prov = RelationshipProvenance(provenance)

        with self.db.transaction() as conn:
            # 1. Verify concepts exist and belong to course_id
            c_rows = conn.execute(
                "SELECT id, course_id, name FROM concepts WHERE id IN (?, ?);",
                (source_concept_id, target_concept_id),
            ).fetchall()
            found = {r["id"]: r for r in c_rows}
            if source_concept_id not in found:
                raise ValueError(f"Source concept '{source_concept_id}' not found.")
            if target_concept_id not in found:
                raise ValueError(f"Target concept '{target_concept_id}' not found.")

            if found[source_concept_id]["course_id"] != course_id or found[target_concept_id]["course_id"] != course_id:
                raise ValueError(f"Cross-course relationship rejected: concepts must belong to course '{course_id}'.")

            # 2. Cycle validation for accepted prerequisites
            if rel_type == RelationshipType.PREREQUISITE and status == RelationshipStatus.ACCEPTED:
                existing_rels = self.list_concept_relationships(
                    course_id=course_id,
                    status=RelationshipStatus.ACCEPTED,
                    rel_type=RelationshipType.PREREQUISITE,
                    conn=conn,
                )
                graph = ConceptGraph(course_id, existing_rels)
                if graph.would_create_cycle(source_concept_id, target_concept_id):
                    raise CycleDetectedError(
                        f"Cycle detected: adding prerequisite from '{source_concept_id}' to "
                        f"'{target_concept_id}' would create a directed cycle in course '{course_id}'."
                    )

            rid = f"cr_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO concept_relationships (
                    id, course_id, source_concept_id, target_concept_id,
                    relationship_type, provenance, source_evidence, confidence, review_status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(course_id, source_concept_id, target_concept_id, relationship_type) DO UPDATE SET
                    provenance = excluded.provenance,
                    source_evidence = excluded.source_evidence,
                    confidence = excluded.confidence,
                    review_status = excluded.review_status,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (
                    rid,
                    course_id,
                    source_concept_id,
                    target_concept_id,
                    rel_type.value,
                    prov.value,
                    source_evidence,
                    max(0.0, min(1.0, confidence)),
                    status.value,
                ),
            )

            # Retrieve final row
            final_row = conn.execute(
                """
                SELECT cr.*, s.name as source_concept_name, t.name as target_concept_name
                FROM concept_relationships cr
                JOIN concepts s ON cr.source_concept_id = s.id
                JOIN concepts t ON cr.target_concept_id = t.id
                WHERE cr.course_id = ? AND cr.source_concept_id = ? AND cr.target_concept_id = ? AND cr.relationship_type = ?;
                """,
                (course_id, source_concept_id, target_concept_id, rel_type.value),
            ).fetchone()
            return self._row_to_concept_relationship(final_row)

    def update_relationship_status(
        self,
        relationship_id: str,
        status: RelationshipStatus | str,
    ) -> ConceptRelationship:
        """Update review status of a relationship, validating cycles if accepting a prerequisite."""
        new_status = RelationshipStatus(status)
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM concept_relationships WHERE id = ?;",
                (relationship_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Concept relationship '{relationship_id}' not found.")

            rel_type = RelationshipType(row["relationship_type"])
            course_id = row["course_id"]
            src_id = row["source_concept_id"]
            tgt_id = row["target_concept_id"]

            if rel_type == RelationshipType.PREREQUISITE and new_status == RelationshipStatus.ACCEPTED:
                existing_rels = self.list_concept_relationships(
                    course_id=course_id,
                    status=RelationshipStatus.ACCEPTED,
                    rel_type=RelationshipType.PREREQUISITE,
                    conn=conn,
                )
                filtered_rels = [r for r in existing_rels if r.id != relationship_id]
                graph = ConceptGraph(course_id, filtered_rels)
                if graph.would_create_cycle(src_id, tgt_id):
                    raise CycleDetectedError(
                        f"Cannot accept relationship '{relationship_id}': creates a cycle in course '{course_id}'."
                    )

            conn.execute(
                """
                UPDATE concept_relationships
                SET review_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (new_status.value, relationship_id),
            )

        updated = self.get_concept_relationship(relationship_id)
        if not updated:
            raise RuntimeError(f"Failed to load relationship '{relationship_id}'.")
        return updated

    def delete_concept_relationship(self, relationship_id: str) -> bool:
        """Delete a concept relationship edge."""
        with self.db.transaction() as conn:
            res = conn.execute(
                "DELETE FROM concept_relationships WHERE id = ?;",
                (relationship_id,),
            )
            return res.rowcount > 0

    def get_concept_relationship(self, relationship_id: str) -> ConceptRelationship | None:
        """Retrieve relationship by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT cr.*, s.name as source_concept_name, t.name as target_concept_name
                FROM concept_relationships cr
                JOIN concepts s ON cr.source_concept_id = s.id
                JOIN concepts t ON cr.target_concept_id = t.id
                WHERE cr.id = ?;
                """,
                (relationship_id,),
            ).fetchone()
            return self._row_to_concept_relationship(row) if row else None

    def list_concept_relationships(
        self,
        course_id: str,
        status: RelationshipStatus | str | None = None,
        rel_type: RelationshipType | str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[ConceptRelationship]:
        """List concept relationships within a course, optionally filtered by status or type."""
        query = """
            SELECT cr.*, s.name as source_concept_name, t.name as target_concept_name
            FROM concept_relationships cr
            JOIN concepts s ON cr.source_concept_id = s.id
            JOIN concepts t ON cr.target_concept_id = t.id
            WHERE cr.course_id = ?
        """
        params: list[Any] = [course_id]

        if status:
            stat_val = RelationshipStatus(status).value
            query += " AND cr.review_status = ?"
            params.append(stat_val)

        if rel_type:
            type_val = RelationshipType(rel_type).value
            query += " AND cr.relationship_type = ?"
            params.append(type_val)

        query += " ORDER BY cr.created_at ASC;"

        if conn is not None:
            rows = conn.execute(query, params).fetchall()
        else:
            with self.db.get_connection() as c:
                rows = c.execute(query, params).fetchall()

        return [self._row_to_concept_relationship(r) for r in rows]

    def get_concept_graph(self, course_id: str) -> ConceptGraph:
        """Construct an in-memory ConceptGraph of accepted relationships for course_id."""
        rels = self.list_concept_relationships(course_id=course_id, status=RelationshipStatus.ACCEPTED)
        return ConceptGraph(course_id, rels)

    def record_discovery_run(
        self,
        course_id: str,
        document_id: str | None,
        status: str,
        concepts_discovered: int,
        chunks_processed: int,
        error_message: str | None = None,
    ) -> str:
        """Log a concept discovery run."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO concept_discovery_runs (
                    id, course_id, document_id, status, concepts_discovered,
                    chunks_processed, error_message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                """,
                (
                    run_id,
                    course_id,
                    document_id,
                    status,
                    concepts_discovered,
                    chunks_processed,
                    error_message,
                ),
            )
        return run_id

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _row_to_concept_relationship(row) -> ConceptRelationship:
        return ConceptRelationship(
            id=row["id"],
            course_id=row["course_id"],
            source_concept_id=row["source_concept_id"],
            target_concept_id=row["target_concept_id"],
            relationship_type=RelationshipType(row["relationship_type"]),
            provenance=RelationshipProvenance(row["provenance"]),
            source_evidence=row["source_evidence"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else 1.0,
            review_status=RelationshipStatus(row["review_status"]),
            created_at=str(row["created_at"]) if row["created_at"] else None,
            updated_at=str(row["updated_at"]) if row["updated_at"] else None,
            source_concept_name=row["source_concept_name"] if "source_concept_name" in row.keys() else None,
            target_concept_name=row["target_concept_name"] if "target_concept_name" in row.keys() else None,
        )

    @staticmethod
    def _row_to_concept_mastery(row) -> ConceptMastery:
        obs = int(row["observation_count"])
        corr = float(row["correct_count"])
        incorr = float(row["incorrect_count"])
        cat = MasteryCategory(row["mastery_category"])
        has_suff = obs >= 3

        estimate = calculate_concept_mastery(obs, success_credits=corr)
        uncertainty = (
            float(row["uncertainty"])
            if ("uncertainty" in row.keys() and row["uncertainty"] is not None)
            else estimate.uncertainty
        )

        stability = (
            float(row["stability"])
            if ("stability" in row.keys() and row["stability"] is not None)
            else float(row["interval_days"])
        )
        difficulty = (
            float(row["difficulty"])
            if ("difficulty" in row.keys() and row["difficulty"] is not None)
            else 5.0
        )
        retrievability = (
            float(row["retrievability"])
            if ("retrievability" in row.keys() and row["retrievability"] is not None)
            else 1.0
        )
        policy = (
            str(row["scheduler_policy"])
            if ("scheduler_policy" in row.keys() and row["scheduler_policy"] is not None)
            else "local-fsrs-v1"
        )

        return ConceptMastery(
            id=row["id"],
            learner_id=row["learner_id"],
            concept_id=row["concept_id"],
            course_id=row["course_id"],
            concept_name=row["concept_name"] if "concept_name" in row.keys() else None,
            mastery_score=float(row["mastery_score"]),
            mastery_category=cat,
            observation_count=obs,
            correct_count=corr,
            incorrect_count=incorr,
            uncertainty=uncertainty,
            has_sufficient_evidence=has_suff,
            explanation_reason=estimate.reason,
            last_assessed_at=str(row["last_assessed_at"]) if row["last_assessed_at"] else None,
            next_review_due=str(row["next_review_due"]) if row["next_review_due"] else None,
            interval_days=float(row["interval_days"]),
            ease_factor=float(row["ease_factor"]),
            repetition_count=int(row["repetition_count"]),
            stability=stability,
            difficulty=difficulty,
            retrievability=retrievability,
            scheduler_policy=policy,
            updated_at=str(row["updated_at"]) if row["updated_at"] else None,
        )

    @staticmethod
    def _row_to_review_task(row) -> ReviewTask:
        due_str = str(row["due_date"])
        status = ReviewStatus(row["status"])
        rating = ReviewRating(row["rating"]) if row["rating"] else None

        # Check due / overdue relative to now
        is_due = False
        is_over = False
        try:
            due_dt = datetime.fromisoformat(due_str)
            if status == ReviewStatus.SCHEDULED:
                is_due = is_review_due(due_dt)
                is_over = is_review_overdue(due_dt)
        except Exception:
            pass

        sched_policy = (
            str(row["scheduler_policy"])
            if ("scheduler_policy" in row.keys() and row["scheduler_policy"])
            else "local-fsrs-v1"
        )
        sched_interval = (
            float(row["scheduled_interval"])
            if ("scheduled_interval" in row.keys() and row["scheduled_interval"] is not None)
            else 1.0
        )
        est_ret = (
            float(row["estimated_retention"])
            if ("estimated_retention" in row.keys() and row["estimated_retention"] is not None)
            else 1.0
        )

        return ReviewTask(
            id=row["id"],
            learner_id=row["learner_id"],
            concept_id=row["concept_id"],
            course_id=row["course_id"],
            concept_name=row["concept_name"] if "concept_name" in row.keys() else None,
            due_date=due_str,
            status=status,
            is_due=is_due,
            is_overdue=is_over,
            completed_at=str(row["completed_at"]) if row["completed_at"] else None,
            rating=rating,
            scheduler_policy=sched_policy,
            scheduled_interval=sched_interval,
            estimated_retention=est_ret,
            created_at=str(row["created_at"]) if row["created_at"] else None,
        )
