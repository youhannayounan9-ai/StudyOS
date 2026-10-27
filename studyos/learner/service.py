"""Public service facade for StudyOS learner modeling, progress analytics, and review scheduling."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Sequence

from studyos.assessment.models import ExamResult, GradingStatus
from studyos.assessment.repository import AssessmentRepository
from studyos.core.config import AppConfig, get_default_config
from studyos.inference.protocol import ChatProvider
from studyos.learner.concept_mapper import ConceptMapper
from studyos.learner.models import (
    Concept,
    ConceptMastery,
    ConceptProgressSummary,
    CourseProgressSummary,
    LearnerEvent,
    LearnerEventType,
    LearnerProfile,
    LearnerProgressSummary,
    MasteryCategory,
    ReviewRating,
    ReviewStatus,
    ReviewTask,
    StudyRecommendation,
)
from studyos.learner.repository import LearnerRepository
from studyos.learner.scheduler import compute_next_schedule, is_review_due, is_review_overdue
from studyos.storage.db import Database

logger = logging.getLogger(__name__)


class LearnerService:
    """Unified service orchestrating learner state, concept catalog, mastery, and revision."""

    def __init__(
        self,
        db: Database,
        chat_provider: ChatProvider | None = None,
        config: AppConfig | None = None,
        repository: LearnerRepository | None = None,
        assessment_repo: AssessmentRepository | None = None,
    ) -> None:
        self.db = db
        self.chat_provider = chat_provider
        self.config = config or get_default_config()
        self.repository = repository or LearnerRepository(db)
        self.assessment_repo = assessment_repo or AssessmentRepository(db)
        self.concept_mapper = ConceptMapper(chat_provider=chat_provider)

    # ---------------------------------------------------------------------------
    # Profile & Concept Catalog
    # ---------------------------------------------------------------------------

    def get_or_create_profile(
        self,
        learner_id: str = "learner_default",
        name: str = "Local Learner",
    ) -> LearnerProfile:
        """Retrieve or create the local learner profile."""
        return self.repository.get_or_create_profile(learner_id=learner_id, name=name)

    def create_concept(
        self,
        course_id: str,
        name: str,
        description: str | None = None,
    ) -> Concept:
        """Create or retrieve a normalized, course-scoped concept."""
        return self.repository.get_or_create_concept(
            course_id=course_id,
            name=name,
            description=description,
        )

    def get_concept(self, concept_id: str) -> Concept | None:
        """Retrieve concept by ID."""
        return self.repository.get_concept(concept_id)

    def list_concepts(self, course_id: str | None = None) -> list[Concept]:
        """List concepts, optionally scoped to a course."""
        return self.repository.list_concepts(course_id=course_id)

    def map_question_to_concept(self, question_id: str, concept_id: str) -> None:
        """Associate an assessment question with a concept."""
        self.repository.map_question_to_concept(question_id, concept_id)

    def get_concepts_for_question(self, question_id: str) -> list[Concept]:
        """Retrieve concepts mapped to a question."""
        return self.repository.get_concepts_for_question(question_id)

    # ---------------------------------------------------------------------------
    # Assessment-to-Learner Integration (Idempotent & Safe)
    # ---------------------------------------------------------------------------

    def process_exam_attempt_results(
        self,
        attempt_id: str,
        exam_result: ExamResult,
        learner_id: str = "learner_default",
    ) -> None:
        """Process graded question attempts into the learner model idempotently.

        Guarantees:
        - Pending manual-review questions are strictly excluded from mastery calculation.
        - Repeated calls for the same attempt do not duplicate events or double-count.
        - Questions are mapped to course concepts based on trusted metadata.
        - Associated concept masteries and review queues are updated consistently.
        """
        # Ensure learner profile exists
        self.repository.get_or_create_profile(learner_id=learner_id)

        # Retrieve internal exam to inspect trusted source metadata
        internal_exam = self.assessment_repo.get_exam(exam_result.exam_id)
        q_map = {q.id: q for q in internal_exam.questions} if internal_exam else {}

        now = datetime.now(timezone.utc)

        for qr in exam_result.questions:
            # 1. Resolve or extract concept associations
            concepts = self.repository.get_concepts_for_question(qr.question_id)
            if not concepts:
                # Extract candidate labels from question metadata
                orig_q = q_map.get(qr.question_id)
                labels = (
                    self.concept_mapper.extract_concept_labels_from_question(
                        orig_q, exam_topic=exam_result.title
                    )
                    if orig_q
                    else ["Unclassified"]
                )
                for label in labels:
                    cpt = self.repository.get_or_create_concept(
                        course_id=exam_result.course_id,
                        name=label,
                    )
                    self.repository.map_question_to_concept(qr.question_id, cpt.id)
                    concepts.append(cpt)

                    # Map live source chunks to concept
                    if orig_q and orig_q.sources:
                        for src in orig_q.sources:
                            if src.chunk_id and not src.is_stale:
                                with self.db.get_connection() as conn:
                                    chk_exists = conn.execute(
                                        "SELECT course_id FROM chunks WHERE id = ?;",
                                        (src.chunk_id,),
                                    ).fetchone()
                                    if chk_exists and chk_exists["course_id"] == exam_result.course_id:
                                        self.repository.map_concept_to_chunk(cpt.id, src.chunk_id)

            # 2. Check if this question was already recorded for this attempt
            already_recorded = self.repository.has_event_for_attempt_question(
                attempt_id=attempt_id,
                question_id=qr.question_id,
            )

            # 3. Handle pending vs graded status
            is_pending = qr.status == GradingStatus.PENDING_MANUAL_REVIEW or qr.score is None
            if is_pending:
                # Do not count pending question as negative evidence or graded attempt
                if not already_recorded:
                    for cpt in concepts:
                        self.repository.record_event(
                            learner_id=learner_id,
                            course_id=exam_result.course_id,
                            concept_id=cpt.id,
                            event_type=qr.status.value,
                            reference_id=attempt_id,
                            question_id=qr.question_id,
                            score_achieved=None,
                            score_max=qr.max_score,
                            is_correct=None,
                            details={"status": "pending_manual_review"},
                        )
                continue

            # Question is graded: record event if not already present
            if not already_recorded:
                for cpt in concepts:
                    self.repository.record_event(
                        learner_id=learner_id,
                        course_id=exam_result.course_id,
                        concept_id=cpt.id,
                        event_type=LearnerEventType.EXAM_QUESTION_GRADED.value,
                        reference_id=attempt_id,
                        question_id=qr.question_id,
                        score_achieved=qr.score,
                        score_max=qr.max_score,
                        is_correct=qr.is_correct,
                        details={"question_index": qr.question_index, "feedback": qr.feedback},
                    )

            # 4. Recompute mastery and schedule review for each mapped concept
            for cpt in concepts:
                mastery = self.repository.recompute_mastery_for_concept(
                    learner_id=learner_id,
                    concept_id=cpt.id,
                    course_id=exam_result.course_id,
                )

                # Update review schedule based on graded correctness
                sched = compute_next_schedule(
                    current_interval=mastery.interval_days,
                    current_ease=mastery.ease_factor,
                    repetition_count=mastery.repetition_count,
                    rating_or_correctness=bool(qr.is_correct),
                    now=now,
                )
                due_iso = sched.next_due_date.isoformat()
                mastery.interval_days = sched.interval_days
                mastery.ease_factor = sched.ease_factor
                mastery.repetition_count = sched.repetition_count
                mastery.next_review_due = due_iso
                self.repository.upsert_concept_mastery(mastery)

                # Ensure active review task is scheduled in the queue
                self.repository.create_or_update_review_task(
                    learner_id=learner_id,
                    concept_id=cpt.id,
                    course_id=exam_result.course_id,
                    due_date=due_iso,
                )

    def process_manual_review_finalized(
        self,
        attempt_id: str,
        question_id: str,
        awarded_score: float,
        max_score: float,
        is_correct: bool,
        learner_id: str = "learner_default",
    ) -> None:
        """Handle finalized instructor manual review for a previously pending short-answer question."""
        concepts = self.repository.get_concepts_for_question(question_id)
        if not concepts:
            # Look up course from question
            with self.db.get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT e.course_id, e.title
                    FROM exam_questions eq
                    JOIN exams e ON eq.exam_id = e.id
                    WHERE eq.id = ?;
                    """,
                    (question_id,),
                ).fetchone()
                if row:
                    course_id = row["course_id"]
                    cpt = self.repository.get_or_create_concept(course_id, row["title"] or "Unclassified")
                    self.repository.map_question_to_concept(question_id, cpt.id)
                    concepts = [cpt]
                else:
                    return

        course_id = concepts[0].course_id
        for cpt in concepts:
            self.repository.update_or_record_finalized_grade(
                learner_id=learner_id,
                course_id=course_id,
                attempt_id=attempt_id,
                question_id=question_id,
                concept_id=cpt.id,
                awarded_score=awarded_score,
                max_score=max_score,
                is_correct=is_correct,
            )
            mastery = self.repository.recompute_mastery_for_concept(
                learner_id=learner_id,
                concept_id=cpt.id,
                course_id=course_id,
            )
            # Update review schedule
            sched = compute_next_schedule(
                current_interval=mastery.interval_days,
                current_ease=mastery.ease_factor,
                repetition_count=mastery.repetition_count,
                rating_or_correctness=is_correct,
            )
            due_iso = sched.next_due_date.isoformat()
            mastery.interval_days = sched.interval_days
            mastery.ease_factor = sched.ease_factor
            mastery.repetition_count = sched.repetition_count
            mastery.next_review_due = due_iso
            self.repository.upsert_concept_mastery(mastery)
            self.repository.create_or_update_review_task(
                learner_id=learner_id,
                concept_id=cpt.id,
                course_id=course_id,
                due_date=due_iso,
            )

    # ---------------------------------------------------------------------------
    # Progress Summaries & Analytics
    # ---------------------------------------------------------------------------

    def get_progress_summary(
        self,
        learner_id: str = "learner_default",
        course_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> LearnerProgressSummary:
        """Generate comprehensive learner progress summary with category breakdowns and evidence counts."""
        self.repository.get_or_create_profile(learner_id=learner_id)
        concepts = self.repository.list_concepts(course_id=course_id)
        mastery_list = self.repository.list_concept_mastery(learner_id=learner_id, course_id=course_id)
        mastery_map = {m.concept_id: m for m in mastery_list}

        now = datetime.now(timezone.utc)
        concept_summaries: list[ConceptProgressSummary] = []

        unassessed = 0
        needs_evidence = 0
        weak = 0
        moderate = 0
        strong = 0

        scheduled_tasks = self.repository.list_review_tasks(
            learner_id=learner_id,
            course_id=course_id,
            status=ReviewStatus.SCHEDULED,
        )
        task_by_concept = {t.concept_id: t for t in scheduled_tasks}

        for cpt in concepts:
            m = mastery_map.get(cpt.id)
            task = task_by_concept.get(cpt.id)

            if not m or m.observation_count == 0:
                unassessed += 1
                cat = MasteryCategory.UNASSESSED
                score = 0.0
                obs = 0
                corr = 0
                incorr = 0
                has_suff = False
                disp = "Unassessed"
                reason = "No graded assessments recorded for this concept."
                last_assessed = None
            else:
                cat = m.mastery_category
                score = m.mastery_score
                obs = m.observation_count
                corr = m.correct_count
                incorr = m.incorrect_count
                has_suff = m.has_sufficient_evidence
                reason = m.explanation_reason or ""
                last_assessed = m.last_assessed_at

                if cat == MasteryCategory.NEEDS_EVIDENCE:
                    needs_evidence += 1
                    disp = "Needs More Evidence"
                elif cat == MasteryCategory.WEAK:
                    weak += 1
                    disp = "Weak Evidence"
                elif cat == MasteryCategory.MODERATE:
                    moderate += 1
                    disp = "Moderate Progress"
                elif cat == MasteryCategory.STRONG:
                    strong += 1
                    disp = "Currently Strong"
                else:
                    unassessed += 1
                    disp = "Unassessed"

            # Derive review status from active task or mastery next_review_due
            next_due = task.due_date if task else (m.next_review_due if m else None)
            is_due = task.is_due if task else False
            is_over = task.is_overdue if task else False
            if not task and next_due:
                try:
                    due_dt = datetime.fromisoformat(next_due)
                    is_due = is_review_due(due_dt, now)
                    is_over = is_review_overdue(due_dt, now)
                except Exception:
                    pass

            unc = m.uncertainty if m else 0.2887

            concept_summaries.append(
                ConceptProgressSummary(
                    concept_id=cpt.id,
                    concept_name=cpt.name,
                    course_id=cpt.course_id,
                    mastery_score=score,
                    mastery_category=cat,
                    observation_count=obs,
                    correct_count=corr,
                    incorrect_count=incorr,
                    uncertainty=unc,
                    has_sufficient_evidence=has_suff,
                    last_assessed_at=last_assessed,
                    next_review_due=next_due,
                    is_review_due=is_due,
                    is_review_overdue=is_over,
                    status_display=disp,
                    explanation_reason=reason,
                )
            )

        # Query events
        events = self.repository.list_events(
            learner_id=learner_id,
            course_id=course_id,
            limit=50,
            since=since,
            until=until,
        )

        tot_graded = 0
        tot_corr = 0.0
        for e in events:
            if e.event_type == "pending_manual_review" or (e.is_correct is None and e.score_achieved is None):
                continue
            s_ach = e.score_achieved
            s_m = e.score_max
            if s_ach is None or s_m is None:
                if e.is_correct is not None:
                    s_ach = 1.0 if e.is_correct else 0.0
                    s_m = 1.0
                else:
                    continue
            if s_m > 0 and 0.0 <= s_ach <= s_m:
                tot_graded += 1
                tot_corr += (s_ach / s_m)

        tot_corr = round(tot_corr, 3)
        acc_rate = round(tot_corr / tot_graded * 100.0, 1) if tot_graded > 0 else None

        review_tasks = self.repository.list_review_tasks(
            learner_id=learner_id,
            course_id=course_id,
            status=ReviewStatus.SCHEDULED,
        )
        pending_reviews = sum(1 for t in review_tasks if t.is_due)
        overdue_reviews = sum(1 for t in review_tasks if t.is_overdue)

        return LearnerProgressSummary(
            learner_id=learner_id,
            course_id=course_id,
            total_concepts=len(concepts),
            unassessed_count=unassessed,
            needs_evidence_count=needs_evidence,
            weak_count=weak,
            moderate_count=moderate,
            strong_count=strong,
            total_graded_attempts=tot_graded,
            total_correct=tot_corr,
            accuracy_rate=acc_rate,
            pending_reviews_count=pending_reviews,
            overdue_reviews_count=overdue_reviews,
            concept_summaries=concept_summaries,
            recent_activity=events,
        )

    def get_course_summary(
        self,
        course_id: str,
        learner_id: str = "learner_default",
    ) -> CourseProgressSummary:
        """Retrieve progress summary aggregated for a specific course."""
        with self.db.get_connection() as conn:
            c_row = conn.execute("SELECT name FROM courses WHERE id = ?;", (course_id,)).fetchone()
            if not c_row:
                raise ValueError(f"Course '{course_id}' not found.")
            course_name = c_row["name"]

        prog = self.get_progress_summary(learner_id=learner_id, course_id=course_id)
        return CourseProgressSummary(
            course_id=course_id,
            course_name=course_name,
            total_concepts=prog.total_concepts,
            unassessed_count=prog.unassessed_count,
            needs_evidence_count=prog.needs_evidence_count,
            weak_count=prog.weak_count,
            moderate_count=prog.moderate_count,
            strong_count=prog.strong_count,
            total_graded_attempts=prog.total_graded_attempts,
            total_correct=prog.total_correct,
            accuracy_rate=prog.accuracy_rate,
            due_reviews_count=prog.pending_reviews_count,
            overdue_reviews_count=prog.overdue_reviews_count,
        )

    # ---------------------------------------------------------------------------
    # Explainable Study Recommendations
    # ---------------------------------------------------------------------------

    def get_recommendations(
        self,
        learner_id: str = "learner_default",
        course_id: str | None = None,
    ) -> list[StudyRecommendation]:
        """Generate deterministic, explainable recommendations for what the learner should study next."""
        summary = self.get_progress_summary(learner_id=learner_id, course_id=course_id)
        recs: list[StudyRecommendation] = []

        now = datetime.now(timezone.utc)

        for cs in summary.concept_summaries:
            # 1. Overdue Reviews (Priority 1)
            if cs.is_review_overdue:
                recs.append(
                    StudyRecommendation(
                        concept_id=cs.concept_id,
                        concept_name=cs.concept_name,
                        course_id=cs.course_id,
                        action="Complete Overdue Review",
                        reason=f"Spaced review is overdue (due: {cs.next_review_due}). Review to prevent retention decay.",
                        priority=1,
                        mastery_category=cs.mastery_category,
                        observation_count=cs.observation_count,
                        supporting_evidence={"due_date": cs.next_review_due, "is_overdue": True},
                    )
                )
                continue

            # 2. Weak concepts with sufficient observations (Priority 2)
            if cs.mastery_category == MasteryCategory.WEAK:
                recs.append(
                    StudyRecommendation(
                        concept_id=cs.concept_id,
                        concept_name=cs.concept_name,
                        course_id=cs.course_id,
                        action="Targeted Practice on Weak Concept",
                        reason=(
                            f"Low accuracy on graded attempts ({cs.correct_count}/{cs.observation_count} correct, "
                            f"score: {cs.mastery_score:.2f}). Targeted re-examination recommended."
                        ),
                        priority=2,
                        mastery_category=cs.mastery_category,
                        observation_count=cs.observation_count,
                        supporting_evidence={"correct": cs.correct_count, "total": cs.observation_count},
                    )
                )
                continue

            # 3. Scheduled reviews due today (Priority 3)
            if cs.is_review_due:
                recs.append(
                    StudyRecommendation(
                        concept_id=cs.concept_id,
                        concept_name=cs.concept_name,
                        course_id=cs.course_id,
                        action="Complete Scheduled Review",
                        reason=f"Review is due for spaced repetition reinforcement (due: {cs.next_review_due}).",
                        priority=3,
                        mastery_category=cs.mastery_category,
                        observation_count=cs.observation_count,
                        supporting_evidence={"due_date": cs.next_review_due, "is_due": True},
                    )
                )
                continue

            # 4. Concepts needing more evidence (Priority 4)
            if cs.mastery_category == MasteryCategory.NEEDS_EVIDENCE:
                recs.append(
                    StudyRecommendation(
                        concept_id=cs.concept_id,
                        concept_name=cs.concept_name,
                        course_id=cs.course_id,
                        action="Gather Assessment Evidence",
                        reason=(
                            f"Only {cs.observation_count} graded attempt(s) recorded. Take a practice quiz to "
                            f"establish confident mastery estimation."
                        ),
                        priority=4,
                        mastery_category=cs.mastery_category,
                        observation_count=cs.observation_count,
                        supporting_evidence={"observation_count": cs.observation_count},
                    )
                )
                continue

            # 5. Unassessed concepts (Priority 5)
            if cs.mastery_category == MasteryCategory.UNASSESSED:
                recs.append(
                    StudyRecommendation(
                        concept_id=cs.concept_id,
                        concept_name=cs.concept_name,
                        course_id=cs.course_id,
                        action="Initial Diagnostic Quiz",
                        reason="Concept has not yet been assessed in any exam. Take an initial diagnostic quiz.",
                        priority=5,
                        mastery_category=cs.mastery_category,
                        observation_count=0,
                        supporting_evidence={"unassessed": True},
                    )
                )
                continue

            # 6. Moderate progress concepts (Priority 6)
            if cs.mastery_category == MasteryCategory.MODERATE:
                due_info = f" Next scheduled review: {cs.next_review_due[:10]}." if cs.next_review_due else ""
                recs.append(
                    StudyRecommendation(
                        concept_id=cs.concept_id,
                        concept_name=cs.concept_name,
                        course_id=cs.course_id,
                        action="Reinforce Moderate Concept",
                        reason=(
                            f"Moderate progress ({cs.correct_count}/{cs.observation_count} correct, "
                            f"score: {cs.mastery_score:.2f}). Continue study to consolidate understanding.{due_info}"
                        ),
                        priority=6,
                        mastery_category=cs.mastery_category,
                        observation_count=cs.observation_count,
                        supporting_evidence={"correct": cs.correct_count, "total": cs.observation_count},
                    )
                )
                continue

            # 7. Currently strong concepts (Priority 7)
            if cs.mastery_category == MasteryCategory.STRONG:
                due_info = f" Next scheduled review: {cs.next_review_due[:10]}." if cs.next_review_due else ""
                recs.append(
                    StudyRecommendation(
                        concept_id=cs.concept_id,
                        concept_name=cs.concept_name,
                        course_id=cs.course_id,
                        action="Maintain Strong Retention",
                        reason=(
                            f"Currently strong evidence ({cs.correct_count}/{cs.observation_count} correct, "
                            f"score: {cs.mastery_score:.2f}). Retention is solid.{due_info}"
                        ),
                        priority=7,
                        mastery_category=cs.mastery_category,
                        observation_count=cs.observation_count,
                        supporting_evidence={"correct": cs.correct_count, "total": cs.observation_count},
                    )
                )

        recs.sort(key=lambda r: (r.priority, r.observation_count))
        return recs

    def explain_recommendation(
        self,
        concept_id: str,
        learner_id: str = "learner_default",
    ) -> StudyRecommendation | None:
        """Inspect and explain why a specific concept is recommended."""
        recs = self.get_recommendations(learner_id=learner_id)
        for r in recs:
            if r.concept_id == concept_id:
                return r
        # If not in top recommendations, build explanation from current concept state
        cpt = self.repository.get_concept(concept_id)
        if not cpt:
            return None
        m = self.repository.get_concept_mastery(learner_id, concept_id)
        cat = m.mastery_category if m else MasteryCategory.UNASSESSED
        obs = m.observation_count if m else 0
        return StudyRecommendation(
            concept_id=cpt.id,
            concept_name=cpt.name,
            course_id=cpt.course_id,
            action="Maintain Retention",
            reason=m.explanation_reason if m else "Concept has no recent review urgency.",
            priority=5,
            mastery_category=cat,
            observation_count=obs,
        )

    # ---------------------------------------------------------------------------
    # Review Queue Operations
    # ---------------------------------------------------------------------------

    def list_pending_reviews(
        self,
        learner_id: str = "learner_default",
        course_id: str | None = None,
        only_due: bool = False,
    ) -> list[ReviewTask]:
        """List active scheduled review tasks."""
        tasks = self.repository.list_review_tasks(
            learner_id=learner_id,
            course_id=course_id,
            status=ReviewStatus.SCHEDULED,
        )
        if only_due:
            return [t for t in tasks if t.is_due]
        return tasks

    def record_review_outcome(
        self,
        task_id: str,
        rating: ReviewRating,
    ) -> ReviewTask:
        """Record completed review rating and reschedule next review interval."""
        return self.repository.complete_review_task(task_id=task_id, rating=rating)
