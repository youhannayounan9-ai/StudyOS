"""Database repository for persisting and querying assessment entities and attempts."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Sequence

from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    Exam,
    ExamAttempt,
    ExamQuestion,
    ExamResult,
    GradingStatus,
    MCQOption,
    PublicExam,
    PublicQuestion,
    QuestionResult,
    QuestionSourceRef,
    QuestionType,
    ShortAnswerRubric,
)
from studyos.storage.db import Database

logger = logging.getLogger(__name__)


class AssessmentRepository:
    """Manages transactional SQLite storage for exams, questions, attempts, and grading reports."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save_exam(self, exam: Exam) -> Exam:
        """Atomically persist a validated Exam and its questions and source references."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO exams (id, course_id, title, topic, difficulty, question_count)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    exam.id,
                    exam.course_id,
                    exam.title,
                    exam.topic,
                    exam.difficulty.value,
                    exam.question_count,
                ),
            )

            for q in exam.questions:
                conn.execute(
                    """
                    INSERT INTO exam_questions (
                        id, exam_id, question_index, question_type, difficulty,
                        question_text, options_json, correct_answer, explanation,
                        rubric_json, points
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        q.id,
                        q.exam_id,
                        q.question_index,
                        q.question_type.value,
                        q.difficulty.value,
                        q.question_text,
                        q.options_to_json(),
                        q.correct_answer,
                        q.explanation,
                        q.rubric_to_json(),
                        q.points,
                    ),
                )

                for src in q.sources:
                    src_id = f"eqs_{uuid.uuid4().hex[:12]}"
                    # Query current chunk metadata if snapshot fields are missing on src
                    chunk_meta = None
                    if src.chunk_id:
                        chunk_meta = conn.execute(
                            """
                            SELECT c.id, c.location_type, c.location_index, c.location_title, c.content, d.filename
                            FROM chunks c
                            LEFT JOIN documents d ON c.document_id = d.id
                            WHERE c.id = ?;
                            """,
                            (src.chunk_id,),
                        ).fetchone()

                    snap_fn = src.filename or (chunk_meta["filename"] if chunk_meta and chunk_meta["filename"] else "unknown_document")
                    snap_lt = src.location_type or (chunk_meta["location_type"] if chunk_meta else None)
                    snap_li = src.location_index if src.location_index is not None else (chunk_meta["location_index"] if chunk_meta else None)
                    snap_ttl = src.location_title or (chunk_meta["location_title"] if chunk_meta else None)
                    snap_ex = src.excerpt or (chunk_meta["content"][:200].strip() if chunk_meta and chunk_meta["content"] else None)

                    conn.execute(
                        """
                        INSERT INTO exam_question_sources (
                            id, question_id, chunk_id, original_chunk_id, evidence_marker,
                            snapshot_filename, snapshot_location_type, snapshot_location_index,
                            snapshot_location_title, snapshot_excerpt
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            src_id,
                            q.id,
                            src.chunk_id,
                            src.chunk_id or src.original_chunk_id or "unknown_chunk",
                            src.evidence_marker,
                            snap_fn,
                            snap_lt,
                            snap_li,
                            snap_ttl,
                            snap_ex,
                        ),
                    )

        logger.info("Persisted exam '%s' (%s) with %d questions", exam.title, exam.id, len(exam.questions))
        return exam

    def get_exam(self, exam_id: str) -> Exam | None:
        """Retrieve full internal exam record with answer keys and source links."""
        with self.db.get_connection() as conn:
            exam_row = conn.execute("SELECT * FROM exams WHERE id = ?;", (exam_id,)).fetchone()
            if not exam_row:
                return None

            q_rows = conn.execute(
                "SELECT * FROM exam_questions WHERE exam_id = ? ORDER BY question_index ASC;",
                (exam_id,),
            ).fetchall()

            questions: list[ExamQuestion] = []
            for qr in q_rows:
                # Load choices
                options: list[MCQOption] | None = None
                if qr["options_json"]:
                    raw_opts = json.loads(qr["options_json"])
                    options = [MCQOption(**opt) for opt in raw_opts]

                # Load rubric
                rubric: ShortAnswerRubric | None = None
                if qr["rubric_json"]:
                    rubric = ShortAnswerRubric(**json.loads(qr["rubric_json"]))

                # Load source references with authoritative SQLite joins and historical snapshots
                src_rows = conn.execute(
                    """
                    SELECT eqs.evidence_marker,
                           eqs.chunk_id AS live_chunk_id,
                           eqs.original_chunk_id,
                           eqs.snapshot_filename,
                           eqs.snapshot_location_type,
                           eqs.snapshot_location_index,
                           eqs.snapshot_location_title,
                           eqs.snapshot_excerpt,
                           c.id AS current_chunk_id,
                           c.content AS current_content,
                           c.location_type AS current_location_type,
                           c.location_index AS current_location_index,
                           c.location_title AS current_location_title,
                           d.filename AS current_filename
                    FROM exam_question_sources eqs
                    LEFT JOIN chunks c ON eqs.chunk_id = c.id
                    LEFT JOIN documents d ON c.document_id = d.id
                    WHERE eqs.question_id = ?;
                    """,
                    (qr["id"],),
                ).fetchall()

                sources: list[QuestionSourceRef] = []
                for sr in src_rows:
                    is_live = sr["current_chunk_id"] is not None and sr["current_content"] is not None
                    if is_live:
                        # Live chunk exists: current metadata is authoritative
                        sources.append(
                            QuestionSourceRef(
                                chunk_id=sr["current_chunk_id"],
                                original_chunk_id=sr["original_chunk_id"],
                                evidence_marker=sr["evidence_marker"],
                                filename=sr["current_filename"] or sr["snapshot_filename"],
                                location_type=sr["current_location_type"] or sr["snapshot_location_type"],
                                location_index=sr["current_location_index"] if sr["current_location_index"] is not None else sr["snapshot_location_index"],
                                location_title=sr["current_location_title"] or sr["snapshot_location_title"],
                                excerpt=sr["current_content"][:200].strip() if sr["current_content"] else sr["snapshot_excerpt"],
                                is_stale=False,
                            )
                        )
                    else:
                        # Chunk or document was deleted: mark stale and preserve historical snapshot
                        sources.append(
                            QuestionSourceRef(
                                chunk_id=None,
                                original_chunk_id=sr["original_chunk_id"],
                                evidence_marker=sr["evidence_marker"],
                                filename=sr["snapshot_filename"],
                                location_type=sr["snapshot_location_type"],
                                location_index=sr["snapshot_location_index"],
                                location_title=sr["snapshot_location_title"],
                                excerpt=sr["snapshot_excerpt"],
                                is_stale=True,
                            )
                        )

                questions.append(
                    ExamQuestion(
                        id=qr["id"],
                        exam_id=qr["exam_id"],
                        question_index=qr["question_index"],
                        question_type=QuestionType(qr["question_type"]),
                        difficulty=Difficulty(qr["difficulty"]),
                        question_text=qr["question_text"],
                        options=options,
                        correct_answer=qr["correct_answer"],
                        explanation=qr["explanation"],
                        rubric=rubric,
                        sources=sources,
                        points=float(qr["points"]),
                    )
                )

            return Exam(
                id=exam_row["id"],
                course_id=exam_row["course_id"],
                title=exam_row["title"],
                topic=exam_row["topic"],
                difficulty=Difficulty(exam_row["difficulty"]),
                question_count=exam_row["question_count"],
                questions=questions,
                created_at=str(exam_row["created_at"]),
            )

    def get_public_exam(self, exam_id: str) -> PublicExam | None:
        """Retrieve student-facing public exam without any answer keys or rubrics."""
        exam = self.get_exam(exam_id)
        if not exam:
            return None

        public_questions: list[PublicQuestion] = []
        for q in exam.questions:
            public_questions.append(
                PublicQuestion(
                    id=q.id,
                    question_index=q.question_index,
                    question_type=q.question_type,
                    difficulty=q.difficulty,
                    question_text=q.question_text,
                    options=q.options,
                    points=q.points,
                )
            )

        return PublicExam(
            id=exam.id,
            course_id=exam.course_id,
            title=exam.title,
            topic=exam.topic,
            difficulty=exam.difficulty,
            question_count=exam.question_count,
            questions=public_questions,
            created_at=exam.created_at,
        )

    def list_exams(self, course_id: str | None = None) -> list[Exam]:
        """List all stored exams, optionally filtered by course."""
        with self.db.get_connection() as conn:
            if course_id:
                rows = conn.execute(
                    "SELECT id FROM exams WHERE course_id = ? ORDER BY created_at DESC;",
                    (course_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id FROM exams ORDER BY created_at DESC;"
                ).fetchall()

            exams: list[Exam] = []
            for r in rows:
                ex = self.get_exam(r["id"])
                if ex:
                    exams.append(ex)
            return exams

    def create_attempt(self, exam_id: str) -> ExamAttempt:
        """Create a new exam attempt in IN_PROGRESS state."""
        att_id = f"att_{uuid.uuid4().hex[:12]}"
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO exam_attempts (id, exam_id, status)
                VALUES (?, ?, ?);
                """,
                (att_id, exam_id, AttemptStatus.IN_PROGRESS.value),
            )
        return ExamAttempt(
            id=att_id,
            exam_id=exam_id,
            status=AttemptStatus.IN_PROGRESS,
        )

    def get_attempt(self, attempt_id: str) -> ExamAttempt | None:
        """Retrieve attempt status."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM exam_attempts WHERE id = ?;", (attempt_id,)
            ).fetchone()
            if not row:
                return None
            return ExamAttempt(
                id=row["id"],
                exam_id=row["exam_id"],
                status=AttemptStatus(row["status"]),
                score=float(row["score"]) if row["score"] is not None else None,
                provisional_score=float(row["provisional_score"]) if "provisional_score" in row.keys() and row["provisional_score"] is not None else None,
                max_score=float(row["max_score"]) if row["max_score"] is not None else None,
                percentage=float(row["percentage"]) if row["percentage"] is not None else None,
                started_at=str(row["started_at"]) if row["started_at"] else None,
                completed_at=str(row["completed_at"]) if row["completed_at"] else None,
            )

    def save_attempt_results(
        self,
        attempt_id: str,
        results: Sequence[QuestionResult],
        provisional_score: float,
        max_score: float,
    ) -> AttemptStatus:
        """Atomically persist graded question attempts and update exam attempt status."""
        has_pending = any(qr.status == GradingStatus.PENDING_MANUAL_REVIEW for qr in results)
        attempt_status = AttemptStatus.PENDING_REVIEW if has_pending else AttemptStatus.COMPLETED
        final_score = None if has_pending else provisional_score
        final_percentage = None if has_pending else (round((provisional_score / max_score * 100.0), 2) if max_score > 0 else 0.0)

        with self.db.transaction() as conn:
            for qr in results:
                qa_id = f"qa_{uuid.uuid4().hex[:12]}"
                is_corr = 1 if qr.is_correct is True else (0 if qr.is_correct is False else None)
                conn.execute(
                    """
                    INSERT INTO question_attempts (
                        id, attempt_id, question_id, status, submitted_answer, is_correct,
                        score, max_score, feedback, graded_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                    """,
                    (
                        qa_id,
                        attempt_id,
                        qr.question_id,
                        qr.status.value,
                        qr.submitted_answer,
                        is_corr,
                        qr.score,
                        qr.max_score,
                        qr.feedback,
                    ),
                )

            conn.execute(
                """
                UPDATE exam_attempts
                SET status = ?, score = ?, provisional_score = ?, max_score = ?, percentage = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (
                    attempt_status.value,
                    final_score,
                    provisional_score,
                    max_score,
                    final_percentage,
                    attempt_id,
                ),
            )
        return attempt_status

    def get_attempt_result(self, attempt_id: str) -> ExamResult | None:
        """Reconstruct full ExamResult for a completed or review-pending attempt."""
        attempt = self.get_attempt(attempt_id)
        if not attempt or attempt.status == AttemptStatus.IN_PROGRESS:
            return None

        exam = self.get_exam(attempt.exam_id)
        if not exam:
            return None

        with self.db.get_connection() as conn:
            qa_rows = conn.execute(
                """
                SELECT qa.*, eq.question_index, eq.question_text, eq.question_type, eq.correct_answer, eq.explanation
                FROM question_attempts qa
                JOIN exam_questions eq ON qa.question_id = eq.id
                WHERE qa.attempt_id = ?
                ORDER BY eq.question_index ASC;
                """,
                (attempt_id,),
            ).fetchall()

            q_map = {q.id: q for q in exam.questions}
            question_results: list[QuestionResult] = []
            needs_manual_review = False

            for r in qa_rows:
                qid = r["question_id"]
                original_q = q_map.get(qid)
                sources = original_q.sources if original_q else []

                raw_status = r["status"] if "status" in r.keys() and r["status"] else None
                if raw_status:
                    q_status = GradingStatus(raw_status)
                else:
                    q_status = (
                        GradingStatus.PENDING_MANUAL_REVIEW
                        if r["is_correct"] is None and r["score"] is None
                        else GradingStatus.GRADED
                    )

                if q_status == GradingStatus.PENDING_MANUAL_REVIEW:
                    needs_manual_review = True
                    q_score = None
                    is_corr = None
                else:
                    q_score = float(r["score"]) if r["score"] is not None else 0.0
                    is_corr = True if r["is_correct"] == 1 else False

                question_results.append(
                    QuestionResult(
                        question_id=qid,
                        question_index=r["question_index"],
                        question_text=r["question_text"],
                        question_type=QuestionType(r["question_type"]),
                        submitted_answer=r["submitted_answer"],
                        correct_answer=r["correct_answer"],
                        is_correct=is_corr,
                        status=q_status,
                        score=q_score,
                        max_score=float(r["max_score"]),
                        explanation=r["explanation"],
                        feedback=r["feedback"] or "",
                        citations=sources,
                    )
                )

            prov_score = (
                attempt.provisional_score
                if attempt.provisional_score is not None
                else sum(q.score for q in question_results if q.score is not None)
            )
            tot_max = attempt.max_score or sum(q.max_score for q in question_results)
            prov_pct = round((prov_score / tot_max * 100.0), 2) if tot_max > 0 else 0.0

            final_score = attempt.score if not needs_manual_review else None
            final_percentage = attempt.percentage if not needs_manual_review else None

            return ExamResult(
                attempt_id=attempt.id,
                exam_id=exam.id,
                course_id=exam.course_id,
                title=exam.title,
                status=attempt.status,
                score=final_score,
                provisional_score=prov_score,
                max_score=tot_max,
                percentage=final_percentage,
                provisional_percentage=prov_pct,
                completed_at=attempt.completed_at,
                questions=question_results,
                needs_manual_review=needs_manual_review,
            )

    def grade_question_manually(
        self,
        attempt_id: str,
        question_id: str,
        awarded_score: float,
        feedback: str | None = None,
    ) -> ExamResult:
        """Complete instructor review for an ungraded short-answer question."""
        attempt = self.get_attempt(attempt_id)
        if not attempt:
            raise ValueError(f"Attempt '{attempt_id}' not found.")
        if attempt.status != AttemptStatus.PENDING_REVIEW:
            raise ValueError(
                f"Attempt '{attempt_id}' is not pending review (current status: {attempt.status.value})."
            )

        with self.db.transaction() as conn:
            qa_row = conn.execute(
                """
                SELECT qa.*, eq.points
                FROM question_attempts qa
                JOIN exam_questions eq ON qa.question_id = eq.id
                WHERE qa.attempt_id = ? AND qa.question_id = ?;
                """,
                (attempt_id, question_id),
            ).fetchone()

            if not qa_row:
                raise ValueError(
                    f"Question attempt for question '{question_id}' in attempt '{attempt_id}' not found."
                )

            max_points = float(qa_row["points"])
            if awarded_score < 0.0 or awarded_score > max_points:
                raise ValueError(
                    f"Awarded score {awarded_score} out of bounds; must be between 0.0 and {max_points}."
                )

            is_correct_val = 1 if awarded_score >= max_points * 0.7 else 0
            fb = feedback or "Evaluated via instructor manual review."

            conn.execute(
                """
                UPDATE question_attempts
                SET status = ?, score = ?, is_correct = ?, feedback = ?, graded_at = CURRENT_TIMESTAMP
                WHERE attempt_id = ? AND question_id = ?;
                """,
                (
                    GradingStatus.GRADED.value,
                    awarded_score,
                    is_correct_val,
                    fb,
                    attempt_id,
                    question_id,
                ),
            )

            # Check if any questions remain pending review
            pending_count = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM question_attempts
                WHERE attempt_id = ? AND status = ?;
                """,
                (attempt_id, GradingStatus.PENDING_MANUAL_REVIEW.value),
            ).fetchone()["cnt"]

            rows = conn.execute(
                "SELECT score, max_score FROM question_attempts WHERE attempt_id = ?;",
                (attempt_id,),
            ).fetchall()
            tot_score = sum(r["score"] for r in rows if r["score"] is not None)
            tot_max = sum(r["max_score"] for r in rows if r["max_score"] is not None)

            if pending_count == 0:
                final_pct = round((tot_score / tot_max * 100.0), 2) if tot_max > 0 else 0.0
                conn.execute(
                    """
                    UPDATE exam_attempts
                    SET status = ?, score = ?, provisional_score = ?, percentage = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?;
                    """,
                    (
                        AttemptStatus.COMPLETED.value,
                        tot_score,
                        tot_score,
                        final_pct,
                        attempt_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE exam_attempts
                    SET provisional_score = ?
                    WHERE id = ?;
                    """,
                    (tot_score, attempt_id),
                )

        result = self.get_attempt_result(attempt_id)
        if not result:
            raise RuntimeError(f"Failed to retrieve updated attempt result for '{attempt_id}'.")
        return result
