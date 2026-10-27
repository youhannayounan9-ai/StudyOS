"""Unit tests for Stage 5 migration upgrade safety, fresh initialization, and foreign key integrity."""

from pathlib import Path
import sqlite3
import pytest

from studyos.storage.db import Database


def test_fresh_database_applies_all_migrations(tmp_path):
    db_file = tmp_path / "fresh.db"
    db = Database(db_file)
    db.migrate()

    with db.get_connection() as conn:
        # Verify schema_migrations versions 1 through 6
        versions = [r["version"] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;").fetchall()]
        assert versions == [1, 2, 3, 4, 5, 6]

        # Verify Stage 5 tables exist
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        expected_tables = [
            "courses",
            "documents",
            "chunks",
            "exams",
            "exam_questions",
            "exam_question_sources",
            "exam_attempts",
            "question_attempts",
            "learner_profile",
            "concepts",
            "question_concepts",
            "concept_chunks",
            "concept_mastery",
            "learner_events",
            "review_tasks",
        ]
        for t in expected_tables:
            assert t in tables, f"Expected table '{t}' missing from fresh database."

        # Verify foreign keys check cleanly
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        assert len(fk_violations) == 0


def test_stage4_to_stage5_upgrade_safety(tmp_path):
    db_file = tmp_path / "stage4_upgrade.db"
    migrations_dir = Path(__file__).resolve().parent.parent.parent / "studyos" / "storage" / "migrations"

    # 1. Simulate existing Stage 4 database: apply migrations 0001 through 0005 only
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    for v in [1, 2, 3, 4, 5]:
        mf = next(migrations_dir.glob(f"000{v}_*.sql"))
        conn.executescript(mf.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations (version) VALUES (?);", (v,))

    # Insert baseline course, exam, and attempt data into Stage 4 schema
    conn.execute("INSERT INTO courses (id, name) VALUES ('crs_stage4', 'Stage 4 Baseline Course');")
    conn.execute(
        """
        INSERT INTO exams (id, course_id, title, difficulty, question_count)
        VALUES ('ex_stage4', 'crs_stage4', 'Existing Exam', 'medium', 1);
        """
    )
    conn.execute(
        """
        INSERT INTO exam_questions (id, exam_id, question_index, question_type, difficulty, question_text, correct_answer, explanation)
        VALUES ('q_stage4', 'ex_stage4', 1, 'mcq', 'medium', 'Question text', 'A', 'Explanation');
        """
    )
    conn.execute(
        """
        INSERT INTO exam_attempts (id, exam_id, status, score, provisional_score, max_score, percentage)
        VALUES ('att_stage4', 'ex_stage4', 'completed', 1.0, 1.0, 1.0, 100.0);
        """
    )
    conn.execute(
        """
        INSERT INTO question_attempts (id, attempt_id, question_id, status, submitted_answer, is_correct, score, max_score)
        VALUES ('qa_stage4', 'att_stage4', 'q_stage4', 'graded', 'A', 1, 1.0, 1.0);
        """
    )
    conn.commit()
    conn.close()

    # 2. Upgrade using StudyOS Database migration runner
    db = Database(db_file)
    db.migrate()

    with db.get_connection() as check_conn:
        # Check migrations recorded
        versions = [r["version"] for r in check_conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;").fetchall()]
        assert versions == [1, 2, 3, 4, 5, 6]

        # Verify existing data preserved
        exam_row = check_conn.execute("SELECT * FROM exams WHERE id = 'ex_stage4';").fetchone()
        assert exam_row is not None
        assert exam_row["title"] == "Existing Exam"

        attempt_row = check_conn.execute("SELECT * FROM exam_attempts WHERE id = 'att_stage4';").fetchone()
        assert attempt_row is not None
        assert attempt_row["score"] == 1.0

        # Verify foreign keys check cleanly
        fk_violations = check_conn.execute("PRAGMA foreign_key_check;").fetchall()
        assert len(fk_violations) == 0


def test_repeated_migration_is_idempotent(tmp_path):
    db_file = tmp_path / "idempotent.db"
    db = Database(db_file)

    # First run
    db.migrate()

    # Repeated runs must succeed without error and apply no extra versions
    db.migrate()
    db.migrate()

    with db.get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM schema_migrations;").fetchone()["cnt"]
        assert count == 6
