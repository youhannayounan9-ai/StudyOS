"""Unit tests for StudyOS Stage 6 Migration 0007 upgrade safety and integrity."""

import sqlite3
from pathlib import Path
import pytest
from studyos.storage.db import Database


def test_migration_0007_applies_on_fresh_database(tmp_path):
    """Test that all migrations 0001-0007 execute cleanly on a fresh database."""
    db_file = tmp_path / "fresh_stage6.db"
    db = Database(db_file)
    db.migrate()

    with db.get_connection() as conn:
        # Verify schema_migrations contains 7
        applied = conn.execute("SELECT version FROM schema_migrations ORDER BY version;").fetchall()
        versions = [r["version"] for r in applied]
        assert versions == [1, 2, 3, 4, 5, 6, 7]

        # Verify new tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "concept_relationships" in table_names
        assert "concept_discovery_runs" in table_names
        assert "external_research_records" in table_names
        assert "external_research_sources" in table_names

        # Verify new columns in concepts
        c_cols = conn.execute("PRAGMA table_info(concepts);").fetchall()
        c_col_names = {r["name"] for r in c_cols}
        assert "review_status" in c_col_names
        assert "provenance" in c_col_names
        assert "learning_objectives" in c_col_names

        # Verify new columns in concept_mastery
        m_cols = conn.execute("PRAGMA table_info(concept_mastery);").fetchall()
        m_col_names = {r["name"] for r in m_cols}
        assert "stability" in m_col_names
        assert "difficulty" in m_col_names
        assert "retrievability" in m_col_names
        assert "scheduler_policy" in m_col_names

        # Verify new columns in review_tasks
        rt_cols = conn.execute("PRAGMA table_info(review_tasks);").fetchall()
        rt_col_names = {r["name"] for r in rt_cols}
        assert "scheduler_policy" in rt_col_names
        assert "scheduled_interval" in rt_col_names
        assert "estimated_retention" in rt_col_names


def test_migration_0007_upgrades_existing_stage5_database(tmp_path):
    """Test that a database with migrations 0001-0006 upgrades safely to 0007 without data loss."""
    db_file = tmp_path / "existing_stage5.db"

    # 1. Manually apply 0001 through 0006 only
    migrations_dir = Path("studyos/storage/migrations")
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
    for v in [1, 2, 3, 4, 5, 6]:
        sql_file = list(migrations_dir.glob(f"{v:04d}_*.sql"))[0]
        conn.executescript(sql_file.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations (version) VALUES (?);", (v,))
    conn.commit()

    # Insert sample Stage 5 data
    conn.execute("INSERT INTO courses (id, name, code) VALUES ('crs_test', 'Algorithms', 'CS101');")
    conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_1', 'crs_test', 'Trees', 'trees');")
    conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_2', 'crs_test', 'BST', 'bst');")
    conn.execute(
        """
        INSERT INTO concept_mastery (
            id, learner_id, concept_id, course_id, mastery_score, mastery_category,
            observation_count, correct_count, incorrect_count, uncertainty
        )
        VALUES ('cm_1', 'learner_default', 'cpt_1', 'crs_test', 0.667, 'moderate', 1, 1.0, 0.0, 0.2357);
        """
    )
    conn.commit()
    conn.close()

    # 2. Run Database.migrate() to trigger upgrade to 0007
    db = Database(db_file)
    db.migrate()

    with db.get_connection() as check_conn:
        # Check migration version updated to 7
        applied = check_conn.execute("SELECT version FROM schema_migrations ORDER BY version;").fetchall()
        assert [r["version"] for r in applied] == [1, 2, 3, 4, 5, 6, 7]

        # Check existing data preserved intact
        cpt = check_conn.execute("SELECT * FROM concepts WHERE id = 'cpt_1';").fetchone()
        assert cpt["name"] == "Trees"
        assert cpt["review_status"] == "accepted"
        assert cpt["provenance"] == "assessment_derived"

        cm = check_conn.execute("SELECT * FROM concept_mastery WHERE id = 'cm_1';").fetchone()
        assert cm["mastery_score"] == 0.667
        assert cm["stability"] == 1.0
        assert cm["difficulty"] == 5.0
        assert cm["scheduler_policy"] == "local-fsrs-v1"

        # Check triggers exist on upgraded database
        triggers = check_conn.execute("SELECT name FROM sqlite_master WHERE type='trigger';").fetchall()
        trig_names = {r["name"] for r in triggers}
        assert "trg_concept_rel_insert_course_integrity" in trig_names
        assert "trg_concept_rel_update_course_integrity" in trig_names
        assert "trg_concept_rel_insert_cycle_check" in trig_names
        assert "trg_concept_rel_update_cycle_check" in trig_names
        assert "trg_concepts_insert_status" in trig_names
        assert "concepts_validate_course_au" in trig_names
        assert "trg_concept_chunks_insert_course_integrity" in trig_names


def test_migration_0007_is_idempotent(tmp_path):
    """Calling migrate() multiple times is safe and a no-op."""
    db_file = tmp_path / "idempotent.db"
    db = Database(db_file)
    db.migrate()
    # Second call
    db.migrate()
    with db.get_connection() as conn:
        applied = conn.execute("SELECT version FROM schema_migrations ORDER BY version;").fetchall()
        assert len(applied) == 7
