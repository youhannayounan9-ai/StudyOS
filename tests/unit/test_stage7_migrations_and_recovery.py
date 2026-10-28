"""Unit and integration tests for StudyOS Stage 7 migration integrity, recoverability, and backup/restore."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pytest

from studyos.assessment.models import Difficulty, Exam, ExamQuestion, QuestionType
from studyos.assessment.repository import AssessmentRepository
from studyos.ingestion.models import DocumentChunk
from studyos.learner.models import ConceptMastery, MasteryCategory
from studyos.learner.repository import LearnerRepository
from studyos.research.models import ExternalSource, ResearchRecord, ResearchStatus, SourceType
from studyos.research.repository import ResearchRepository
from studyos.storage.db import Database


class TestMigrationIntegrityAndRecoverability:
    """Rigorous audit of database creation, migration upgrades, foreign key invariants, and backup/restore."""

    def test_fresh_database_creation_and_integrity(self, tmp_path):
        """1. Brand-new database applies all migrations 0001-0007 with 0 FK violations and 100% integrity."""
        db_path = tmp_path / "fresh_test.db"
        db = Database(db_path)
        db.migrate()

        with db.get_connection() as conn:
            # Check schema migrations
            applied = [r["version"] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version;").fetchall()]
            assert applied == [1, 2, 3, 4, 5, 6, 7]

            # Integrity check
            integrity_rows = [r[0] for r in conn.execute("PRAGMA integrity_check;").fetchall()]
            assert integrity_rows == ["ok"]

            # Foreign key check
            fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
            assert len(fk_violations) == 0

    def test_upgrade_from_pre_stage6_with_data_preservation(self, tmp_path):
        """2. Upgrade database representing pre-Stage-6 schema (0001-0005) with populated data to 0007."""
        db_path = tmp_path / "upgrade_pre6.db"
        migrations_dir = Path(__file__).parents[2] / "studyos" / "storage" / "migrations"

        # Apply 0001 through 0005 manually
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
        for v in [1, 2, 3, 4, 5]:
            sql_file = list(migrations_dir.glob(f"{v:04d}_*.sql"))[0]
            conn.executescript(sql_file.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (?);", (v,))
        conn.commit()

        # Seed pre-Stage-6 records
        conn.execute("INSERT INTO courses (id, name, code) VALUES ('crs_pre6', 'Database Systems', 'CS301');")
        conn.execute(
            "INSERT INTO documents (id, course_id, filename, file_type, file_size_bytes, sha256_hash, status) "
            "VALUES ('doc_1', 'crs_pre6', 'notes.pdf', 'pdf', 1024, 'sha_hash_1', 'ready');"
        )
        conn.execute(
            "INSERT INTO chunks (id, document_id, course_id, chunk_index, location_type, location_index, content, token_count, content_hash) "
            "VALUES ('chk_1', 'doc_1', 'crs_pre6', 0, 'page', 1, 'ACID transactions guarantee durability.', 10, 'hash_c1');"
        )
        conn.commit()
        conn.close()

        # Upgrade to latest using Database.migrate()
        db = Database(db_path)
        db.migrate()

        with db.get_connection() as conn:
            applied = [r["version"] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version;").fetchall()]
            assert applied == [1, 2, 3, 4, 5, 6, 7]

            # Verify existing records preserved intact
            course = conn.execute("SELECT * FROM courses WHERE id = 'crs_pre6';").fetchone()
            assert course["name"] == "Database Systems"

            doc = conn.execute("SELECT * FROM documents WHERE id = 'doc_1';").fetchone()
            assert doc["filename"] == "notes.pdf"

            chunk = conn.execute("SELECT * FROM chunks WHERE id = 'chk_1';").fetchone()
            assert "ACID" in chunk["content"]

            # Verify foreign keys and integrity
            assert [r[0] for r in conn.execute("PRAGMA integrity_check;").fetchall()] == ["ok"]
            assert len(conn.execute("PRAGMA foreign_key_check;").fetchall()) == 0

    def test_reopening_migrated_database_is_noop_and_idempotent(self, tmp_path):
        """4. Reopening an already migrated database causes no errors and does not reapply changes."""
        db_path = tmp_path / "reopen_test.db"
        db = Database(db_path)
        db.migrate()

        # Connect with a second instance and migrate
        db2 = Database(db_path)
        db2.migrate()

        with db2.get_connection() as conn:
            applied = [r["version"] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version;").fetchall()]
            assert applied == [1, 2, 3, 4, 5, 6, 7]
            assert [r[0] for r in conn.execute("PRAGMA integrity_check;").fetchall()] == ["ok"]

    def test_full_system_record_preservation_across_all_entities(self, tmp_path):
        """6. Comprehensive test verifying all domain entities coexist and preserve foreign key integrity."""
        db_path = tmp_path / "comprehensive_test.db"
        db = Database(db_path)
        db.migrate()

        # 1. Course, Document, Chunks
        course = db.create_course("Distributed Systems", "CS401")
        doc = db.create_document(
            course_id=course.id,
            filename="raft.pdf",
            file_type="pdf",
            file_size_bytes=2048,
            sha256_hash="hash_raft_123",
        )
        chunk = DocumentChunk.create(
            chunk_id="chk_raft_1",
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Raft consensus achieves state machine replication via leader election.",
            token_count=12,
        )
        db.insert_chunks([chunk])

        # 2. Assessment Exam, Question, Citation Snapshot
        assessment_repo = AssessmentRepository(db)
        exam = Exam(
            id="exam_raft_1",
            course_id=course.id,
            title="Raft Consensus Quiz",
            difficulty=Difficulty.MEDIUM,
            question_count=1,
            questions=[
                ExamQuestion(
                    id="q_raft_1",
                    exam_id="exam_raft_1",
                    question_index=1,
                    question_type=QuestionType.MCQ,
                    difficulty=Difficulty.MEDIUM,
                    question_text="How does Raft handle leader failure?",
                    correct_answer="A",
                    explanation="Followers timeout and start election.",
                    points=1.0,
                )
            ],
        )
        assessment_repo.save_exam(exam)
        attempt = assessment_repo.create_attempt(exam.id)

        # 3. Learner Profile, Concept, Mastery, and Review Task
        learner_repo = LearnerRepository(db)
        learner = learner_repo.get_or_create_profile("learner_default")
        concept = learner_repo.get_or_create_concept(
            course_id=course.id,
            name="Leader Election",
            description="Process of electing a Raft cluster leader",
        )
        learner_repo.map_concept_to_chunk(concept.id, chunk.chunk_id)
        mastery = ConceptMastery(
            id="cm_raft_1",
            learner_id=learner.id,
            concept_id=concept.id,
            course_id=course.id,
            mastery_score=0.85,
            mastery_category=MasteryCategory.STRONG,
            observation_count=3,
            correct_count=2.5,
            incorrect_count=0.5,
            uncertainty=0.15,
            stability=2.5,
            difficulty=4.0,
            retrievability=0.92,
            scheduler_policy="local-fsrs-v1",
        )
        learner_repo.upsert_concept_mastery(mastery)
        task = learner_repo.create_or_update_review_task(
            learner_id=learner.id,
            concept_id=concept.id,
            course_id=course.id,
            due_date="2026-11-01T00:00:00Z",
            scheduled_interval=3.0,
            estimated_retention=0.9,
            scheduler_policy="local-fsrs-v1",
        )

        # 4. Concept Graph Relationships
        concept2 = learner_repo.get_or_create_concept(
            course_id=course.id,
            name="Log Replication",
            description="Appending entries across followers",
        )
        rel = learner_repo.add_concept_relationship(
            course_id=course.id,
            source_concept_id=concept.id,
            target_concept_id=concept2.id,
            relationship_type="prerequisite",
        )

        # 5. External Research Records
        research_repo = ResearchRepository(db)
        record = ResearchRecord(
            id="res_raft_ext",
            query="Raft consensus proof of safety",
            course_id=course.id,
            summary="Raft ensures safety through state machine safety property.",
            status=ResearchStatus.SUCCESS,
            sources=[
                ExternalSource(
                    id="src_raft_1",
                    research_id="res_raft_ext",
                    citation_index=1,
                    url="https://en.wikipedia.org/wiki/Raft_(algorithm)",
                    title="Raft Algorithm",
                    snippet="Raft is a consensus algorithm.",
                    source_type=SourceType.WEB_PAGE,
                )
            ],
        )
        research_repo.save_research_record(record)

        # Verify all foreign keys and integrity
        with db.get_connection() as conn:
            assert [r[0] for r in conn.execute("PRAGMA integrity_check;").fetchall()] == ["ok"]
            assert len(conn.execute("PRAGMA foreign_key_check;").fetchall()) == 0

            # Verify entities exist
            assert conn.execute("SELECT COUNT(*) FROM courses;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM exams;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM exam_questions;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM exam_attempts;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM concepts;").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM concept_relationships;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM concept_chunks;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM concept_mastery;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM review_tasks;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM external_research_records;").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM external_research_sources;").fetchone()[0] == 1

    def test_sqlite_online_backup_and_restore_fidelity(self, tmp_path):
        """7. Test SQLite online backup and restore protocol, verifying 100% data fidelity and integrity."""
        primary_path = tmp_path / "primary.db"
        backup_path = tmp_path / "backup.db"

        # Initialize primary DB and seed records
        db = Database(primary_path)
        db.migrate()

        c = db.create_course("Operating Systems", "CS201")
        d = db.create_document(c.id, "kernel.pdf", "pdf", 4096, "hash_kernel")
        chunk = DocumentChunk.create(
            chunk_id="chk_k1",
            document_id=d.id,
            course_id=c.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Monolithic kernels run all core OS services in supervisor mode.",
            token_count=11,
        )
        db.insert_chunks([chunk])

        # Execute safe online backup using sqlite3.Connection.backup
        with sqlite3.connect(primary_path) as src_conn:
            with sqlite3.connect(backup_path) as dst_conn:
                src_conn.backup(dst_conn)

        # Open the restored backup database and verify integrity
        restored_db = Database(backup_path)
        with restored_db.get_connection() as conn:
            # Check PRAGMA integrity
            assert [r[0] for r in conn.execute("PRAGMA integrity_check;").fetchall()] == ["ok"]
            assert len(conn.execute("PRAGMA foreign_key_check;").fetchall()) == 0

            # Verify tables and records
            courses = conn.execute("SELECT * FROM courses;").fetchall()
            assert len(courses) == 1
            assert courses[0]["name"] == "Operating Systems"

            chunks = conn.execute("SELECT * FROM chunks;").fetchall()
            assert len(chunks) == 1
            assert "Monolithic kernels" in chunks[0]["content"]

            # Verify FTS search works on restored database
            fts_rows = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'supervisor';").fetchall()
            assert len(fts_rows) == 1
