"""Direct SQL tests verifying database-enforced integrity and DAG cycle prevention.

These tests deliberately bypass Python repository and service logic to ensure that SQLite
constraints and triggers strictly enforce:
1. Cross-course isolation on relationship insert and update.
2. Self-edge prevention.
3. Enum and confidence bounds validation.
4. Prerequisite DAG cycle rejection on direct SQL insert.
5. Prerequisite DAG cycle rejection when updating proposed edges to accepted.
6. Transaction rollback cleanliness on rejected operations.
7. Valid DAG diamond structures and bidirectional related edges succeed.
"""

import sqlite3
import pytest
from studyos.storage.db import Database


@pytest.fixture
def fresh_db(tmp_path):
    """Set up a fresh database with all migrations applied."""
    db_file = tmp_path / "graph_sql_test.db"
    db = Database(db_file)
    db.migrate()
    return db


def test_sql_cross_course_relationship_insert_rejected(fresh_db):
    """Direct SQL insert connecting concepts from different courses is aborted by trigger."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_a', 'Course A'), ('crs_b', 'Course B');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_a', 'crs_a', 'Concept A', 'concept a');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_b', 'crs_b', 'Concept B', 'concept b');")

        # Attempt insert in crs_a referencing cpt_b (which belongs to crs_b)
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute(
                """
                INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type)
                VALUES ('rel_cross_1', 'crs_a', 'cpt_a', 'cpt_b', 'prerequisite');
                """
            )
        assert "Cross-course concept relationship rejected" in str(exc_info.value)

        # Verify nothing was inserted
        rows = conn.execute("SELECT * FROM concept_relationships;").fetchall()
        assert len(rows) == 0


def test_sql_cross_course_relationship_update_rejected(fresh_db):
    """Direct SQL update reassigning a relationship's course_id or concepts across courses is rejected."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_a', 'Course A'), ('crs_b', 'Course B');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_a1', 'crs_a', 'Concept A1', 'concept a1');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_a2', 'crs_a', 'Concept A2', 'concept a2');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_b1', 'crs_b', 'Concept B1', 'concept b1');")

        # Insert valid relationship within crs_a
        conn.execute(
            """
            INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type)
            VALUES ('rel_valid_1', 'crs_a', 'cpt_a1', 'cpt_a2', 'prerequisite');
            """
        )

        # Attempt to change course_id to crs_b where cpt_a1 and cpt_a2 do not belong
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("UPDATE concept_relationships SET course_id = 'crs_b' WHERE id = 'rel_valid_1';")
        assert "Cross-course concept relationship rejected" in str(exc_info.value)

        # Attempt to change target_concept_id to cpt_b1 from a different course
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("UPDATE concept_relationships SET target_concept_id = 'cpt_b1' WHERE id = 'rel_valid_1';")
        assert "Cross-course concept relationship rejected" in str(exc_info.value)


def test_sql_self_edge_insertion_rejected(fresh_db):
    """A concept cannot have a relationship with itself."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_1', 'Course 1');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_1', 'crs_1', 'Loops', 'loops');")

        # Prerequisite self edge
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type)
                VALUES ('rel_self_1', 'crs_1', 'cpt_1', 'cpt_1', 'prerequisite');
                """
            )

        # Related self edge
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type)
                VALUES ('rel_self_2', 'crs_1', 'cpt_1', 'cpt_1', 'related');
                """
            )


def test_sql_enum_and_confidence_bounds_enforced(fresh_db):
    """Table CHECK constraints reject invalid enums and out-of-range confidence values."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_1', 'Course 1');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('c1', 'crs_1', 'C1', 'c1');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('c2', 'crs_1', 'C2', 'c2');")

        # Invalid relationship_type
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type) "
                "VALUES ('r1', 'crs_1', 'c1', 'c2', 'hierarchical');"
            )

        # Invalid review_status
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type, review_status) "
                "VALUES ('r2', 'crs_1', 'c1', 'c2', 'prerequisite', 'pending_approval');"
            )

        # Invalid provenance
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type, provenance) "
                "VALUES ('r3', 'crs_1', 'c1', 'c2', 'prerequisite', 'external_web');"
            )

        # Confidence > 1.0
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type, confidence) "
                "VALUES ('r4', 'crs_1', 'c1', 'c2', 'prerequisite', 1.25);"
            )

        # Confidence < 0.0
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type, confidence) "
                "VALUES ('r5', 'crs_1', 'c1', 'c2', 'prerequisite', -0.5);"
            )


def test_sql_cycle_rejected_on_prerequisite_insert(fresh_db):
    """Trigger trg_concept_rel_insert_cycle_check detects and aborts cycles on direct SQL insert."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_dag', 'Algorithms');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('A', 'crs_dag', 'Concept A', 'a');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('B', 'crs_dag', 'Concept B', 'b');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('C', 'crs_dag', 'Concept C', 'c');")

        # Insert A -> B, B -> C
        conn.execute("INSERT INTO concept_relationships VALUES ('r_ab', 'crs_dag', 'A', 'B', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")
        conn.execute("INSERT INTO concept_relationships VALUES ('r_bc', 'crs_dag', 'B', 'C', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")

        # Now attempt C -> A (cycle: A -> B -> C -> A)
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("INSERT INTO concept_relationships VALUES ('r_ca', 'crs_dag', 'C', 'A', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")
        assert "Cycle detected in prerequisite graph" in str(exc_info.value)

        # Verify r_ca was not inserted
        assert conn.execute("SELECT COUNT(*) FROM concept_relationships WHERE id = 'r_ca';").fetchone()[0] == 0


def test_sql_cycle_rejected_on_updating_proposed_to_accepted(fresh_db):
    """Trigger trg_concept_rel_update_cycle_check rejects updating a proposed cycle edge to accepted."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_dag2', 'Data Structures');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('X', 'crs_dag2', 'X', 'x');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('Y', 'crs_dag2', 'Y', 'y');")

        # X -> Y accepted
        conn.execute("INSERT INTO concept_relationships VALUES ('r_xy', 'crs_dag2', 'X', 'Y', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")

        # Y -> X proposed (allowed while proposed)
        conn.execute("INSERT INTO concept_relationships VALUES ('r_yx', 'crs_dag2', 'Y', 'X', 'prerequisite', 'model_proposed', NULL, 0.8, 'proposed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")
        assert conn.execute("SELECT review_status FROM concept_relationships WHERE id = 'r_yx';").fetchone()[0] == "proposed"

        # Update proposed -> accepted MUST be rejected
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("UPDATE concept_relationships SET review_status = 'accepted' WHERE id = 'r_yx';")
        assert "Cycle detected in prerequisite graph" in str(exc_info.value)

        # Verify status remains 'proposed' and was not partially committed
        status = conn.execute("SELECT review_status FROM concept_relationships WHERE id = 'r_yx';").fetchone()[0]
        assert status == "proposed"


def test_sql_valid_dag_and_related_bidirectional_succeed(fresh_db):
    """Diamond DAGs and bidirectional related relationships are completely valid."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_valid', 'Calculus');")
        for name in ["Limits", "Derivatives", "Integrals", "FTC"]:
            conn.execute(f"INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('{name}', 'crs_valid', '{name}', '{name.lower()}');")

        # Diamond DAG: Limits -> Derivatives, Limits -> Integrals, Derivatives -> FTC, Integrals -> FTC
        conn.execute("INSERT INTO concept_relationships VALUES ('e1', 'crs_valid', 'Limits', 'Derivatives', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")
        conn.execute("INSERT INTO concept_relationships VALUES ('e2', 'crs_valid', 'Limits', 'Integrals', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")
        conn.execute("INSERT INTO concept_relationships VALUES ('e3', 'crs_valid', 'Derivatives', 'FTC', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")
        conn.execute("INSERT INTO concept_relationships VALUES ('e4', 'crs_valid', 'Integrals', 'FTC', 'prerequisite', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")

        # Bidirectional related relationships between Derivatives and Integrals
        conn.execute("INSERT INTO concept_relationships VALUES ('e5', 'crs_valid', 'Derivatives', 'Integrals', 'related', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")
        conn.execute("INSERT INTO concept_relationships VALUES ('e6', 'crs_valid', 'Integrals', 'Derivatives', 'related', 'manual', NULL, 1.0, 'accepted', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);")

        assert conn.execute("SELECT COUNT(*) FROM concept_relationships WHERE course_id = 'crs_valid';").fetchone()[0] == 6


def test_sql_concept_course_immutability_with_relationships_and_chunks(fresh_db):
    """Changing a concept's course_id is rejected if it has relationships or chunk provenance."""
    with fresh_db.get_connection() as conn:
        # 1. Create concepts in two different courses
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_101', 'Intro CS'), ('crs_201', 'Data Structures');")
        conn.execute("INSERT INTO documents (id, course_id, filename, file_type, file_size_bytes, sha256_hash, status) VALUES ('doc_1', 'crs_101', 'syllabus.txt', 'txt', 100, 'h1', 'ready');")
        conn.execute(
            """
            INSERT INTO chunks (id, document_id, course_id, chunk_index, location_type, location_index, content, token_count, content_hash)
            VALUES ('chk_1', 'doc_1', 'crs_101', 0, 'section', 1, 'Variables and control flow.', 10, 'h_chk1');
            """
        )

        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_var', 'crs_101', 'Variables', 'variables');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_loop', 'crs_101', 'Loops', 'loops');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_unlinked', 'crs_101', 'Unlinked', 'unlinked');")

        # 2. Create a valid prerequisite relationship in the first course (Variables -> Loops)
        conn.execute(
            """
            INSERT INTO concept_relationships (id, course_id, source_concept_id, target_concept_id, relationship_type)
            VALUES ('rel_var_loop', 'crs_101', 'cpt_var', 'cpt_loop', 'prerequisite');
            """
        )

        # 3. Attach source-chunk provenance to cpt_loop
        conn.execute("INSERT INTO concept_chunks (concept_id, chunk_id) VALUES ('cpt_loop', 'chk_1');")

        # 4. Attempt to change cpt_var course_id so the relationship would become cross-course
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("UPDATE concepts SET course_id = 'crs_201' WHERE id = 'cpt_var';")
        assert "Cannot update concept course_id" in str(exc_info.value)

        # Attempt to change cpt_loop course_id so relationship and chunk provenance would become cross-course
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("UPDATE concepts SET course_id = 'crs_201' WHERE id = 'cpt_loop';")
        assert "Cannot update concept course_id" in str(exc_info.value)

        # 5. Confirm original concepts, relationships, and provenance remain intact
        cpt_var = conn.execute("SELECT course_id FROM concepts WHERE id = 'cpt_var';").fetchone()
        assert cpt_var["course_id"] == "crs_101"

        cpt_loop = conn.execute("SELECT course_id FROM concepts WHERE id = 'cpt_loop';").fetchone()
        assert cpt_loop["course_id"] == "crs_101"

        rel = conn.execute("SELECT course_id FROM concept_relationships WHERE id = 'rel_var_loop';").fetchone()
        assert rel["course_id"] == "crs_101"

        cc = conn.execute("SELECT * FROM concept_chunks WHERE concept_id = 'cpt_loop';").fetchone()
        assert cc["chunk_id"] == "chk_1"

        # 6. Confirm ordinary valid operation: unlinked concept with no relationships/chunks CAN have course updated
        conn.execute("UPDATE concepts SET course_id = 'crs_201' WHERE id = 'cpt_unlinked';")
        cpt_unlinked = conn.execute("SELECT course_id FROM concepts WHERE id = 'cpt_unlinked';").fetchone()
        assert cpt_unlinked["course_id"] == "crs_201"


def test_sql_cross_course_chunk_provenance_rejected(fresh_db):
    """Direct SQL insertion or update of concept_chunks with mismatched course_id is rejected."""
    with fresh_db.get_connection() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('crs_a', 'Course A'), ('crs_b', 'Course B');")
        conn.execute("INSERT INTO documents (id, course_id, filename, file_type, file_size_bytes, sha256_hash, status) VALUES ('doc_a', 'crs_a', 'a.txt', 'txt', 50, 'ha', 'ready'), ('doc_b', 'crs_b', 'b.txt', 'txt', 50, 'hb', 'ready');")
        conn.execute("INSERT INTO chunks (id, document_id, course_id, chunk_index, location_type, location_index, content, token_count, content_hash) VALUES ('chk_a', 'doc_a', 'crs_a', 0, 'section', 1, 'Content A', 5, 'ha1'), ('chk_b', 'doc_b', 'crs_b', 0, 'section', 1, 'Content B', 5, 'hb1');")
        conn.execute("INSERT INTO concepts (id, course_id, name, normalized_name) VALUES ('cpt_a', 'crs_a', 'Concept A', 'concept a');")

        # Attempt to map cpt_a (in crs_a) to chk_b (in crs_b)
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute("INSERT INTO concept_chunks (concept_id, chunk_id) VALUES ('cpt_a', 'chk_b');")
        assert "Cross-course chunk provenance rejected" in str(exc_info.value)


