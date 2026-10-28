"""Unit tests for direct concept discovery from raw ingested document chunks."""

import pytest
from studyos.inference.fake import FakeChatProvider
from studyos.ingestion.models import DocumentChunk
from studyos.learner.discovery import ConceptDiscoveryService
from studyos.learner.models import ConceptProvenance, ConceptReviewStatus
from studyos.learner.repository import LearnerRepository
from studyos.storage.db import Database


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_discovery.db"
    db = Database(db_file)
    db.migrate()
    return db


def test_discover_concepts_from_raw_chunks_without_exams(test_db):
    """Concept discovery identifies concepts from raw chunks without requiring any exams."""
    crs = test_db.create_course("Algorithms", "CS101")
    doc = test_db.create_document(crs.id, "graph_theory.pdf", "pdf", 1024, "hash1", "ready")

    chunk1 = DocumentChunk(
        chunk_id="chk_dfs_1",
        document_id=doc.id,
        course_id=crs.id,
        chunk_index=0,
        location_type="section",
        location_index=1,
        location_title="Depth-First Search",
        content="Depth-First Search (DFS) traverses graph edges using a stack or recursion.",
        token_count=30,
        content_hash="hash_dfs",
    )
    chunk2 = DocumentChunk(
        chunk_id="chk_bfs_1",
        document_id=doc.id,
        course_id=crs.id,
        chunk_index=1,
        location_type="section",
        location_index=2,
        location_title="Breadth-First Search",
        content="Breadth-First Search (BFS) explores vertices layer by layer using a FIFO queue.",
        token_count=30,
        content_hash="hash_bfs",
    )
    test_db.insert_chunks([chunk1, chunk2])

    # Fake chat provider returning structured JSON concepts
    fake_response = """
    {
      "concepts": [
        {
          "name": "Depth-First Search",
          "description": "Recursive graph traversal exploring branch depth before backtracking.",
          "learning_objectives": ["Implement DFS using recursion", "Detect graph cycles"],
          "supporting_chunk_ids": ["chk_dfs_1"]
        },
        {
          "name": "Breadth-First Search",
          "description": "Queue-based graph traversal visiting neighbor vertices level by level.",
          "learning_objectives": ["Implement BFS using queues", "Find unweighted shortest paths"],
          "supporting_chunk_ids": ["chk_bfs_1"]
        }
      ]
    }
    """
    fake_chat = FakeChatProvider([fake_response])
    service = ConceptDiscoveryService(db=test_db, chat_provider=fake_chat)

    result = service.discover_concepts(course_id=crs.id, auto_accept=True)
    assert result.status == "completed"
    assert result.total_discovered == 2
    assert result.new_concepts_created == 2

    # Verify concepts in database
    repo = LearnerRepository(test_db)
    concepts = repo.list_concepts(course_id=crs.id)
    assert len(concepts) == 2
    c_names = {c.name for c in concepts}
    assert "Depth-First Search" in c_names
    assert "Breadth-First Search" in c_names

    # Verify chunk provenance preserved in concept_chunks
    c_dfs = [c for c in concepts if c.name == "Depth-First Search"][0]
    chunks = repo.get_chunks_for_concept(c_dfs.id)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chk_dfs_1"
    assert chunks[0]["is_live"] is True


def test_concept_deduplication_and_idempotency_within_course(test_db):
    """Re-running discovery against unchanged or repeated concepts is idempotent."""
    crs = test_db.create_course("Operating Systems", "CS301")
    doc = test_db.create_document(crs.id, "concurrency.pdf", "pdf", 2048, "hash2", "ready")

    chunk1 = DocumentChunk(
        chunk_id="chk_mutex_1",
        document_id=doc.id,
        course_id=crs.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        location_title="Mutual Exclusion",
        content="Mutual exclusion guarantees that only one thread accesses a critical section.",
        token_count=25,
        content_hash="hash_m1",
    )
    test_db.insert_chunks([chunk1])

    fake_resp = """
    {
      "concepts": [
        {
          "name": "Mutual Exclusion",
          "description": "Synchronization primitive ensuring single-thread access.",
          "learning_objectives": ["Prevent race conditions"],
          "supporting_chunk_ids": ["chk_mutex_1"]
        }
      ]
    }
    """
    fake_chat = FakeChatProvider([fake_resp, fake_resp])
    service = ConceptDiscoveryService(db=test_db, chat_provider=fake_chat)

    # First run creates 1 concept
    res1 = service.discover_concepts(course_id=crs.id, auto_accept=True)
    assert res1.new_concepts_created == 1
    assert res1.existing_concepts_updated == 0

    # Second run updates existing concept without duplication
    res2 = service.discover_concepts(course_id=crs.id, auto_accept=True)
    assert res2.new_concepts_created == 0
    assert res2.existing_concepts_updated == 1

    repo = LearnerRepository(test_db)
    concepts = repo.list_concepts(course_id=crs.id)
    assert len(concepts) == 1
    assert concepts[0].name == "Mutual Exclusion"


def test_course_isolation_across_identical_concept_names(test_db):
    """Concepts with identical names in different courses remain strictly isolated."""
    crs1 = test_db.create_course("Physics", "PHYS101")
    crs2 = test_db.create_course("Calculus", "MATH101")

    doc1 = test_db.create_document(crs1.id, "vectors.pdf", "pdf", 1024, "h1", "ready")
    doc2 = test_db.create_document(crs2.id, "linear_algebra.pdf", "pdf", 1024, "h2", "ready")

    c1 = DocumentChunk(
        chunk_id="chk_phys_v",
        document_id=doc1.id,
        course_id=crs1.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        location_title="Vectors",
        content="Vectors possess magnitude and spatial direction.",
        token_count=20,
        content_hash="ch1",
    )
    c2 = DocumentChunk(
        chunk_id="chk_math_v",
        document_id=doc2.id,
        course_id=crs2.id,
        chunk_index=0,
        location_type="page",
        location_index=1,
        location_title="Vectors",
        content="Vectors are elements of a vector space over a field.",
        token_count=20,
        content_hash="ch2",
    )
    test_db.insert_chunks([c1, c2])

    resp1 = '{"concepts": [{"name": "Vectors", "description": "Physics vectors", "supporting_chunk_ids": ["chk_phys_v"]}]}'
    resp2 = '{"concepts": [{"name": "Vectors", "description": "Linear algebra vectors", "supporting_chunk_ids": ["chk_math_v"]}]}'
    fake_chat = FakeChatProvider([resp1, resp2])
    service = ConceptDiscoveryService(db=test_db, chat_provider=fake_chat)

    service.discover_concepts(course_id=crs1.id, auto_accept=True)
    service.discover_concepts(course_id=crs2.id, auto_accept=True)

    repo = LearnerRepository(test_db)
    phys_cpts = repo.list_concepts(course_id=crs1.id)
    math_cpts = repo.list_concepts(course_id=crs2.id)

    assert len(phys_cpts) == 1
    assert len(math_cpts) == 1
    assert phys_cpts[0].id != math_cpts[0].id
    assert phys_cpts[0].course_id == crs1.id
    assert math_cpts[0].course_id == crs2.id


def test_malformed_model_output_fails_safely_to_metadata(test_db):
    """When LLM returns malformed JSON or errors, discovery falls back to trusted metadata."""
    crs = test_db.create_course("Networks", "CS401")
    doc = test_db.create_document(crs.id, "transport.pdf", "pdf", 1024, "h_net", "ready")

    chunk = DocumentChunk(
        chunk_id="chk_tcp_1",
        document_id=doc.id,
        course_id=crs.id,
        chunk_index=0,
        location_type="section",
        location_index=1,
        location_title="TCP Handshake",
        content="TCP utilizes a 3-way handshake (SYN, SYN-ACK, ACK) to establish reliable connections.",
        token_count=30,
        content_hash="ch_tcp",
    )
    test_db.insert_chunks([chunk])

    # Fake chat returning invalid non-JSON garbage
    fake_chat = FakeChatProvider(["Sorry, I cannot produce valid JSON right now!"])
    service = ConceptDiscoveryService(db=test_db, chat_provider=fake_chat)

    res = service.discover_concepts(course_id=crs.id, auto_accept=True)
    assert res.status == "completed"
    assert res.total_discovered == 1

    repo = LearnerRepository(test_db)
    concepts = repo.list_concepts(course_id=crs.id)
    assert len(concepts) == 1
    assert concepts[0].name == "TCP Handshake"


def test_ollama_unavailable_graceful_fallback(test_db):
    """When no chat provider is configured, discovery operates 100% offline via chunk metadata."""
    crs = test_db.create_course("Security", "CS501")
    doc = test_db.create_document(crs.id, "crypto.pdf", "pdf", 1024, "h_sec", "ready")

    chunk = DocumentChunk(
        chunk_id="chk_rsa_1",
        document_id=doc.id,
        course_id=crs.id,
        chunk_index=0,
        location_type="section",
        location_index=1,
        location_title="Asymmetric Encryption",
        content="Asymmetric encryption relies on public and private key pairs.",
        token_count=25,
        content_hash="ch_rsa",
    )
    test_db.insert_chunks([chunk])

    # chat_provider is None
    service = ConceptDiscoveryService(db=test_db, chat_provider=None)
    res = service.discover_concepts(course_id=crs.id, auto_accept=False)
    assert res.status == "completed"
    assert res.total_discovered == 1

    repo = LearnerRepository(test_db)
    concepts = repo.list_concepts(course_id=crs.id)
    assert len(concepts) == 1
    assert concepts[0].name == "Asymmetric Encryption"
    assert concepts[0].review_status == ConceptReviewStatus.PROPOSED


def test_discovery_repeat_run_preserves_accepted_status_and_chunk_provenance(test_db):
    """Repeat discovery run preserves explicitly accepted concepts and avoids duplicate provenance links."""
    crs = test_db.create_course("Distributed Systems", "CS601")
    doc = test_db.create_document(crs.id, "consensus.pdf", "pdf", 1024, "h_raft", "ready")

    chunk = DocumentChunk(
        chunk_id="chk_raft_1",
        document_id=doc.id,
        course_id=crs.id,
        chunk_index=0,
        location_type="section",
        location_index=1,
        location_title="Raft Consensus",
        content="Raft is a consensus algorithm designed to be easy to understand.",
        token_count=20,
        content_hash="ch_raft",
    )
    test_db.insert_chunks([chunk])

    service = ConceptDiscoveryService(db=test_db, chat_provider=None)

    # 1. First run: proposes concept
    res1 = service.discover_concepts(course_id=crs.id, auto_accept=False)
    assert res1.new_concepts_created == 1
    assert res1.existing_concepts_updated == 0

    repo = LearnerRepository(test_db)
    concepts = repo.list_concepts(course_id=crs.id)
    assert len(concepts) == 1
    cpt = concepts[0]
    assert cpt.review_status == ConceptReviewStatus.PROPOSED

    # 2. User explicitly updates concept to ACCEPTED
    with test_db.get_connection() as conn:
        conn.execute("UPDATE concepts SET review_status = 'accepted' WHERE id = ?;", (cpt.id,))

    updated_cpt = repo.get_concept(cpt.id)
    assert updated_cpt.review_status == ConceptReviewStatus.ACCEPTED

    # Verify initial chunk mapping count
    with test_db.get_connection() as conn:
        initial_cc_count = conn.execute("SELECT COUNT(*) FROM concept_chunks;").fetchone()[0]
    assert initial_cc_count == 1

    # 3. Second run with auto_accept=False over unchanged material
    res2 = service.discover_concepts(course_id=crs.id, auto_accept=False)
    assert res2.new_concepts_created == 0
    assert res2.existing_concepts_updated == 1

    # Status must NOT be demoted back to 'proposed'
    cpt_after = repo.get_concept(cpt.id)
    assert cpt_after.review_status == ConceptReviewStatus.ACCEPTED

    # Provenance links in concept_chunks must NOT be duplicated
    with test_db.get_connection() as conn:
        final_cc_count = conn.execute("SELECT COUNT(*) FROM concept_chunks;").fetchone()[0]
    assert final_cc_count == 1
