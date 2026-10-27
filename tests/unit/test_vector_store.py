"""Unit tests for ChromaVectorStore synchronization, filtering, and model compatibility."""

from __future__ import annotations

import pytest

from studyos.ingestion.models import DocumentChunk
from studyos.retrieval.embedding import FakeEmbeddingProvider
from studyos.retrieval.vector_store import (
    ChromaVectorStore,
    IncompatibleCollectionError,
    VectorStoreError,
)
from studyos.storage.db import Database


@pytest.fixture
def test_db(tmp_path) -> Database:
    db = Database(tmp_path / "test_store.db")
    db.migrate()
    yield db
    db.close()


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(model_name="fake-minilm", dimension=384)


class TestChromaVectorStore:
    def test_empty_database_and_collection(self, tmp_path, test_db, fake_provider):
        """Syncing an empty SQLite database creates collection and reports zero changes cleanly."""
        store = ChromaVectorStore(
            chroma_dir=tmp_path / "chroma_empty",
            provider=fake_provider,
        )
        try:
            stats = store.sync(test_db)
            assert stats.total_chunks == 0
            assert stats.added == 0
            assert stats.deleted == 0
            assert stats.updated == 0
            assert stats.unchanged == 0
            assert store.count() == 0

            # Searching empty store returns empty list
            candidates = store.search_candidates(fake_provider.embed_query("test query"), top_k=5)
            assert candidates == []
        finally:
            store.close()

    def test_sync_upsert_and_idempotency(self, tmp_path, test_db, fake_provider):
        """Syncing inserts new chunks, and repeated sync without changes is a no-op (idempotent)."""
        course = test_db.create_course("Operating Systems", "CS301")
        doc = test_db.create_document(
            course_id=course.id,
            filename="kernel.pdf",
            file_type="pdf",
            file_size_bytes=2048,
            sha256_hash="hash_k1",
        )
        c1 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Process scheduling and context switching.",
            token_count=5,
        )
        c2 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=1,
            location_type="page",
            location_index=2,
            content="Virtual memory and page replacement policies.",
            token_count=6,
        )
        test_db.insert_chunks([c1, c2])

        store = ChromaVectorStore(
            chroma_dir=tmp_path / "chroma_sync",
            provider=fake_provider,
        )
        try:
            # First sync: both chunks added
            s1 = store.sync(test_db)
            assert s1.added == 2
            assert s1.updated == 0
            assert s1.deleted == 0
            assert s1.unchanged == 0
            assert store.count() == 2

            # Second sync: idempotent, zero additions or updates
            s2 = store.sync(test_db)
            assert s2.added == 0
            assert s2.updated == 0
            assert s2.deleted == 0
            assert s2.unchanged == 2
            assert store.count() == 2
        finally:
            store.close()

    def test_reconciliation_detects_changed_content(self, tmp_path, test_db, fake_provider):
        """When a chunk's content changes, sync updates its vector and metadata."""
        course = test_db.create_course("Compilers", "CS401")
        doc = test_db.create_document(
            course_id=course.id,
            filename="lexer.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            sha256_hash="hash_lex",
        )
        c1 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Lexical analysis with deterministic finite automata.",
            token_count=6,
        )
        test_db.insert_chunks([c1])

        store = ChromaVectorStore(
            chroma_dir=tmp_path / "chroma_change",
            provider=fake_provider,
        )
        try:
            store.sync(test_db)
            assert store.count() == 1

            # Update chunk content and content_hash in SQLite
            new_content = "Lexical analysis using regex and non-deterministic finite automata (NFA)."
            import hashlib
            new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            with test_db.get_connection() as conn:
                conn.execute(
                    "UPDATE chunks SET content = ?, content_hash = ? WHERE id = ?;",
                    (new_content, new_hash, c1.chunk_id),
                )

            # Sync detects change and updates
            s2 = store.sync(test_db)
            assert s2.updated == 1
            assert s2.added == 0
            assert s2.deleted == 0
            assert s2.unchanged == 0
            assert store.count() == 1
        finally:
            store.close()

    def test_stale_record_cleanup(self, tmp_path, test_db, fake_provider):
        """When a document is deleted in SQLite, sync deletes its orphaned vectors from Chroma."""
        course = test_db.create_course("Databases", "CS303")
        doc = test_db.create_document(
            course_id=course.id,
            filename="btree.pdf",
            file_type="pdf",
            file_size_bytes=1000,
            sha256_hash="hash_btree",
        )
        c1 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="B-tree indexing and disk paging.",
            token_count=5,
        )
        test_db.insert_chunks([c1])

        store = ChromaVectorStore(
            chroma_dir=tmp_path / "chroma_del",
            provider=fake_provider,
        )
        try:
            store.sync(test_db)
            assert store.count() == 1

            # Delete document in SQLite (cascades deletion to chunks)
            test_db.delete_document(doc.id)
            assert len(test_db.get_chunks_for_document(doc.id)) == 0

            # Sync detects stale Chroma record and deletes it
            s2 = store.sync(test_db)
            assert s2.deleted == 1
            assert s2.added == 0
            assert store.count() == 0
        finally:
            store.close()

    def test_persistent_collection_reopening(self, tmp_path, test_db, fake_provider):
        """A persistent Chroma collection can be closed and reopened with data intact."""
        chroma_dir = tmp_path / "chroma_persist"
        course = test_db.create_course("Distributed Systems", "CS404")
        doc = test_db.create_document(
            course_id=course.id,
            filename="raft.pdf",
            file_type="pdf",
            file_size_bytes=2048,
            sha256_hash="hash_raft",
        )
        c1 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Raft consensus algorithm: leader election and log replication.",
            token_count=8,
        )
        test_db.insert_chunks([c1])

        # Session 1: create and sync
        s1 = ChromaVectorStore(chroma_dir=chroma_dir, provider=fake_provider)
        s1.sync(test_db)
        assert s1.count() == 1
        s1.close()

        # Session 2: reopen same directory
        s2 = ChromaVectorStore(chroma_dir=chroma_dir, provider=fake_provider)
        try:
            assert s2.count() == 1
            # Search finds candidate
            candidates = s2.search_candidates(fake_provider.embed_query("consensus algorithm"), top_k=5)
            assert len(candidates) == 1
            assert candidates[0][0] == c1.chunk_id
        finally:
            s2.close()

    def test_valid_collection_with_matching_metadata(self, tmp_path, test_db, fake_provider):
        """Opening an existing collection with matching metadata succeeds normally."""
        chroma_dir = tmp_path / "chroma_valid_meta"
        s1 = ChromaVectorStore(chroma_dir=chroma_dir, provider=fake_provider)
        try:
            s1.get_or_create_collection()
            col = s1.client.get_collection(s1.collection_name)
            assert col.metadata.get("embedding_model") == fake_provider.model_name
            assert col.metadata.get("embedding_dimension") == fake_provider.dimension
        finally:
            s1.close()

        # Reopen in a second session with same provider
        s2 = ChromaVectorStore(chroma_dir=chroma_dir, provider=fake_provider)
        try:
            col2 = s2.get_or_create_collection()
            assert col2.metadata.get("embedding_model") == fake_provider.model_name
        finally:
            s2.close()

    def test_model_incompatibility_detection_and_rebuild(self, tmp_path, test_db):
        """Detect model mismatch on existing collection and safely rebuild through explicit operation."""
        chroma_dir = tmp_path / "chroma_mismatch"
        provider_a = FakeEmbeddingProvider(model_name="model-alpha", dimension=384)
        provider_b = FakeEmbeddingProvider(model_name="model-beta", dimension=384)

        course = test_db.create_course("AI", "CS405")
        doc = test_db.create_document(
            course_id=course.id,
            filename="search.pdf",
            file_type="pdf",
            file_size_bytes=1000,
            sha256_hash="hash_search",
        )
        c1 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="A* search and heuristic admissibility.",
            token_count=6,
        )
        test_db.insert_chunks([c1])

        # Step 1: index with model-alpha
        s1 = ChromaVectorStore(chroma_dir=chroma_dir, provider=provider_a)
        s1.sync(test_db)
        assert s1.count() == 1
        s1.close()

        # Step 2: attempt to open collection with model-beta -> raises IncompatibleCollectionError
        s2 = ChromaVectorStore(chroma_dir=chroma_dir, provider=provider_b)
        try:
            with pytest.raises(IncompatibleCollectionError, match="model-alpha"):
                s2.sync(test_db)

            # Step 3: explicit rebuild succeeds, replacing index with model-beta
            rebuild_stats = s2.rebuild(test_db)
            assert rebuild_stats.added == 1
            assert s2.count() == 1
            col = s2.get_or_create_collection()
            assert col.metadata.get("embedding_model") == "model-beta"

            # Subsequent sync is idempotent
            s3 = s2.sync(test_db)
            assert s3.unchanged == 1
            assert s3.added == 0
        finally:
            s2.close()

    def test_dimension_incompatibility_detection(self, tmp_path, test_db):
        """Detect dimension mismatch on existing collection and reject reuse."""
        chroma_dir = tmp_path / "chroma_dim_mismatch"
        provider_384 = FakeEmbeddingProvider(model_name="test-model", dimension=384)
        provider_768 = FakeEmbeddingProvider(model_name="test-model", dimension=768)

        s1 = ChromaVectorStore(chroma_dir=chroma_dir, provider=provider_384)
        s1.get_or_create_collection()
        s1.close()

        # Reopen with dimension 768
        s2 = ChromaVectorStore(chroma_dir=chroma_dir, provider=provider_768)
        try:
            with pytest.raises(IncompatibleCollectionError, match="which does not match current model dimension 768"):
                s2.get_or_create_collection()
        finally:
            s2.close()

    def test_absent_or_empty_metadata_fails_closed(self, tmp_path, test_db, fake_provider):
        """Pre-existing collections with absent (None) metadata fail closed on reuse."""
        chroma_dir = tmp_path / "chroma_no_meta"
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.create_collection("studyos_chunks", metadata=None)
        client.close()

        # Opening with provider must reject absent metadata
        store = ChromaVectorStore(chroma_dir=chroma_dir, provider=fake_provider)
        try:
            with pytest.raises(IncompatibleCollectionError, match="absent or empty compatibility metadata"):
                store.get_or_create_collection()
        finally:
            store.close()

    def test_partial_or_malformed_metadata_fails_closed(self, tmp_path, test_db, fake_provider):
        """Collections with partial or malformed metadata fail closed."""
        chroma_dir = tmp_path / "chroma_partial_meta"
        import chromadb

        # 1. Missing dimension
        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.create_collection("col_no_dim", metadata={"embedding_model": fake_provider.model_name})
        client.close()

        s1 = ChromaVectorStore(chroma_dir=chroma_dir, collection_name="col_no_dim", provider=fake_provider)
        try:
            with pytest.raises(IncompatibleCollectionError, match="incomplete compatibility metadata"):
                s1.get_or_create_collection()
        finally:
            s1.close()

        # 2. Missing model
        client2 = chromadb.PersistentClient(path=str(chroma_dir))
        client2.create_collection("col_no_model", metadata={"embedding_dimension": fake_provider.dimension})
        client2.close()

        s2 = ChromaVectorStore(chroma_dir=chroma_dir, collection_name="col_no_model", provider=fake_provider)
        try:
            with pytest.raises(IncompatibleCollectionError, match="incomplete compatibility metadata"):
                s2.get_or_create_collection()
        finally:
            s2.close()

        # 3. Malformed dimension string
        client3 = chromadb.PersistentClient(path=str(chroma_dir))
        client3.create_collection(
            "col_bad_dim",
            metadata={"embedding_model": fake_provider.model_name, "embedding_dimension": "not_an_int"},
        )
        client3.close()

        s3 = ChromaVectorStore(chroma_dir=chroma_dir, collection_name="col_bad_dim", provider=fake_provider)
        try:
            with pytest.raises(IncompatibleCollectionError, match="malformed embedding dimension"):
                s3.get_or_create_collection()
        finally:
            s3.close()

    def test_rebuild_recovers_unknown_collection_and_preserves_sqlite(self, tmp_path, test_db, fake_provider):
        """Explicit rebuild recovers an unknown collection and preserves SQLite ground truth."""
        chroma_dir = tmp_path / "chroma_rebuild_preserve"

        # Populate SQLite with courses and chunks
        course = test_db.create_course("Computer Architecture", "CS201")
        doc = test_db.create_document(course_id=course.id, filename="cpu.pdf", file_type="pdf", file_size_bytes=1000, sha256_hash="h_cpu")
        c1 = DocumentChunk.create(
            document_id=doc.id,
            course_id=course.id,
            chunk_index=0,
            location_type="page",
            location_index=1,
            content="Pipelining hazards: data, control, and structural.",
            token_count=7,
        )
        test_db.insert_chunks([c1])

        # Create unknown collection in Chroma without metadata
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.create_collection("studyos_chunks", metadata=None)
        client.close()

        store = ChromaVectorStore(chroma_dir=chroma_dir, provider=fake_provider)
        try:
            # Normal sync fails closed
            with pytest.raises(IncompatibleCollectionError):
                store.sync(test_db)

            # Explicit rebuild succeeds
            stats = store.rebuild(test_db)
            assert stats.added == 1
            assert store.count() == 1

            # Verify collection now has valid tagged metadata
            col = store.get_or_create_collection()
            assert col.metadata.get("embedding_model") == fake_provider.model_name
            assert col.metadata.get("embedding_dimension") == fake_provider.dimension

            # Verify SQLite data survived completely intact
            sqlite_doc = test_db.get_document(doc.id)
            assert sqlite_doc is not None
            assert sqlite_doc.filename == "cpu.pdf"
            sqlite_chunks = test_db.get_chunks_for_document(doc.id)
            assert len(sqlite_chunks) == 1
            assert sqlite_chunks[0].content == c1.content
        finally:
            store.close()

    def test_search_candidates_filtering(self, tmp_path, test_db, fake_provider):
        """Vector candidate search respects course and document metadata filters."""
        c_os = test_db.create_course("Operating Systems", "CS301")
        c_db = test_db.create_course("Databases", "CS303")

        d_os1 = test_db.create_document(course_id=c_os.id, filename="os1.pdf", file_type="pdf", file_size_bytes=500, sha256_hash="h_os1")
        d_os2 = test_db.create_document(course_id=c_os.id, filename="os2.pdf", file_type="pdf", file_size_bytes=500, sha256_hash="h_os2")
        d_db1 = test_db.create_document(course_id=c_db.id, filename="db1.pdf", file_type="pdf", file_size_bytes=500, sha256_hash="h_db1")

        chk_os1 = DocumentChunk.create(document_id=d_os1.id, course_id=c_os.id, chunk_index=0, location_type="page", location_index=1, content="OS scheduling queues", token_count=4)
        chk_os2 = DocumentChunk.create(document_id=d_os2.id, course_id=c_os.id, chunk_index=0, location_type="page", location_index=1, content="OS memory paging", token_count=4)
        chk_db1 = DocumentChunk.create(document_id=d_db1.id, course_id=c_db.id, chunk_index=0, location_type="page", location_index=1, content="Database buffer pool", token_count=4)

        test_db.insert_chunks([chk_os1, chk_os2, chk_db1])

        store = ChromaVectorStore(chroma_dir=tmp_path / "chroma_filters", provider=fake_provider)
        try:
            store.sync(test_db)
            assert store.count() == 3

            q_vec = fake_provider.embed_query("scheduling memory buffer")

            # 1. Filter by Course OS: only os1 and os2
            cands_os = store.search_candidates(q_vec, top_k=10, course_ids=[c_os.id])
            ids_os = [c[0] for c in cands_os]
            assert set(ids_os) == {chk_os1.chunk_id, chk_os2.chunk_id}
            assert chk_db1.chunk_id not in ids_os

            # 2. Filter by Document OS1: only os1
            cands_doc = store.search_candidates(q_vec, top_k=10, document_ids=[d_os1.id])
            ids_doc = [c[0] for c in cands_doc]
            assert ids_doc == [chk_os1.chunk_id]

            # 3. Explicit empty filter list returns empty
            assert store.search_candidates(q_vec, top_k=10, course_ids=[]) == []
            assert store.search_candidates(q_vec, top_k=10, document_ids=[]) == []
        finally:
            store.close()
