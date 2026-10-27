"""Persistent ChromaDB vector index management, synchronization, and query services."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any
import chromadb
from chromadb.config import Settings

from studyos.core.config import AppConfig
from studyos.retrieval.embedding import EmbeddingProvider
from studyos.retrieval.models import SyncStats
from studyos.storage.db import Database


class VectorStoreError(Exception):
    """Base exception for vector index operations."""


class IncompatibleCollectionError(VectorStoreError):
    """Raised when the vector collection on disk was created with a different embedding model or dimension."""


class ChromaVectorStore:
    """Manages local persistent vector storage using ChromaDB.

    Enforces stable chunk IDs, SQLite metadata supremacy, model incompatibility detection,
    and idempotent incremental synchronization.
    """

    def __init__(
        self,
        chroma_dir: Path | str,
        collection_name: str = "studyos_chunks",
        provider: EmbeddingProvider | None = None,
        anonymized_telemetry: bool = False,
    ) -> None:
        self.chroma_dir = Path(chroma_dir).expanduser().resolve()
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.provider = provider
        self.anonymized_telemetry = anonymized_telemetry

        # Disable external telemetry for privacy
        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        settings = Settings(
            anonymized_telemetry=anonymized_telemetry,
            is_persistent=True,
        )
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir), settings=settings)
        self._collection = None

    @classmethod
    def from_config(cls, config: AppConfig, provider: EmbeddingProvider | None = None) -> ChromaVectorStore:
        """Instantiate ChromaVectorStore from application configuration."""
        return cls(
            chroma_dir=config.chroma_dir or (config.data_dir / "chroma"),
            collection_name=config.chroma_collection_name,
            provider=provider,
            anonymized_telemetry=config.anonymized_telemetry,
        )

    def close(self) -> None:
        """Explicitly close the persistent ChromaDB client to release Windows file locks."""
        if hasattr(self.client, "close"):
            self.client.close()

    def get_or_create_collection(self) -> chromadb.Collection:
        """Retrieve or initialize the Chroma collection and verify model compatibility."""
        if self._collection is not None:
            return self._collection

        # Prepare metadata for tracking model identity
        expected_meta: dict[str, Any] = {}
        if self.provider is not None:
            expected_meta = {
                "embedding_model": self.provider.model_name,
                "embedding_dimension": self.provider.dimension,
            }

        existing_names = {c.name for c in self.client.list_collections()}

        if self.collection_name not in existing_names:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata=expected_meta or None,
            )
        else:
            collection = self.client.get_collection(self.collection_name)
            # Require complete, valid compatibility metadata when opening an existing collection
            if self.provider is not None:
                meta = collection.metadata
                if not meta or not isinstance(meta, dict):
                    raise IncompatibleCollectionError(
                        f"Vector collection '{self.collection_name}' has absent or empty compatibility metadata. "
                        f"Cannot safely verify compatibility with configured model '{self.provider.model_name}'. "
                        f"Run 'scripts/sync_index.py --rebuild' or call rebuild() to rebuild the index safely."
                    )

                stored_model = meta.get("embedding_model")
                stored_dim = meta.get("embedding_dimension")

                if not stored_model or stored_dim is None:
                    raise IncompatibleCollectionError(
                        f"Vector collection '{self.collection_name}' has incomplete compatibility metadata "
                        f"(embedding_model={stored_model!r}, embedding_dimension={stored_dim!r}). "
                        f"Run 'scripts/sync_index.py --rebuild' or call rebuild() to rebuild the index safely."
                    )

                if stored_model != self.provider.model_name:
                    raise IncompatibleCollectionError(
                        f"Vector collection '{self.collection_name}' was indexed with embedding model "
                        f"'{stored_model}', but current provider is configured for '{self.provider.model_name}'. "
                        f"Run 'scripts/sync_index.py --rebuild' or call rebuild() to re-index safely."
                    )

                try:
                    dim_int = int(stored_dim)
                except (ValueError, TypeError):
                    raise IncompatibleCollectionError(
                        f"Vector collection '{self.collection_name}' has malformed embedding dimension {stored_dim!r}. "
                        f"Run 'scripts/sync_index.py --rebuild' or call rebuild() to re-index safely."
                    )

                if dim_int != self.provider.dimension:
                    raise IncompatibleCollectionError(
                        f"Vector collection '{self.collection_name}' has dimension {stored_dim}, "
                        f"which does not match current model dimension {self.provider.dimension}. "
                        f"Run 'scripts/sync_index.py --rebuild' or call rebuild() to re-index safely."
                    )

        self._collection = collection
        return self._collection

    def count(self) -> int:
        """Return total number of vectors in the active collection."""
        col = self.get_or_create_collection()
        return col.count()

    def sync(self, db: Database, batch_size: int = 100) -> SyncStats:
        """Reconcile the vector index against SQLite source of truth.

        - Adds vectors for new chunks.
        - Updates vectors for chunks with changed content hashes.
        - Deletes vectors for chunks that no longer exist in SQLite.
        - Leaves unchanged chunks untouched (idempotent).
        """
        if self.provider is None:
            raise VectorStoreError("Cannot synchronize vector index without an EmbeddingProvider.")

        start_time = time.perf_counter()
        collection = self.get_or_create_collection()

        # 1. Fetch all authoritative chunks from SQLite
        with db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, course_id, content, content_hash, location_type, location_index
                FROM chunks;
                """
            ).fetchall()

        sqlite_chunks = {r["id"]: dict(r) for r in rows}
        total_chunks = len(sqlite_chunks)

        # 2. Fetch existing Chroma IDs and content hashes
        existing = collection.get(include=["metadatas"])
        existing_ids = set(existing.get("ids", []))
        existing_metas = {
            cid: meta for cid, meta in zip(existing.get("ids", []), existing.get("metadatas", []))
        }

        # 3. Detect stale records to delete (in Chroma but no longer in SQLite)
        stale_ids = existing_ids - set(sqlite_chunks.keys())
        deleted_count = 0
        if stale_ids:
            # Delete in chunks to respect batch limits
            stale_list = list(stale_ids)
            for i in range(0, len(stale_list), batch_size):
                collection.delete(ids=stale_list[i : i + batch_size])
            deleted_count = len(stale_list)

        # 4. Classify chunks into new, modified, or unchanged
        chunks_to_upsert: list[dict[str, Any]] = []
        unchanged_count = 0

        for chunk_id, chunk_data in sqlite_chunks.items():
            if chunk_id not in existing_ids:
                chunks_to_upsert.append(chunk_data)
            else:
                existing_meta = existing_metas.get(chunk_id) or {}
                if existing_meta.get("content_hash") != chunk_data["content_hash"]:
                    chunks_to_upsert.append(chunk_data)
                else:
                    unchanged_count += 1

        added_count = 0
        updated_count = 0

        # 5. Embed and upsert required chunks in batches
        if chunks_to_upsert:
            for i in range(0, len(chunks_to_upsert), batch_size):
                batch = chunks_to_upsert[i : i + batch_size]
                texts = [c["content"] for c in batch]
                embeddings = self.provider.embed_documents(texts)

                ids = [c["id"] for c in batch]
                metadatas = [
                    {
                        "chunk_id": c["id"],
                        "document_id": c["document_id"],
                        "course_id": c["course_id"],
                        "content_hash": c["content_hash"],
                        "embedding_model": self.provider.model_name,
                        "location_type": c["location_type"],
                        "location_index": int(c["location_index"]),
                    }
                    for c in batch
                ]

                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts,
                )

                for c in batch:
                    if c["id"] in existing_ids:
                        updated_count += 1
                    else:
                        added_count += 1

        duration = time.perf_counter() - start_time
        return SyncStats(
            added=added_count,
            updated=updated_count,
            deleted=deleted_count,
            unchanged=unchanged_count,
            total_chunks=total_chunks,
            duration_seconds=round(duration, 3),
        )

    def rebuild(self, db: Database) -> SyncStats:
        """Drop existing vector collection and rebuild all embeddings from SQLite source data."""
        if self.provider is None:
            raise VectorStoreError("Cannot rebuild vector index without an EmbeddingProvider.")

        # Delete existing collection if present
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._collection = None
        # Create fresh collection and sync
        return self.sync(db)

    def search_candidates(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        course_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Query nearest vector candidates with optional course and document filters.

        Returns list of (chunk_id, distance) tuples.
        """
        # If an explicit empty filter was passed, nothing matches
        if (course_ids is not None and len(course_ids) == 0) or (
            document_ids is not None and len(document_ids) == 0
        ):
            return []

        col = self.get_or_create_collection()
        total = col.count()
        if total == 0:
            return []

        # Construct Chroma where filter
        filters: list[dict[str, Any]] = []
        if course_ids:
            if len(course_ids) == 1:
                filters.append({"course_id": course_ids[0]})
            else:
                filters.append({"course_id": {"$in": course_ids}})

        if document_ids:
            if len(document_ids) == 1:
                filters.append({"document_id": document_ids[0]})
            else:
                filters.append({"document_id": {"$in": document_ids}})

        where_clause: dict[str, Any] | None = None
        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}

        n_results = min(top_k, total)
        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["distances"],
            )
        except Exception as e:
            # If Chroma fails due to no matching where clause, return empty list
            if "No documents found" in str(e) or "empty" in str(e).lower():
                return []
            raise VectorStoreError(f"ChromaDB search failed: {e}") from e

        ids_list = results.get("ids", [[]])[0]
        distances_list = results.get("distances", [[]])[0] if results.get("distances") else []

        candidates: list[tuple[str, float]] = []
        for idx, chunk_id in enumerate(ids_list):
            dist = float(distances_list[idx]) if idx < len(distances_list) else 1.0
            candidates.append((chunk_id, dist))

        return candidates
