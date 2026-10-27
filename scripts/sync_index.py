"""CLI utility to inspect, synchronize, and rebuild the local ChromaDB vector index."""

from __future__ import annotations

import argparse
import sys

from studyos.core.config import get_default_config
from studyos.core.health import check_ollama_health
from studyos.retrieval.embedding import EmbeddingError, OllamaEmbeddingProvider
from studyos.retrieval.vector_store import ChromaVectorStore, IncompatibleCollectionError
from studyos.storage.db import Database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StudyOS Vector Index Synchronization and Management CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sync",
        action="store_true",
        help="Incrementally synchronize ChromaDB vector index with SQLite chunks",
    )
    group.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop vector collection and rebuild all embeddings from scratch (SQLite remains untouched)",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Inspect status and counts of SQLite chunks and ChromaDB vectors",
    )

    args = parser.parse_args()
    config = get_default_config()
    db = Database(config.db_path)
    db.migrate()

    # Pre-flight check on Ollama
    health = check_ollama_health(config)

    provider: OllamaEmbeddingProvider | None = None
    if health.is_ready:
        provider = OllamaEmbeddingProvider.from_config(config)

    vector_store = ChromaVectorStore.from_config(config, provider=provider)

    try:
        if args.status:
            print("=" * 65)
            print("StudyOS - Vector Index & Storage Status")
            print("=" * 65)
            print(f"Database Path:    {config.db_path}")
            print(f"Chroma Directory: {config.chroma_dir}")
            print(f"Collection Name:  {config.chroma_collection_name}")
            print(f"Ollama Status:    {health.status.value.upper()} ({health.base_url})")
            print(f"Embedding Model:  {config.embedding_model} (dim: {config.embedding_dimension})")

            # SQLite count
            with db.get_connection() as conn:
                chunk_cnt = conn.execute("SELECT count(*) as c FROM chunks;").fetchone()["c"]
                doc_cnt = conn.execute("SELECT count(*) as c FROM documents;").fetchone()["c"]
                course_cnt = conn.execute("SELECT count(*) as c FROM courses;").fetchone()["c"]

            chroma_cnt = vector_store.count()
            print("-" * 65)
            print(f"Courses in SQLite:   {course_cnt}")
            print(f"Documents in SQLite: {doc_cnt}")
            print(f"Chunks in SQLite:    {chunk_cnt}")
            print(f"Vectors in ChromaDB: {chroma_cnt}")

            if chunk_cnt != chroma_cnt:
                print("Status: Index out of sync. Run 'python scripts/sync_index.py --sync'")
            else:
                print("Status: Index fully synchronized.")
            print("=" * 65)

        elif args.sync or args.rebuild:
            if not health.is_ready:
                print("ERROR: Cannot perform embedding operations because Ollama is not ready.")
                print(f"Status:  {health.status.value.upper()}")
                print(f"Message: {health.message}")
                if health.actionable_remediation:
                    print("-" * 65)
                    print("Remediation:")
                    print(health.actionable_remediation)
                print("=" * 65)
                sys.exit(1)

            action = "rebuilding" if args.rebuild else "synchronizing"
            print(f"Starting vector index {action} using model '{config.embedding_model}'...")

            if args.rebuild:
                stats = vector_store.rebuild(db)
            else:
                stats = vector_store.sync(db)

            print("=" * 65)
            print(f"Vector Index {action.capitalize()} Complete")
            print("=" * 65)
            print(f"Total SQLite Chunks: {stats.total_chunks}")
            print(f"Vectors Added:       {stats.added}")
            print(f"Vectors Updated:     {stats.updated}")
            print(f"Vectors Deleted:     {stats.deleted}")
            print(f"Vectors Unchanged:   {stats.unchanged}")
            print(f"Elapsed Time:        {stats.duration_seconds}s")
            print("=" * 65)

    except IncompatibleCollectionError as e:
        print(f"ERROR: Incompatible vector collection detected: {e}")
        print("Run 'python scripts/sync_index.py --rebuild' to regenerate embeddings with the new model.")
        sys.exit(1)
    except EmbeddingError as e:
        print(f"ERROR: Embedding generation failed: {e}")
        sys.exit(1)
    finally:
        vector_store.close()
        db.close()


if __name__ == "__main__":
    main()
