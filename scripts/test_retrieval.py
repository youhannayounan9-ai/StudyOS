"""CLI query utility for testing StudyOS lexical, semantic, and hybrid retrieval."""

from __future__ import annotations

import argparse
import sys

from studyos.core.config import get_default_config
from studyos.core.health import check_ollama_health
from studyos.retrieval.embedding import FakeEmbeddingProvider, OllamaEmbeddingProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievalMode
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="StudyOS Hybrid Retrieval Query CLI")
    parser.add_argument("--query", type=str, required=True, help="Search query string")
    parser.add_argument("--course", type=str, default=None, help="Course ID or course code filter")
    parser.add_argument("--document", type=str, default=None, help="Document ID filter")
    parser.add_argument("--top-k", type=int, default=5, help="Maximum number of results (default: 5)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "lexical", "lexical_only", "semantic", "semantic_only"],
        default="hybrid",
        help="Retrieval modality to execute (default: hybrid)",
    )
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use deterministic local fake embeddings for testing without Ollama",
    )

    args = parser.parse_args()
    mode = args.mode
    if mode == "lexical":
        mode = "lexical_only"
    elif mode == "semantic":
        mode = "semantic_only"

    config = get_default_config()
    db = Database(config.db_path)
    db.migrate()

    # Resolve course filter if passed as course code
    course_id = args.course
    if course_id:
        course_obj = db.get_course_by_code(course_id)
        if course_obj:
            course_id = course_obj.id

    # Configure embedding provider
    provider = None
    if args.fake_embeddings:
        provider = FakeEmbeddingProvider(dimension=config.embedding_dimension)
    else:
        health = check_ollama_health(config)
        if health.is_ready:
            provider = OllamaEmbeddingProvider.from_config(config)
        elif mode in ("hybrid", "semantic_only"):
            print(f"Notice: Ollama is offline ({health.message}).")
            if mode == "semantic_only":
                print("Cannot perform semantic-only retrieval without Ollama (use --fake-embeddings to test).")
                sys.exit(1)
            print("Hybrid search will execute in DEGRADED_LEXICAL mode.\n")

    vector_store = ChromaVectorStore.from_config(config, provider=provider)
    service = HybridRetrievalService(db=db, vector_store=vector_store, rrf_k=config.rrf_k)

    try:
        response = service.search(
            query=args.query,
            top_k=args.top_k,
            course_id=course_id,
            document_ids=[args.document] if args.document else None,
            mode=RetrievalMode(mode),
        )

        print("=" * 70)
        print("StudyOS - Retrieval Query Results")
        print("=" * 70)
        print(f"Query:            '{response.query}'")
        print(f"Effective Mode:   {response.mode.value.upper()}")
        print(f"Total Candidates: {response.total_candidates}")
        print(f"Returned Count:   {len(response.results)}")
        if response.warnings:
            print(f"Warnings:         {'; '.join(response.warnings)}")
        print("=" * 70)

        if not response.results:
            print("No matching chunks found.")
            return

        for idx, chunk in enumerate(response.results, start=1):
            ranks = []
            if chunk.lexical_rank is not None:
                ranks.append(f"Lexical #{chunk.lexical_rank}")
            if chunk.semantic_rank is not None:
                ranks.append(f"Semantic #{chunk.semantic_rank}")
            rank_str = f" ({', '.join(ranks)})" if ranks else ""

            print(f"\n[{idx}] Score: {chunk.score:.5f}{rank_str} | Sources: {', '.join(chunk.sources)}")
            print(f"    Breadcrumb: {chunk.breadcrumb}")
            print(f"    Chunk ID:   {chunk.chunk_id} | Document: {chunk.document_id} | Course: {chunk.course_name}")
            print(f"    Content:")
            # Indent snippet
            snippet = chunk.content[:240].replace("\n", " ")
            print(f"      \"{snippet}...\"")

        print("\n" + "=" * 70)

    finally:
        vector_store.close()
        db.close()


if __name__ == "__main__":
    main()
