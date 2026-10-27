"""CLI interface for asking StudyOS evidence-grounded study questions."""

from __future__ import annotations

import argparse
import sys

from studyos.core.config import get_default_config
from studyos.core.health import HealthStatus, check_ollama_health
from studyos.inference.fake import FakeChatProvider
from studyos.inference.models import ChatRole
from studyos.inference.ollama import OllamaChatProvider
from studyos.inference.protocol import (
    ChatModelNotFoundError,
    ChatTimeoutError,
    InferenceError,
    OllamaChatConnectionError,
)
from studyos.retrieval.embedding import FakeEmbeddingProvider, OllamaEmbeddingProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievalMode
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database
from studyos.tutoring.models import ResponseStyle, TutorRequest
from studyos.tutoring.service import TutorService


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="StudyOS AI Tutor — Ask questions grounded in your course materials"
    )
    parser.add_argument("--query", "-q", type=str, required=True, help="Question or study inquiry")
    parser.add_argument("--course", "-c", type=str, default=None, help="Filter by course ID or course code")
    parser.add_argument("--document", "-d", type=str, default=None, help="Filter by specific document ID")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of evidence chunks to retrieve (default: 5)")
    parser.add_argument(
        "--style",
        type=str,
        choices=["balanced", "concise", "detailed"],
        default="balanced",
        help="Explanation style (default: balanced)",
    )
    parser.add_argument(
        "--fake-chat",
        action="store_true",
        help="Use deterministic offline fake chat provider (for testing without Ollama)",
    )
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use deterministic offline fake embeddings (for testing without Ollama)",
    )

    args = parser.parse_args()

    config = get_default_config()
    db = Database(config.db_path)
    db.migrate()

    # 1. Resolve Course Code if Provided
    course_id = args.course
    if course_id:
        course_obj = db.get_course_by_code(course_id)
        if course_obj:
            course_id = course_obj.id

    # 2. Configure Embedding Provider for Retrieval
    embedding_provider = None
    if args.fake_embeddings:
        embedding_provider = FakeEmbeddingProvider(dimension=config.embedding_dimension)
    else:
        embed_health = check_ollama_health(config, model_name=config.embedding_model)
        if embed_health.is_ready:
            embedding_provider = OllamaEmbeddingProvider.from_config(config)
        else:
            # Let hybrid search gracefully degrade to lexical search
            embedding_provider = None

    vector_store = ChromaVectorStore.from_config(config, provider=embedding_provider)
    retrieval_service = HybridRetrievalService(
        db=db,
        vector_store=vector_store,
        rrf_k=config.rrf_k,
    )

    # 3. Configure Chat Provider for Answering
    if args.fake_chat:
        chat_provider = FakeChatProvider(
            model_name="fake-tutor",
            default_response="This concept is grounded in the provided course notes [E1].",
        )
    else:
        # Pre-flight check for chat model
        chat_health = check_ollama_health(config, model_name=config.tutor_model)
        if not chat_health.is_ready:
            print("\n" + "=" * 65, file=sys.stderr)
            print("ERROR: Local AI Chat Service Unavailable", file=sys.stderr)
            print("=" * 65, file=sys.stderr)
            print(f"Status:   {chat_health.status.value.upper()}", file=sys.stderr)
            print(f"Base URL: {chat_health.base_url}", file=sys.stderr)
            print(f"Model:    {chat_health.model_name}", file=sys.stderr)
            print(f"Message:  {chat_health.message}", file=sys.stderr)
            if chat_health.actionable_remediation:
                print("-" * 65, file=sys.stderr)
                print("Action Required:", file=sys.stderr)
                print(chat_health.actionable_remediation, file=sys.stderr)
            print("=" * 65, file=sys.stderr)
            print("Tip: Use '--fake-chat' to test the tutor without running Ollama.", file=sys.stderr)
            sys.exit(1)

        chat_provider = OllamaChatProvider.from_config(config)

    tutor_service = TutorService(
        retrieval_service=retrieval_service,
        chat_provider=chat_provider,
        config=config,
    )

    request = TutorRequest(
        query=args.query,
        course_id=course_id,
        document_ids=[args.document] if args.document else None,
        top_k=args.top_k,
        style=ResponseStyle(args.style),
    )

    try:
        response = tutor_service.answer(request)
    except OllamaChatConnectionError as e:
        print(f"\nError: Ollama daemon connection lost during generation: {e}", file=sys.stderr)
        sys.exit(1)
    except ChatModelNotFoundError as e:
        print(f"\nError: Chat model not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ChatTimeoutError as e:
        print(f"\nError: Generation timed out: {e}", file=sys.stderr)
        sys.exit(1)
    except InferenceError as e:
        print(f"\nError: Local inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Formatted Display Output
    print("\n" + "=" * 70)
    print("StudyOS — Evidence-Grounded Tutor")
    print("=" * 70)
    print(f"Question:       {response.query}")
    print(f"Retrieval Mode: {response.retrieval_mode.value.upper()}")
    print(f"Evidence Found: {response.evidence_count} passage(s)")

    if response.warnings:
        print("\nWarnings:")
        for w in response.warnings:
            print(f"  • {w}")

    print("\n" + "-" * 70)
    print("ANSWER:")
    print("-" * 70)
    print(response.answer)
    print("-" * 70)

    if response.unsupported:
        print("\n[Notice: The tutor indicated insufficient evidence to answer this question reliably.]")

    if response.citations:
        print("\nVALIDATED CITATIONS:")
        for c in response.citations:
            print(f"  • {c.citation_label}")
            print(f"    Excerpt: \"{c.excerpt}\"")
            print(f"    Chunk ID: {c.chunk_id}")
    else:
        if not response.unsupported:
            print("\n[No citations were resolved for this answer.]")

    if response.rejected_markers:
        print("\nREJECTED / UNVERIFIED CITATION MARKERS:")
        for r in response.rejected_markers:
            print(f"  • [{r}] (Rejected: marker was not in active retrieval evidence set)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
