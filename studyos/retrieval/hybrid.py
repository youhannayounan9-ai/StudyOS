"""Hybrid retrieval engine combining lexical and semantic search via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from typing import Any

from studyos.core.config import AppConfig
from studyos.retrieval.embedding import EmbeddingError, EmbeddingProvider
from studyos.retrieval.lexical import LexicalSearchService
from studyos.retrieval.models import RetrievalMode, RetrievalResponse, RetrievedChunk
from studyos.retrieval.semantic import SemanticSearchService
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """Unified hybrid retrieval service combining SQLite FTS5 BM25 and ChromaDB vector search.

    Implements Reciprocal Rank Fusion (RRF):
        RRF(c) = sum(1 / (k + rank_m(c)))
    with authoritative SQLite metadata enrichment, strict course/document isolation,
    deterministic tie-breaking, and explicit degraded-mode reporting.
    """

    def __init__(
        self,
        db: Database,
        vector_store: ChromaVectorStore | None = None,
        lexical_service: LexicalSearchService | None = None,
        semantic_service: SemanticSearchService | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.db = db
        self.rrf_k = rrf_k
        self.lexical_service = lexical_service or LexicalSearchService(db)

        if semantic_service is not None:
            self.semantic_service: SemanticSearchService | None = semantic_service
        elif vector_store is not None:
            self.semantic_service = SemanticSearchService(db, vector_store)
        else:
            self.semantic_service = None

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        db: Database,
        provider: EmbeddingProvider | None = None,
    ) -> HybridRetrievalService:
        """Construct HybridRetrievalService from application configuration."""
        vector_store = ChromaVectorStore.from_config(config, provider=provider)
        return cls(
            db=db,
            vector_store=vector_store,
            rrf_k=config.rrf_k,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        course_id: str | None = None,
        course_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
    ) -> RetrievalResponse:
        """Execute hybrid, lexical-only, or semantic-only retrieval with filtering and RRF ranking."""
        # 1. Parameter Validation
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return RetrievalResponse(
                query=query,
                results=[],
                mode=RetrievalMode(mode),
                warnings=[],
                total_candidates=0,
            )

        # Normalize requested mode
        req_mode = RetrievalMode(mode) if isinstance(mode, str) else mode

        # Normalize course filters
        effective_course_ids: list[str] | None = None
        if course_id or course_ids is not None:
            courses_set: set[str] = set()
            if course_id:
                courses_set.add(course_id)
            if course_ids is not None:
                courses_set.update(course_ids)
            effective_course_ids = sorted(courses_set)

            # If course_ids was passed as an explicit empty list and no course_id was given
            if course_ids is not None and len(course_ids) == 0 and not course_id:
                return RetrievalResponse(
                    query=query,
                    results=[],
                    mode=req_mode,
                    warnings=[],
                    total_candidates=0,
                    course_ids_filter=[],
                    document_ids_filter=document_ids,
                )

        if document_ids is not None and len(document_ids) == 0:
            return RetrievalResponse(
                query=query,
                results=[],
                mode=req_mode,
                warnings=[],
                total_candidates=0,
                course_ids_filter=effective_course_ids,
                document_ids_filter=[],
            )

        # 2. Candidate Retrieval
        candidate_k = max(top_k * 3, 50)
        warnings: list[str] = []

        lexical_candidates: list[RetrievedChunk] = []
        semantic_candidates: list[RetrievedChunk] = []
        actual_mode = req_mode

        # Fetch Lexical
        if req_mode in (RetrievalMode.HYBRID, RetrievalMode.LEXICAL_ONLY):
            try:
                lexical_candidates = self.lexical_service.search(
                    query=cleaned_query,
                    top_k=candidate_k,
                    course_ids=effective_course_ids,
                    document_ids=document_ids,
                )
            except Exception as e:
                logger.error("Lexical search failed: %s", e)
                warnings.append(f"Lexical search error: {e}")

        # Fetch Semantic
        if req_mode in (RetrievalMode.HYBRID, RetrievalMode.SEMANTIC_ONLY):
            if self.semantic_service is None:
                warn_msg = "Semantic search is unavailable (no vector store or embedding provider configured)."
                warnings.append(warn_msg)
                if req_mode == RetrievalMode.HYBRID:
                    actual_mode = RetrievalMode.DEGRADED_LEXICAL
            else:
                try:
                    semantic_candidates = self.semantic_service.search(
                        query=cleaned_query,
                        top_k=candidate_k,
                        course_ids=effective_course_ids,
                        document_ids=document_ids,
                    )
                except (EmbeddingError, Exception) as e:
                    warn_msg = f"Semantic search failed ({type(e).__name__}): {e}"
                    logger.warning(warn_msg)
                    warnings.append(warn_msg)
                    if req_mode == RetrievalMode.HYBRID:
                        actual_mode = RetrievalMode.DEGRADED_LEXICAL
                    elif req_mode == RetrievalMode.SEMANTIC_ONLY:
                        # If semantic-only was explicitly requested and failed, return empty with warning
                        return RetrievalResponse(
                            query=query,
                            results=[],
                            mode=RetrievalMode.SEMANTIC_ONLY,
                            warnings=warnings,
                            total_candidates=0,
                            course_ids_filter=effective_course_ids,
                            document_ids_filter=document_ids,
                        )

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score(c) = sum(1 / (k + rank))
        fused_scores: dict[str, float] = {}
        chunk_data: dict[str, RetrievedChunk] = {}
        lexical_ranks: dict[str, int] = {}
        semantic_ranks: dict[str, int] = {}
        sources_map: dict[str, set[str]] = {}

        # Process lexical candidates
        for rank, chunk in enumerate(lexical_candidates, start=1):
            cid = chunk.chunk_id
            score_contrib = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + score_contrib
            chunk_data[cid] = chunk
            lexical_ranks[cid] = rank
            sources_map.setdefault(cid, set()).add("lexical")

        # Process semantic candidates
        for rank, chunk in enumerate(semantic_candidates, start=1):
            cid = chunk.chunk_id
            score_contrib = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + score_contrib
            # Prefer existing chunk data or populate
            if cid not in chunk_data:
                chunk_data[cid] = chunk
            semantic_ranks[cid] = rank
            sources_map.setdefault(cid, set()).add("semantic")

        total_candidates = len(fused_scores)
        if total_candidates == 0:
            return RetrievalResponse(
                query=query,
                results=[],
                mode=actual_mode,
                warnings=warnings,
                total_candidates=0,
                course_ids_filter=effective_course_ids,
                document_ids_filter=document_ids,
            )

        # 4. Deterministic Sorting
        # Primary key: -score (descending)
        # Secondary key: chunk_id (alphabetical ascending for tie-breaking)
        sorted_chunk_ids = sorted(
            fused_scores.keys(),
            key=lambda cid: (-fused_scores[cid], cid),
        )

        # 5. Build final result list
        final_results: list[RetrievedChunk] = []
        for cid in sorted_chunk_ids[:top_k]:
            base_chunk = chunk_data[cid]
            final_results.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    document_id=base_chunk.document_id,
                    course_id=base_chunk.course_id,
                    course_name=base_chunk.course_name,
                    filename=base_chunk.filename,
                    file_type=base_chunk.file_type,
                    chunk_index=base_chunk.chunk_index,
                    location_type=base_chunk.location_type,
                    location_index=base_chunk.location_index,
                    location_title=base_chunk.location_title,
                    content=base_chunk.content,
                    token_count=base_chunk.token_count,
                    content_hash=base_chunk.content_hash,
                    score=round(fused_scores[cid], 6),
                    lexical_rank=lexical_ranks.get(cid),
                    semantic_rank=semantic_ranks.get(cid),
                    sources=sorted(sources_map.get(cid, set())),
                )
            )

        return RetrievalResponse(
            query=query,
            results=final_results,
            mode=actual_mode,
            warnings=warnings,
            total_candidates=total_candidates,
            course_ids_filter=effective_course_ids,
            document_ids_filter=document_ids,
        )
