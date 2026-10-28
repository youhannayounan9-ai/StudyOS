"""Direct concept discovery service from ingested document chunks.

Discovers course-scoped concepts from raw uploaded material independently of whether
assessments exist, preserving chunk provenance, metadata hierarchies, and structured validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence
import uuid

from studyos.core.config import AppConfig, get_default_config
from studyos.inference.models import ChatMessage, ChatRequest, ChatRole
from studyos.inference.protocol import ChatProvider
from studyos.learner.concept_mapper import normalize_concept_name
from studyos.learner.models import (
    Concept,
    ConceptProvenance,
    ConceptReviewStatus,
    DiscoveredConcept,
    DiscoveryResult,
)
from studyos.learner.repository import LearnerRepository
from studyos.storage.db import Database

logger = logging.getLogger(__name__)

# Exclusion patterns for generic non-concepts
GENERIC_CONCEPT_PATTERNS = [
    r"^(slide|page|section|chapter|part|lecture|unit)\s*\d+$",
    r"^(overview|introduction|summary|conclusion|agenda|table of contents|questions|q&a|outline|review)$",
    r"^\s*$",
]


class ConceptDiscoveryService:
    """Discovers course-scoped concepts directly from ingested SQLite document chunks."""

    def __init__(
        self,
        db: Database,
        chat_provider: ChatProvider | None = None,
        config: AppConfig | None = None,
        repository: LearnerRepository | None = None,
    ) -> None:
        self.db = db
        self.chat_provider = chat_provider
        self.config = config or get_default_config()
        self.repository = repository or LearnerRepository(db)

    def discover_concepts(
        self,
        course_id: str,
        document_id: str | None = None,
        batch_size: int = 5,
        max_chunks: int = 100,
        auto_accept: bool = False,
    ) -> DiscoveryResult:
        """Discover concepts from ingested chunks for a course or document."""
        # 1. Verify course exists
        course = self.db.get_course(course_id)
        if not course:
            raise ValueError(f"Course '{course_id}' does not exist.")

        # 2. Query chunks from SQLite
        with self.db.get_connection() as conn:
            if document_id:
                chunks_rows = conn.execute(
                    """
                    SELECT id as chunk_id, document_id, course_id, chunk_index,
                           location_type, location_index, location_title, content
                    FROM chunks
                    WHERE course_id = ? AND document_id = ?
                    ORDER BY chunk_index ASC
                    LIMIT ?;
                    """,
                    (course_id, document_id, max_chunks),
                ).fetchall()
            else:
                chunks_rows = conn.execute(
                    """
                    SELECT id as chunk_id, document_id, course_id, chunk_index,
                           location_type, location_index, location_title, content
                    FROM chunks
                    WHERE course_id = ?
                    ORDER BY document_id, chunk_index ASC
                    LIMIT ?;
                    """,
                    (course_id, max_chunks),
                ).fetchall()

        if not chunks_rows:
            run_id = self.repository.record_discovery_run(
                course_id=course_id,
                document_id=document_id,
                status="completed",
                concepts_discovered=0,
                chunks_processed=0,
                error_message="No chunks found for course/document.",
            )
            return DiscoveryResult(
                run_id=run_id,
                course_id=course_id,
                document_id=document_id,
                status="completed",
                concepts_discovered=[],
                total_discovered=0,
                new_concepts_created=0,
                existing_concepts_updated=0,
                chunks_scanned=0,
            )

        chunks_data = [dict(r) for r in chunks_rows]
        discovered_candidates: list[DiscoveredConcept] = []

        # 3. Model-assisted or metadata-based discovery in bounded batches
        b_size = max(1, min(10, batch_size))
        for i in range(0, len(chunks_data), b_size):
            batch = chunks_data[i : i + b_size]
            candidates = self._process_chunk_batch(course.name, batch)
            discovered_candidates.extend(candidates)

        # Fallback if no model candidates: extract from chunk metadata
        if not discovered_candidates:
            discovered_candidates = self._extract_candidates_from_metadata(chunks_data)

        # 4. Deduplicate candidates by normalized_name within this run
        unique_by_norm: dict[str, DiscoveredConcept] = {}
        for c in discovered_candidates:
            if c.normalized_name not in unique_by_norm:
                unique_by_norm[c.normalized_name] = c
            else:
                # Merge supporting chunks
                existing_c = unique_by_norm[c.normalized_name]
                for chk_id in c.supporting_chunk_ids:
                    if chk_id not in existing_c.supporting_chunk_ids:
                        existing_c.supporting_chunk_ids.append(chk_id)
                for loc in c.source_locations:
                    if loc not in existing_c.source_locations:
                        existing_c.source_locations.append(loc)

        # 5. Persist concepts and link chunk provenance
        new_count = 0
        updated_count = 0
        final_discovered: list[DiscoveredConcept] = []
        target_status = ConceptReviewStatus.ACCEPTED if auto_accept else ConceptReviewStatus.PROPOSED

        for norm, cand in unique_by_norm.items():
            cand.review_status = target_status
            existing_cpt = self._find_existing_concept(course_id, norm)
            if existing_cpt:
                # Keep existing description if present; link chunks
                concept_id = existing_cpt.id
                updated_count += 1
            else:
                cpt_obj = self.repository.get_or_create_concept(
                    course_id=course_id,
                    name=cand.name,
                    description=cand.description,
                    review_status=target_status,
                    provenance=cand.provenance,
                    learning_objectives=cand.learning_objectives,
                )
                concept_id = cpt_obj.id
                new_count += 1

            # Map chunks in concept_chunks
            for chk_id in cand.supporting_chunk_ids:
                try:
                    self.repository.map_concept_to_chunk(concept_id, chk_id)
                except Exception as e:
                    logger.debug(f"Failed to map concept {concept_id} to chunk {chk_id}: {e}")

            final_discovered.append(cand)

        run_id = self.repository.record_discovery_run(
            course_id=course_id,
            document_id=document_id,
            status="completed",
            concepts_discovered=len(final_discovered),
            chunks_processed=len(chunks_data),
        )

        return DiscoveryResult(
            run_id=run_id,
            course_id=course_id,
            document_id=document_id,
            status="completed",
            concepts_discovered=final_discovered,
            total_discovered=len(final_discovered),
            new_concepts_created=new_count,
            existing_concepts_updated=updated_count,
            chunks_scanned=len(chunks_data),
        )

    def _process_chunk_batch(
        self,
        course_name: str,
        batch: list[dict[str, Any]],
    ) -> list[DiscoveredConcept]:
        """Process a batch of chunks using local inference or fallback to metadata."""
        if not self.chat_provider:
            return self._extract_candidates_from_metadata(batch)

        valid_chunk_ids = {c["chunk_id"] for c in batch}
        chunk_summaries = []
        for c in batch:
            title = c.get("location_title") or f"{c['location_type']} {c['location_index']}"
            excerpt = (c.get("content") or "").strip()[:400]
            chunk_summaries.append(f"Chunk ID: {c['chunk_id']}\nLocation: {title}\nExcerpt: {excerpt}")

        context_str = "\n---\n".join(chunk_summaries)

        prompt = (
            f"You are an academic concept cataloger for the course '{course_name}'.\n"
            f"Given the source chunks below, identify key educational concepts.\n"
            f"Requirements:\n"
            f"1. Extract 1 to 3 distinct technical concepts covered in these excerpts.\n"
            f"2. For each concept, provide: name, 1-sentence definition, 1-2 learning objectives, and supporting Chunk IDs.\n"
            f"3. Do NOT extract generic terms like 'Overview', 'Summary', 'Page 1', 'Table of Contents'.\n"
            f"4. Output strictly valid JSON conforming to this schema:\n"
            f'{{\n  "concepts": [\n    {{\n      "name": "Concept Name",\n      "description": "Definition",\n'
            f'      "learning_objectives": ["Objective 1"],\n      "supporting_chunk_ids": ["Chunk ID"]\n    }}\n  ]\n}}\n\n'
            f"Source Chunks:\n{context_str}\n\nJSON Output:"
        )

        try:
            req = ChatRequest(
                model=self.chat_provider.model_name,
                messages=[
                    ChatMessage(role=ChatRole.SYSTEM, content="You are a strict academic curriculum extractor that outputs JSON."),
                    ChatMessage(role=ChatRole.USER, content=prompt),
                ],
                temperature=0.0,
                max_tokens=512,
            )
            resp = self.chat_provider.chat(req)
            content = (resp.content or "").strip()
            return self._parse_json_concepts(content, valid_chunk_ids, batch)
        except Exception as e:
            logger.warning(f"Local LLM concept extraction failed or unavailable: {e}. Falling back to metadata.")
            return self._extract_candidates_from_metadata(batch)

    def _parse_json_concepts(
        self,
        raw_output: str,
        valid_chunk_ids: set[str],
        batch: list[dict[str, Any]],
    ) -> list[DiscoveredConcept]:
        """Safely parse and validate structured JSON concept output from the model."""
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            return self._extract_candidates_from_metadata(batch)

        try:
            data = json.loads(match.group(0))
        except Exception:
            return self._extract_candidates_from_metadata(batch)

        concepts_raw = data.get("concepts", [])
        if not isinstance(concepts_raw, list):
            return self._extract_candidates_from_metadata(batch)

        chunk_map = {c["chunk_id"]: c for c in batch}
        results: list[DiscoveredConcept] = []

        for item in concepts_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name or len(name) > 80:
                continue
            if self._is_generic_title(name):
                continue

            desc = str(item.get("description", "")).strip() or None
            objs = [str(o).strip() for o in item.get("learning_objectives", []) if str(o).strip()]

            # Validate supporting chunk IDs
            raw_ids = item.get("supporting_chunk_ids", [])
            valid_ids = [cid for cid in raw_ids if cid in valid_chunk_ids]
            if not valid_ids:
                # Default to first chunk in batch if none cited
                valid_ids = [batch[0]["chunk_id"]]

            locs = []
            for cid in valid_ids:
                c = chunk_map.get(cid)
                if c:
                    locs.append(
                        {
                            "chunk_id": cid,
                            "location_title": c.get("location_title"),
                            "location_type": c.get("location_type"),
                            "location_index": c.get("location_index"),
                        }
                    )

            norm = normalize_concept_name(name)
            results.append(
                DiscoveredConcept(
                    name=name,
                    normalized_name=norm,
                    description=desc,
                    learning_objectives=objs[:3],
                    supporting_chunk_ids=valid_ids,
                    provenance=ConceptProvenance.DISCOVERED,
                    review_status=ConceptReviewStatus.ACCEPTED,
                    source_locations=locs,
                )
            )

        return results if results else self._extract_candidates_from_metadata(batch)

    def _extract_candidates_from_metadata(
        self,
        batch: list[dict[str, Any]],
    ) -> list[DiscoveredConcept]:
        """Deterministic fallback: extract concepts from trusted location titles and headings."""
        results: list[DiscoveredConcept] = []
        for c in batch:
            title = (c.get("location_title") or "").strip()
            if not title or self._is_generic_title(title):
                continue

            norm = normalize_concept_name(title)
            results.append(
                DiscoveredConcept(
                    name=title,
                    normalized_name=norm,
                    description=f"Core concept introduced in {c.get('location_type', 'section')} {c.get('location_index', 1)}.",
                    learning_objectives=[f"Understand fundamentals of {title}."],
                    supporting_chunk_ids=[c["chunk_id"]],
                    provenance=ConceptProvenance.DISCOVERED,
                    review_status=ConceptReviewStatus.ACCEPTED,
                    source_locations=[
                        {
                            "chunk_id": c["chunk_id"],
                            "location_title": title,
                            "location_type": c.get("location_type"),
                            "location_index": c.get("location_index"),
                        }
                    ],
                )
            )
        return results

    def _is_generic_title(self, title: str) -> bool:
        """Check if a title matches generic non-concept patterns."""
        clean = title.strip().lower()
        for pat in GENERIC_CONCEPT_PATTERNS:
            if re.match(pat, clean, re.IGNORECASE):
                return True
        return False

    def _find_existing_concept(self, course_id: str, normalized_name: str) -> Concept | None:
        """Find concept in course by normalized name."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM concepts WHERE course_id = ? AND normalized_name = ?;",
                (course_id, normalized_name),
            ).fetchone()
            return self.repository._row_to_concept(row) if row else None
