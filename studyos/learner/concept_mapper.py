"""Concept normalization, extraction, and mapping service for StudyOS.

Associates assessment questions, course materials, and source chunks with normalized,
course-scoped concepts using trusted metadata and bounded local inference.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from studyos.assessment.models import ExamQuestion, QuestionSourceRef
from studyos.inference.models import ChatMessage, ChatRequest, ChatRole
from studyos.inference.protocol import ChatProvider

logger = logging.getLogger(__name__)


def normalize_concept_name(name: str) -> str:
    """Normalize a concept name string for deterministic deduplication within a course.

    Transforms to lowercase, strips leading/trailing punctuation and whitespace,
    and collapses internal whitespace.
    """
    clean = re.sub(r"[\s_]+", " ", (name or "").strip().lower())
    clean = re.sub(r"^[^\w]+|[^\w]+$", "", clean)
    return clean or "unclassified"


class ConceptMapper:
    """Maps assessment questions and source chunks to normalized course concepts."""

    def __init__(self, chat_provider: ChatProvider | None = None) -> None:
        self.chat_provider = chat_provider

    @staticmethod
    def extract_concept_labels_from_question(
        question: ExamQuestion,
        exam_topic: str | None = None,
    ) -> list[str]:
        """Extract candidate concept labels from existing trusted metadata.

        Hierarchy of evidence:
        1. Question sources location_title (e.g. 'Dijkstra Relaxations', 'Virtual Memory')
        2. Explicit exam topic (e.g. 'Shortest Path Algorithms')
        3. Fallback: 'Unclassified'
        """
        candidates: list[str] = []

        # 1. Check source location titles
        if question.sources:
            for src in question.sources:
                if src.location_title:
                    title = src.location_title.strip()
                    # Exclude generic titles like 'Slide 1', 'Page 4'
                    if title and not re.match(r"^(slide|page|section)\s*\d+$", title, re.IGNORECASE):
                        if title not in candidates:
                            candidates.append(title)

        # 2. Check exam-level topic
        if exam_topic and exam_topic.strip():
            clean_topic = exam_topic.strip()
            if clean_topic not in candidates:
                candidates.append(clean_topic)

        # 3. Fallback if no specific concept could be determined
        if not candidates:
            candidates.append("Unclassified")

        return candidates

    def propose_concepts_with_llm(
        self,
        question_text: str,
        sources: Sequence[QuestionSourceRef],
        course_name: str,
    ) -> list[str]:
        """Optionally use local LLM to propose 1-2 concise concept labels grounded strictly in sources.

        Guaranteed offline-safe: falls back immediately to trusted metadata if chat provider
        is unavailable or returns non-compliant output.
        """
        if not self.chat_provider:
            return []

        excerpts = [s.excerpt for s in sources if s.excerpt]
        context = "\n---\n".join(excerpts[:2]) if excerpts else ""
        if not context:
            return []

        prompt = (
            f"You are cataloging concepts for the course '{course_name}'.\n"
            f"Given the question and source excerpts below, identify the 1 or 2 specific technical "
            f"concepts being tested.\n"
            f"Rules:\n"
            f"- Output ONLY a comma-separated list of short concept names (e.g. 'Binary Search Trees, Inorder Traversal').\n"
            f"- Do NOT invent concepts not supported by the excerpt.\n"
            f"- Do NOT include conversational text or bullet points.\n\n"
            f"Question: {question_text}\n"
            f"Excerpts:\n{context}\n\n"
            f"Concepts:"
        )

        try:
            req = ChatRequest(
                model=self.chat_provider.model_name,
                messages=[
                    ChatMessage(role=ChatRole.SYSTEM, content="You are a strict academic concept tagger."),
                    ChatMessage(role=ChatRole.USER, content=prompt),
                ],
                temperature=0.0,
                max_tokens=64,
            )
            resp = self.chat_provider.chat(req)
            raw = (resp.content or "").strip()
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            valid = [p for p in parts if len(p) <= 80 and not p.lower().startswith("here")]
            return valid[:2]
        except Exception as e:
            logger.warning("LLM concept extraction failed or was offline: %s", e)
            return []
