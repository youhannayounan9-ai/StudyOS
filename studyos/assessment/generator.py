"""Evidence-grounded exam generation orchestrator."""

from __future__ import annotations

import logging
import uuid
from typing import Sequence

from studyos.assessment.models import (
    Difficulty,
    Exam,
    ExamQuestion,
    ExamRequest,
)
from studyos.assessment.prompt import EXAM_SYSTEM_PROMPT, build_exam_user_prompt
from studyos.assessment.repository import AssessmentRepository
from studyos.assessment.validator import ExamValidationError, ExamValidator
from studyos.core.config import AppConfig, get_default_config
from studyos.inference.models import ChatMessage, ChatRequest, ChatRole
from studyos.inference.protocol import ChatProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievedChunk
from studyos.tutoring.models import EvidencePassage

logger = logging.getLogger(__name__)


class InsufficientEvidenceError(Exception):
    """Raised when retrieved course evidence is absent or inadequate to generate an exam."""


class ExamGenerator:
    """Orchestrates hybrid retrieval, evidence gating, LLM structured JSON generation, and persistence."""

    def __init__(
        self,
        retrieval_service: HybridRetrievalService,
        chat_provider: ChatProvider,
        repository: AssessmentRepository,
        config: AppConfig | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.chat_provider = chat_provider
        self.repository = repository
        self.config = config or get_default_config()

    def generate_exam(self, request: ExamRequest) -> Exam:
        """Generate a validated, source-grounded practice exam and save it to SQLite."""
        # 1. Parameter Validation
        clean_course = request.course_id.strip()
        count = request.question_count

        # 2. Evidence Retrieval
        search_query = request.topic or "concepts principles definitions architecture methods"
        # Retrieve diverse candidate pool to span multiple document sections
        top_k = min(max(count * 2, 6), 15)

        retrieval_resp = self.retrieval_service.search(
            query=search_query,
            top_k=top_k,
            course_id=clean_course,
            document_ids=request.document_ids,
        )

        if retrieval_resp.count == 0:
            raise InsufficientEvidenceError(
                f"Cannot generate exam: no relevant course materials found in course '{clean_course}' "
                f"matching topic '{request.topic or 'general'}'."
            )

        # 3. Assign Application-Owned Evidence Markers
        passages: list[EvidencePassage] = []
        evidence_map: dict[str, RetrievedChunk] = {}

        for idx, chunk in enumerate(retrieval_resp.results, start=1):
            marker = f"E{idx}"
            passages.append(EvidencePassage(marker=marker, chunk=chunk))
            evidence_map[marker] = chunk

        # 4. Prompt Assembly
        user_prompt = build_exam_user_prompt(
            passages=passages,
            question_count=count,
            question_types=request.question_types,
            difficulty=request.difficulty,
            topic=request.topic,
        )

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=EXAM_SYSTEM_PROMPT),
            ChatMessage(role=ChatRole.USER, content=user_prompt),
        ]

        chat_req = ChatRequest(
            model=self.chat_provider.model_name,
            messages=messages,
            temperature=0.2,
            format="json",
        )

        exam_id = f"ex_{uuid.uuid4().hex[:12]}"

        # 5. Execute Generation
        chat_resp = self.chat_provider.chat(chat_req)
        raw_output = chat_resp.content.strip()

        # 6. Validate Output with Bounded Single Retry on Failure
        try:
            questions = ExamValidator.validate_raw_payload(
                payload_text=raw_output,
                exam_id=exam_id,
                evidence_map=evidence_map,
                target_count=count,
            )
        except ExamValidationError as e:
            logger.warning("Initial exam generation failed validation: %s; attempting bounded retry.", e)
            correction_prompt = (
                f"Your previous response failed validation with error: {e}\n"
                f"Please regenerate the exam questions adhering strictly to the JSON schema, "
                f"ensuring valid question types ('mcq', 'true_false', 'short_answer'), exactly 1 correct answer for MCQ, "
                f"and citing only active markers: {', '.join(sorted(evidence_map.keys()))}."
            )
            retry_messages = list(messages) + [
                ChatMessage(role=ChatRole.ASSISTANT, content=raw_output),
                ChatMessage(role=ChatRole.USER, content=correction_prompt),
            ]
            retry_req = ChatRequest(
                model=self.chat_provider.model_name,
                messages=retry_messages,
                temperature=0.1,
                format="json",
            )
            retry_resp = self.chat_provider.chat(retry_req)
            questions = ExamValidator.validate_raw_payload(
                payload_text=retry_resp.content.strip(),
                exam_id=exam_id,
                evidence_map=evidence_map,
                target_count=count,
            )

        # Slice to requested count if model generated extra
        final_questions = questions[:count]
        for idx, q in enumerate(final_questions, start=1):
            q.question_index = idx

        title = request.title or (
            f"{request.topic or 'Practice'} Exam ({request.difficulty.value.capitalize()})"
        )

        exam = Exam(
            id=exam_id,
            course_id=clean_course,
            title=title,
            topic=request.topic,
            difficulty=request.difficulty,
            question_count=len(final_questions),
            questions=final_questions,
        )

        # 7. Atomic Persistence
        self.repository.save_exam(exam)
        return exam
