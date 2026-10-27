"""Evidence-grounded tutor orchestration service for StudyOS."""

from __future__ import annotations

import logging
from typing import Mapping

from studyos.core.config import AppConfig, get_default_config
from studyos.inference.models import ChatMessage, ChatRequest, ChatRole
from studyos.inference.protocol import ChatProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.models import RetrievalMode, RetrievedChunk
from studyos.tutoring.citations import CitationValidationResult, CitationValidator
from studyos.tutoring.models import (
    EvidencePassage,
    ResponseStyle,
    TutorRequest,
    TutorResponse,
    ValidatedCitation,
)
from studyos.tutoring.prompt import (
    TUTOR_SYSTEM_PROMPT,
    build_tutor_user_prompt,
)

logger = logging.getLogger(__name__)

STANDARD_REFUSAL_PREFIX = "The uploaded materials do not provide enough evidence to answer this question reliably."


class TutorService:
    """Orchestrates hybrid retrieval, evidence selection, local LLM generation, and citation validation."""

    def __init__(
        self,
        retrieval_service: HybridRetrievalService,
        chat_provider: ChatProvider,
        config: AppConfig | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.chat_provider = chat_provider
        self.config = config or get_default_config()

    def answer(self, request: TutorRequest) -> TutorResponse:
        """Answer a study question with verified citations and provenance tracking.

        Workflow:
        1. Query hybrid retrieval service with course and document filters.
        2. Evaluate evidence sufficiency: abstain immediately if no candidates found.
        3. Assign request-scoped evidence markers [E1]..[En] to top candidates.
        4. Assemble system instructions and untrusted evidence context.
        5. Invoke local chat provider.
        6. Validate extracted citation markers against active candidate set.
        7. If answer is factual but uncited, perform a bounded single repair attempt.
        8. Return final TutorResponse with trusted metadata resolution and warnings.
        """
        # 1. Parameter Normalization
        clean_query = request.query.strip()
        top_k = min(request.top_k, self.config.tutor_evidence_k)

        # 2. Retrieval Execution
        retrieval_resp = self.retrieval_service.search(
            query=clean_query,
            top_k=top_k,
            course_id=request.course_id,
            course_ids=request.course_ids,
            document_ids=request.document_ids,
        )

        effective_mode = retrieval_resp.mode
        all_warnings: list[str] = list(retrieval_resp.warnings)

        if effective_mode == RetrievalMode.DEGRADED_LEXICAL:
            all_warnings.append(
                "Retrieval executed in DEGRADED_LEXICAL mode because dense semantic index was unavailable."
            )

        # 3. Handle Empty or Insufficient Candidates (Grounding Guardrail)
        if retrieval_resp.count == 0:
            return TutorResponse(
                query=clean_query,
                answer=(
                    f"{STANDARD_REFUSAL_PREFIX} "
                    "No relevant passages were found in the selected course documents."
                ),
                citations=[],
                unsupported=True,
                retrieval_mode=effective_mode,
                warnings=all_warnings,
                evidence_count=0,
            )

        # 4. Evidence Selection and Marker Assignment
        passages: list[EvidencePassage] = []
        evidence_map: dict[str, RetrievedChunk] = {}

        for idx, chunk in enumerate(retrieval_resp.results, start=1):
            marker = f"E{idx}"
            passages.append(EvidencePassage(marker=marker, chunk=chunk))
            evidence_map[marker] = chunk

        # 5. Build Messages
        user_prompt = build_tutor_user_prompt(clean_query, passages, style=request.style)
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=TUTOR_SYSTEM_PROMPT),
            ChatMessage(role=ChatRole.USER, content=user_prompt),
        ]

        chat_req = ChatRequest(
            model=self.chat_provider.model_name,
            messages=messages,
            temperature=self.config.tutor_temperature,
            max_tokens=self.config.tutor_max_tokens,
        )

        # 6. Execute Chat Generation
        chat_resp = self.chat_provider.chat(chat_req)
        answer_text = chat_resp.content.strip()

        # 7. Validate Citations
        val_result = CitationValidator.validate_citations(answer_text, evidence_map)
        all_warnings.extend(val_result.warnings)

        # 8. Check for Explicit Abstention / Insufficient Evidence in Model Output
        lower_answer = answer_text.lower()
        model_abstained = (
            "do not provide enough evidence" in lower_answer
            or "does not provide enough evidence" in lower_answer
            or "cannot be answered" in lower_answer
            or "insufficient evidence" in lower_answer
        )

        # 9. Bounded Citation Repair (Attempt exactly 1 repair if answer is factual but has 0 valid citations)
        if not model_abstained and len(val_result.valid_citations) == 0:
            logger.info("Answer lacked valid evidence citations; attempting bounded repair.")
            repair_instructions = (
                f"Your previous answer did not cite any evidence markers. "
                f"Please rewrite your answer and support your claims using ONLY the assigned markers "
                f"([E1] through [E{len(passages)}]). "
                f"If the provided evidence cannot answer the question, state explicitly: "
                f"'{STANDARD_REFUSAL_PREFIX}'"
            )
            repair_messages = list(messages) + [
                ChatMessage(role=ChatRole.ASSISTANT, content=answer_text),
                ChatMessage(role=ChatRole.USER, content=repair_instructions),
            ]
            repair_req = ChatRequest(
                model=self.chat_provider.model_name,
                messages=repair_messages,
                temperature=0.1,
                max_tokens=self.config.tutor_max_tokens,
            )
            try:
                repair_resp = self.chat_provider.chat(repair_req)
                repaired_text = repair_resp.content.strip()
                repaired_val = CitationValidator.validate_citations(repaired_text, evidence_map)

                if len(repaired_val.valid_citations) > 0:
                    answer_text = repaired_text
                    val_result = repaired_val
                    all_warnings.extend(repaired_val.warnings)
                else:
                    all_warnings.append(
                        "Answer was produced without valid evidence citations after repair attempt."
                    )
            except Exception as e:
                logger.warning("Citation repair request failed: %s", e)
                all_warnings.append("Citation repair attempt encountered an error.")

        is_unsupported = model_abstained or (len(val_result.valid_citations) == 0 and len(passages) > 0)

        return TutorResponse(
            query=clean_query,
            answer=answer_text,
            citations=val_result.valid_citations,
            unsupported=is_unsupported,
            retrieval_mode=effective_mode,
            warnings=sorted(set(all_warnings)),
            raw_markers_found=val_result.raw_markers,
            rejected_markers=val_result.rejected_markers,
            evidence_count=len(passages),
        )
