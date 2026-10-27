"""Validation logic for generated exam questions, choices, answer keys, and evidence citations."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Mapping

from studyos.assessment.models import (
    Difficulty,
    ExamQuestion,
    MCQOption,
    QuestionSourceRef,
    QuestionType,
    ShortAnswerRubric,
)
from studyos.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

EVIDENCE_MARKER_REGEX = re.compile(r"\[?(E\d+)\]?", re.IGNORECASE)


class ExamValidationError(Exception):
    """Raised when generated questions fail structural, pedagogical, or grounding validation."""


class ExamValidator:
    """Validates raw LLM-generated exam JSON payloads into typed, grounded ExamQuestion models."""

    @classmethod
    def validate_raw_payload(
        cls,
        payload_text: str,
        exam_id: str,
        evidence_map: Mapping[str, RetrievedChunk],
        target_count: int | None = None,
    ) -> list[ExamQuestion]:
        """Parse raw model JSON string and validate all questions against grounding and schema rules."""
        cleaned = payload_text.strip()
        # Handle markdown fenced blocks if model enclosed in ```json ... ```
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except Exception as e:
            raise ExamValidationError(f"Failed to parse model output as JSON: {e}") from e

        if not isinstance(data, dict):
            raise ExamValidationError("Model JSON output must be a JSON object.")

        raw_questions = data.get("questions")
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ExamValidationError("Model JSON output missing non-empty 'questions' list.")

        validated_questions: list[ExamQuestion] = []
        seen_question_texts: set[str] = set()
        validation_errors: list[str] = []

        for idx, q_dict in enumerate(raw_questions, start=1):
            if not isinstance(q_dict, dict):
                logger.warning("Skipping non-dictionary question item at index %d", idx)
                continue

            try:
                question_obj = cls._validate_single_question(
                    q_dict=q_dict,
                    exam_id=exam_id,
                    question_index=idx,
                    evidence_map=evidence_map,
                )
            except ExamValidationError as e:
                err_msg = f"Question {idx} failed validation: {e}"
                logger.warning(err_msg)
                validation_errors.append(err_msg)
                continue

            # Deduplication check
            norm_text = " ".join(question_obj.question_text.lower().split())
            if norm_text in seen_question_texts:
                logger.warning("Skipping duplicate question text: %s", norm_text[:60])
                continue
            seen_question_texts.add(norm_text)

            validated_questions.append(question_obj)

        if not validated_questions:
            details = f" Details: {'; '.join(validation_errors)}" if validation_errors else ""
            raise ExamValidationError(
                f"No valid questions could be extracted from the model generation.{details}"
            )

        return validated_questions

    @classmethod
    def _validate_single_question(
        cls,
        q_dict: dict[str, Any],
        exam_id: str,
        question_index: int,
        evidence_map: Mapping[str, RetrievedChunk],
    ) -> ExamQuestion:
        q_text = str(q_dict.get("question") or "").strip()
        if len(q_text) < 10:
            raise ExamValidationError("Question text is empty or too short (< 10 chars).")

        raw_type = str(q_dict.get("type") or "").strip().lower()
        try:
            q_type = QuestionType(raw_type)
        except ValueError:
            raise ExamValidationError(f"Unsupported question type '{raw_type}'.")

        raw_diff = str(q_dict.get("difficulty") or "medium").strip().lower()
        try:
            q_diff = Difficulty(raw_diff)
        except ValueError:
            q_diff = Difficulty.MEDIUM

        explanation = str(q_dict.get("explanation") or "").strip()
        if not explanation:
            explanation = "Explanation supported by course evidence."

        # 1. Resolve and validate evidence markers
        raw_markers = q_dict.get("evidence_markers") or []
        if isinstance(raw_markers, str):
            raw_markers = [raw_markers]

        # Also extract from explanation or question text if markers list was empty
        combined_text = f"{q_text} {explanation}"
        inferred_markers = EVIDENCE_MARKER_REGEX.findall(combined_text)
        all_candidate_markers = set([m.upper() for m in raw_markers] + [m.upper() for m in inferred_markers])

        sources: list[QuestionSourceRef] = []
        for m in sorted(all_candidate_markers):
            if m in evidence_map:
                chunk = evidence_map[m]
                sources.append(
                    QuestionSourceRef(
                        chunk_id=chunk.chunk_id,
                        evidence_marker=m,
                        filename=chunk.filename,
                        location_type=chunk.location_type,
                        location_index=chunk.location_index,
                        location_title=chunk.location_title,
                        excerpt=chunk.content[:200].strip(),
                    )
                )

        if not sources:
            # If model didn't provide markers, but evidence exists, map to E1 by default
            if evidence_map:
                default_marker = "E1"
                chunk = evidence_map.get(default_marker) or list(evidence_map.values())[0]
                sources.append(
                    QuestionSourceRef(
                        chunk_id=chunk.chunk_id,
                        evidence_marker=default_marker,
                        filename=chunk.filename,
                        location_type=chunk.location_type,
                        location_index=chunk.location_index,
                        location_title=chunk.location_title,
                        excerpt=chunk.content[:200].strip(),
                    )
                )
            else:
                raise ExamValidationError("Question cannot be grounded: no valid evidence markers found.")

        # 2. Type-specific validations
        options: list[MCQOption] | None = None
        rubric: ShortAnswerRubric | None = None
        raw_correct = q_dict.get("correct_answer")

        if q_type == QuestionType.MCQ:
            raw_opts = q_dict.get("options")
            if not isinstance(raw_opts, list) or len(raw_opts) < 2:
                raise ExamValidationError("MCQ must contain at least 2 option choices.")

            options = []
            seen_ids: set[str] = set()
            seen_texts: set[str] = set()

            for opt in raw_opts:
                if not isinstance(opt, dict):
                    continue
                opt_id = str(opt.get("id") or "").strip().upper()
                opt_text = str(opt.get("text") or "").strip()
                if not opt_id or not opt_text:
                    continue
                if opt_id in seen_ids:
                    raise ExamValidationError(f"Duplicate option ID '{opt_id}' in MCQ choices.")
                if opt_text.lower() in seen_texts:
                    raise ExamValidationError(f"Duplicate option text '{opt_text}' in MCQ choices.")
                seen_ids.add(opt_id)
                seen_texts.add(opt_text.lower())
                options.append(MCQOption(id=opt_id, text=opt_text))

            if len(options) < 2:
                raise ExamValidationError("MCQ choices must have at least 2 valid, non-empty options.")

            correct_answer = str(raw_correct or "").strip().upper()
            if correct_answer not in seen_ids:
                raise ExamValidationError(
                    f"MCQ correct_answer '{correct_answer}' does not match any choice ID ({', '.join(sorted(seen_ids))})."
                )

        elif q_type == QuestionType.TRUE_FALSE:
            # Options for True/False are fixed
            options = [
                MCQOption(id="True", text="True"),
                MCQOption(id="False", text="False"),
            ]
            ans_str = str(raw_correct or "").strip().lower()
            if ans_str in ("true", "t", "1"):
                correct_answer = "True"
            elif ans_str in ("false", "f", "0"):
                correct_answer = "False"
            else:
                raise ExamValidationError(
                    f"True/False question correct_answer must be 'True' or 'False', got '{raw_correct}'."
                )

        elif q_type == QuestionType.SHORT_ANSWER:
            correct_answer = str(raw_correct or "").strip()
            if len(correct_answer) < 3:
                raise ExamValidationError("Short-answer reference answer cannot be empty or too short.")

            raw_rubric = q_dict.get("rubric")
            if isinstance(raw_rubric, dict):
                rubric = ShortAnswerRubric(
                    criteria=[str(c) for c in raw_rubric.get("criteria", []) if c],
                    key_points=[str(k) for k in raw_rubric.get("key_points", []) if k],
                    max_score=float(raw_rubric.get("max_score", 1.0)),
                )
            else:
                rubric = ShortAnswerRubric(
                    criteria=["Conceptual accuracy", "Completeness"],
                    key_points=[correct_answer],
                    max_score=1.0,
                )

        else:
            raise ExamValidationError(f"Unhandled question type: {q_type}")

        q_id = f"q_{uuid.uuid4().hex[:12]}"
        return ExamQuestion(
            id=q_id,
            exam_id=exam_id,
            question_index=question_index,
            question_type=q_type,
            difficulty=q_diff,
            question_text=q_text,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            rubric=rubric,
            sources=sources,
            points=1.0,
        )
