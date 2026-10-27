"""StudyOS evidence-grounded tutoring package."""

from studyos.tutoring.citations import (
    CitationValidationResult,
    CitationValidator,
    extract_excerpt,
)
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
    format_evidence_context,
    sanitize_untrusted_text,
)
from studyos.tutoring.service import TutorService

__all__ = [
    "CitationValidationResult",
    "CitationValidator",
    "EvidencePassage",
    "ResponseStyle",
    "TutorRequest",
    "TutorResponse",
    "TutorService",
    "ValidatedCitation",
    "build_tutor_user_prompt",
    "extract_excerpt",
    "format_evidence_context",
    "sanitize_untrusted_text",
    "TUTOR_SYSTEM_PROMPT",
]
