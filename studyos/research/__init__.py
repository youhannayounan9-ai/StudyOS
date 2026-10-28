"""External Research package for StudyOS."""

from studyos.research.models import (
    ExternalSource,
    ResearchRecord,
    ResearchResult,
    ResearchStatus,
    ResearchSummary,
    SourceType,
)
from studyos.research.protocol import ExternalSearchProvider
from studyos.research.providers.fake import FakeSearchProvider
from studyos.research.providers.wikipedia import WikipediaSearchProvider
from studyos.research.repository import ResearchRepository
from studyos.research.safety import SafeExternalHttpClient, SafeURLValidator, SSRFValidationError
from studyos.research.service import ExternalResearchService

__all__ = [
    "ExternalSource",
    "ResearchRecord",
    "ResearchResult",
    "ResearchStatus",
    "ResearchSummary",
    "SourceType",
    "ExternalSearchProvider",
    "FakeSearchProvider",
    "WikipediaSearchProvider",
    "ResearchRepository",
    "SafeExternalHttpClient",
    "SafeURLValidator",
    "SSRFValidationError",
    "ExternalResearchService",
]
