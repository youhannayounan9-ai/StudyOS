"""Protocol interface for external search providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from studyos.research.models import ResearchResult


@runtime_checkable
class ExternalSearchProvider(Protocol):
    """Abstract interface for querying external web or encyclopedic sources."""

    @property
    def provider_name(self) -> str:
        """Name of the search provider."""
        ...

    def search(self, query: str, max_results: int = 5) -> list[ResearchResult]:
        """Perform search query and return normalized results."""
        ...
