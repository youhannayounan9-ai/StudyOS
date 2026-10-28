"""Deterministic fake search provider for offline testing and development."""

from __future__ import annotations

from studyos.research.models import ResearchResult


class FakeSearchProvider:
    """Offline test search provider with predefined or synthetic search results."""

    def __init__(self, predefined_results: list[ResearchResult] | None = None) -> None:
        self._results = predefined_results or []
        self.recorded_queries: list[str] = []

    @property
    def provider_name(self) -> str:
        return "fake_provider"

    def set_results(self, results: list[ResearchResult]) -> None:
        self._results = results

    def search(self, query: str, max_results: int = 5) -> list[ResearchResult]:
        self.recorded_queries.append(query)
        if self._results:
            return self._results[:max_results]

        # Generate deterministic synthetic results based on query keywords
        q_clean = query.strip()
        return [
            ResearchResult(
                url=f"https://en.wikipedia.org/wiki/{q_clean.replace(' ', '_')}",
                title=f"Encyclopedia Overview of {q_clean.title()}",
                snippet=f"Detailed reference article explaining key concepts and foundations of {q_clean}.",
                publisher="Wikipedia, The Free Encyclopedia",
                published_date="2024-01-01",
            ),
            ResearchResult(
                url=f"https://docs.python.org/3/library/{q_clean.lower().replace(' ', '')}.html",
                title=f"Technical Documentation: {q_clean.title()}",
                snippet=f"Standard reference documentation and API specification for {q_clean}.",
                publisher="Python Software Foundation",
                published_date="2024-02-15",
            ),
        ][:max_results]
