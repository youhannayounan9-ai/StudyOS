"""Providers package for external research."""

from studyos.research.providers.fake import FakeSearchProvider
from studyos.research.providers.wikipedia import WikipediaSearchProvider

__all__ = ["FakeSearchProvider", "WikipediaSearchProvider"]
