"""Wikipedia REST Search provider for free, keyless external research."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus, urljoin
import httpx

from studyos.research.models import ResearchResult
from studyos.research.safety import SafeHTTPTransport, SafeURLValidator, SSRFValidationError

logger = logging.getLogger(__name__)

ALLOWED_WIKIPEDIA_HOSTS = {
    "en.wikipedia.org",
    "wikipedia.org",
    "www.wikipedia.org",
    "api.wikimedia.org",
}


class WikipediaSearchProvider:
    """Free, keyless search provider querying the official Wikimedia REST API."""

    BASE_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    def __init__(
        self,
        timeout: float = 8.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.timeout = timeout
        self.transport = transport

    @property
    def provider_name(self) -> str:
        return "wikipedia"

    def search(self, query: str, max_results: int = 5) -> list[ResearchResult]:
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        # Validate Wikipedia host against allowed hosts and SSRF rules before calling
        host, scheme, port = SafeURLValidator.validate_url_scheme_and_host(self.BASE_URL)
        if host not in ALLOWED_WIKIPEDIA_HOSTS:
            raise SSRFValidationError(
                f"Forbidden search provider endpoint host '{host}'. Must be an official Wikipedia host."
            )
        SafeURLValidator.verify_destination_ip(host, port or 443)

        url = f"{self.BASE_URL}?q={quote_plus(clean_query)}&limit={max_results}"

        transport = self.transport if self.transport is not None else SafeHTTPTransport()
        with httpx.Client(
            transport=transport,
            follow_redirects=False,
            timeout=self.timeout,
            trust_env=False,
            headers={"User-Agent": "StudyOS-Research/1.0 (academic study assistant)"},
        ) as client:
            try:
                response = client.get(url)
            except Exception as e:
                logger.warning(f"Wikipedia search request failed: {e}")
                return []

            # Handle safe redirects within allowed Wikipedia hosts
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    return []
                redirect_url = urljoin(url, location)
                r_host, r_scheme, r_port = SafeURLValidator.validate_url_scheme_and_host(redirect_url)
                if r_host not in ALLOWED_WIKIPEDIA_HOSTS:
                    raise SSRFValidationError(
                        f"Redirect escaped allowed Wikipedia hosts to untrusted host '{r_host}'."
                    )
                SafeURLValidator.verify_destination_ip(r_host, r_port or 443)
                try:
                    response = client.get(redirect_url)
                except Exception as e:
                    logger.warning(f"Wikipedia search redirect failed: {e}")
                    return []

            if response.status_code != 200:
                logger.warning(f"Wikipedia search returned status {response.status_code}")
                return []

            try:
                data = response.json()
            except Exception as e:
                logger.warning(f"Failed to parse Wikipedia JSON response: {e}")
                return []

            pages = data.get("pages", [])
            results: list[ResearchResult] = []

            for p in pages:
                key = p.get("key")
                title = p.get("title")
                excerpt = p.get("excerpt") or p.get("description") or ""
                # Strip HTML tags from excerpt
                clean_snippet = excerpt.replace("<span class=\"searchmatch\">", "").replace("</span>", "").strip()

                if key and title:
                    page_url = f"https://en.wikipedia.org/wiki/{key}"
                    results.append(
                        ResearchResult(
                            url=page_url,
                            title=title,
                            snippet=clean_snippet or f"Wikipedia article on {title}",
                            publisher="Wikipedia",
                            published_date=None,
                        )
                    )

            return results[:max_results]
