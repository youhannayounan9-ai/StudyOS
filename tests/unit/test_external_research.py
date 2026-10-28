"""Unit tests for External Research service, SSRF safety, citation grounding, and privacy boundary."""

import pytest

from studyos.research.models import (
    ResearchRecord,
    ResearchResult,
    ResearchStatus,
    SourceType,
)
from studyos.research.providers.fake import FakeSearchProvider
from studyos.research.repository import ResearchRepository
from studyos.research.safety import (
    SafeExternalHttpClient,
    SafeURLValidator,
    SSRFValidationError,
    extract_plain_text_from_html,
)
from studyos.research.service import ExternalResearchService
from studyos.storage.db import Database


@pytest.fixture
def research_setup(tmp_path):
    db_file = tmp_path / "test_research.db"
    db = Database(db_file)
    db.migrate()

    with db.connect() as conn:
        conn.execute("INSERT INTO courses (id, name) VALUES ('course_phy101', 'Physics 101')")

    repo = ResearchRepository(db)
    provider = FakeSearchProvider()
    service = ExternalResearchService(db, provider=provider, repository=repo)

    return db, repo, provider, service


def test_explicit_opt_in_required(research_setup):
    """External research must raise PermissionError if explicit_opt_in is False."""
    _, _, _, service = research_setup

    with pytest.raises(PermissionError) as exc_info:
        service.research("quantum entanglement", explicit_opt_in=False)
    assert "explicit opt-in is required" in str(exc_info.value)


def test_privacy_boundary_rejects_file_paths_and_huge_queries(research_setup):
    """Queries containing local file paths or excessively long text are rejected."""
    _, _, _, service = research_setup

    # Path injection attempt
    with pytest.raises(ValueError) as exc:
        service.research("Check notes at C:\\Users\\secret\\notes.txt", explicit_opt_in=True)
    assert "privacy bounds" in str(exc.value)

    # Oversized query
    with pytest.raises(ValueError) as exc:
        service.research("a" * 350, explicit_opt_in=True)
    assert "privacy bounds" in str(exc.value)


def test_ssrf_validator_blocks_forbidden_schemes():
    """Only http and https schemes are permitted."""
    with pytest.raises(SSRFValidationError):
        SafeURLValidator.validate_url_scheme_and_host("file:///etc/passwd")

    with pytest.raises(SSRFValidationError):
        SafeURLValidator.validate_url_scheme_and_host("ftp://example.com/file")

    with pytest.raises(SSRFValidationError):
        SafeURLValidator.validate_url_scheme_and_host("gopher://127.0.0.1:70")


def test_ssrf_validator_blocks_loopback_and_metadata_hosts():
    """Localhost and loopback hostnames are immediately blocked."""
    with pytest.raises(SSRFValidationError):
        SafeURLValidator.validate_url_scheme_and_host("http://localhost:8000/api")

    with pytest.raises(SSRFValidationError):
        SafeURLValidator.validate_url_scheme_and_host("http://127.0.0.1:8000/")

    with pytest.raises(SSRFValidationError):
        SafeURLValidator.validate_url_scheme_and_host("http://metadata.google.internal/computeMetadata/v1/")


def test_ssrf_validator_blocks_private_and_link_local_ips(monkeypatch):
    """DNS resolution to private RFC 1918 or link-local IPs raises SSRFValidationError."""
    import socket

    # Simulate DNS resolving 'internal.corp' to 192.168.1.50
    def fake_getaddrinfo_private(host, port, proto=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.50", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo_private)

    with pytest.raises(SSRFValidationError) as exc:
        SafeURLValidator.verify_destination_ip("internal.corp")
    assert "private network IP" in str(exc.value)

    # Simulate DNS resolving 'meta.cloud' to 169.254.169.254 (AWS/GCP metadata)
    def fake_getaddrinfo_metadata(host, port, proto=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo_metadata)

    with pytest.raises(SSRFValidationError) as exc:
        SafeURLValidator.verify_destination_ip("meta.cloud")
    assert "link-local/cloud-metadata IP" in str(exc.value)


def test_safe_html_text_extractor():
    """Text extractor strips script and style tags and leaves plain text."""
    html_sample = """
    <html>
        <head><title>Test Page</title><style>body { color: red; }</style></head>
        <body>
            <h1>Physics Concepts</h1>
            <p>Gravity is an attractive force between masses.</p>
            <script>alert('XSS');</script>
            <div>General relativity describes spacetime curvature.</div>
        </body>
    </html>
    """
    text = extract_plain_text_from_html(html_sample)
    assert "Gravity is an attractive force between masses." in text
    assert "General relativity describes spacetime curvature." in text
    assert "alert('XSS')" not in text
    assert "body { color: red; }" not in text


def test_successful_research_flow_and_citations(research_setup):
    """Verify end-to-end research query, deterministic synthesis, application citations, and persistence."""
    _, repo, provider, service = research_setup

    provider.set_results(
        [
            ResearchResult(
                url="https://example.org/gravity",
                title="Law of Universal Gravitation",
                snippet="Isaac Newton formulated the universal law of gravitation in 1687.",
                publisher="Encyclopedia of Science",
                published_date="2023-01-01",
            ),
            ResearchResult(
                url="https://example.org/relativity",
                title="General Relativity",
                snippet="Albert Einstein published general relativity describing gravitation as spacetime curvature.",
                publisher="Physics Annals",
                published_date="2022-05-10",
            ),
        ]
    )

    summary = service.research(
        query="gravitation theories",
        course_id="course_phy101",
        explicit_opt_in=True,
    )

    assert summary.status == ResearchStatus.SUCCESS
    assert len(summary.sources) == 2
    assert summary.sources[0].citation_marker == "[EXT-1]"
    assert summary.sources[1].citation_marker == "[EXT-2]"
    assert summary.sources[0].url == "https://example.org/gravity"
    assert summary.sources[1].url == "https://example.org/relativity"

    # Check persistence
    persisted = repo.get_research_record(summary.research_id)
    assert persisted is not None
    assert persisted.query == "gravitation theories"
    assert persisted.course_id == "course_phy101"
    assert len(persisted.sources) == 2
    assert persisted.sources[0].citation_index == 1
    assert persisted.sources[1].citation_index == 2


def test_hallucinated_citation_sanitization(research_setup):
    """When a model generates an invalid or hallucinated [EXT-99], it is sanitized."""
    _, _, _, service = research_setup

    sources = [
        service.repository._row_to_source(
            {
                "id": "src_1",
                "research_id": "res_1",
                "citation_index": 1,
                "url": "https://example.com/source1",
                "title": "Verified Source 1",
                "publisher": "Test",
                "published_date": "2024",
                "retrieved_at": "2024-01-01",
                "snippet": "Test snippet",
                "source_type": "web_page",
                "is_valid": 1,
            }
        )
    ]

    class FakeChatClient:
        def chat(self, request):
            class Response:
                content = "Valid finding [EXT-1]. But also invented fact [EXT-99]."

            return Response()

    service.chat_client = FakeChatClient()

    summary_text, valid, invalid = service._generate_grounded_summary("test query", sources)

    assert 1 in valid
    assert "[EXT-99]" in invalid
    assert "[EXT-1]" in summary_text
    assert "[EXT-99]" not in summary_text
    assert "[UNVERIFIED-SOURCE]" in summary_text


def test_safe_external_http_client_valid_fetch(monkeypatch):
    """SafeExternalHttpClient fetches and extracts plain text from valid external HTML."""
    import socket
    import httpx
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))])

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"<html><body><h1>Gravity</h1><p>Universal gravitation is an inverse-square law.</p></body></html>",
        )

    client = SafeExternalHttpClient(transport=httpx.MockTransport(mock_handler))
    final_url, text = client.fetch_url("https://example.org/gravity")
    assert final_url == "https://example.org/gravity"
    assert "Gravity" in text
    assert "Universal gravitation is an inverse-square law." in text


def test_safe_external_http_client_safe_redirect(monkeypatch):
    """SafeExternalHttpClient safely follows redirects after re-validating the destination."""
    import socket
    import httpx
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))])

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url == "https://example.org/short":
            return httpx.Response(
                status_code=301,
                headers={"Location": "https://example.org/final"},
            )
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/plain"},
            content=b"Arrived safely at final destination.",
        )

    client = SafeExternalHttpClient(transport=httpx.MockTransport(mock_handler))
    final_url, text = client.fetch_url("https://example.org/short")
    assert final_url == "https://example.org/final"
    assert text == "Arrived safely at final destination."


def test_safe_external_http_client_blocks_unsafe_redirect(monkeypatch):
    """Redirect to private IP or cloud metadata is blocked immediately before sending request."""
    import socket
    import httpx

    def mock_dns(host, port, proto=0):
        if host == "safe.example.org":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        elif host == "169.254.169.254":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", mock_dns)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://safe.example.org"):
            return httpx.Response(
                status_code=302,
                headers={"Location": "http://169.254.169.254/latest/meta-data/"},
            )
        return httpx.Response(status_code=200, content=b"leak")

    client = SafeExternalHttpClient(transport=httpx.MockTransport(mock_handler))
    with pytest.raises(SSRFValidationError) as exc_info:
        client.fetch_url("https://safe.example.org/redirect-trap")
    assert "link-local/cloud-metadata IP" in str(exc_info.value) or "blocked host" in str(exc_info.value)


def test_safe_external_http_client_exceeds_max_redirects(monkeypatch):
    """Exceeding MAX_REDIRECTS (3) raises SSRFValidationError."""
    import socket
    import httpx
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))])

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=302,
            headers={"Location": f"https://example.org/hop_{request.url.path}"},
        )

    client = SafeExternalHttpClient(transport=httpx.MockTransport(mock_handler))
    with pytest.raises(SSRFValidationError) as exc_info:
        client.fetch_url("https://example.org/start")
    assert "Too many redirects" in str(exc_info.value)


def test_safe_external_http_client_response_size_clamping(monkeypatch):
    """Payloads exceeding MAX_PAGE_BYTES (512 KB) are safely truncated."""
    import socket
    import httpx
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))])

    large_payload = b"A" * (600 * 1024)  # 600 KB
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/plain"},
            content=large_payload,
        )

    client = SafeExternalHttpClient(transport=httpx.MockTransport(mock_handler))
    _, text = client.fetch_url("https://example.org/huge")
    assert len(text.encode("utf-8")) <= 512 * 1024


def test_wikipedia_search_provider_offline_mock(monkeypatch):
    """WikipediaSearchProvider correctly queries endpoint and parses articles offline."""
    import socket
    import httpx
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("208.80.154.224", 443))])

    wiki_json_response = {
        "pages": [
            {
                "id": 1234,
                "key": "Photosynthesis",
                "title": "Photosynthesis",
                "excerpt": "Photosynthesis converts <span class=\"searchmatch\">light</span> into chemical energy.",
                "description": "Biological process",
            }
        ]
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert "en.wikipedia.org" in str(request.url)
        assert "q=photosynthesis" in str(request.url)
        return httpx.Response(status_code=200, json=wiki_json_response)

    from studyos.research.providers.wikipedia import WikipediaSearchProvider
    provider = WikipediaSearchProvider(transport=httpx.MockTransport(mock_handler))
    results = provider.search("photosynthesis", max_results=1)

    assert len(results) == 1
    assert results[0].title == "Photosynthesis"
    assert results[0].url == "https://en.wikipedia.org/wiki/Photosynthesis"
    assert "Photosynthesis converts light into chemical energy." in results[0].snippet
    assert results[0].publisher == "Wikipedia"


def test_wikipedia_search_provider_rejects_unauthorized_host():
    """WikipediaSearchProvider rejects non-Wikimedia endpoints."""
    from studyos.research.providers.wikipedia import WikipediaSearchProvider
    provider = WikipediaSearchProvider()
    provider.BASE_URL = "https://evil-server.com/search"
    with pytest.raises(SSRFValidationError) as exc:
        provider.search("test")
    assert "Must be an official Wikipedia host" in str(exc.value)


def test_real_production_transport_ip_pinning_and_socket_wiring(monkeypatch):
    """Verify real SafeHTTPTransport and SafeSyncBackend wire directly to pinned IP."""
    import socket
    from unittest.mock import patch

    # 1. Instantiate client with NO mock transport (tests real SafeHTTPTransport default)
    client = SafeExternalHttpClient()
    assert client.transport is None

    # 2. Mock DNS resolution to return a specific public IP
    def mock_getaddrinfo(host, port, proto=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    # 3. Create real OS socketpair to simulate remote server cleanly
    server_sock, client_sock = socket.socketpair()
    server_sock.sendall(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 17\r\nConnection: close\r\n\r\nReal Transport OK"
    )

    created_addresses = []

    def mock_create_connection(address, timeout=None, source_address=None):
        created_addresses.append(address)
        return client_sock

    monkeypatch.setattr(socket, "create_connection", mock_create_connection)

    # 4. Execute fetch through real production transport
    final_url, content = client.fetch_url("http://example.org/path")

    # 5. Assertions proving IP pinning and custom backend wiring:
    # Under default httpx.HTTPTransport, create_connection receives ("example.org", 80).
    # Under SafeSyncBackend, create_connection receives the PINNED IP ("93.184.216.34", 80)!
    assert len(created_addresses) == 1
    assert created_addresses[0] == ("93.184.216.34", 80)
    assert created_addresses[0][0] != "example.org"
    assert content == "Real Transport OK"
    assert final_url == "http://example.org/path"

    server_sock.close()


def test_real_production_transport_blocks_socket_on_forbidden_ip(monkeypatch):
    """Forbidden IP destinations fail at DNS check and NEVER attempt socket.create_connection."""
    import socket

    client = SafeExternalHttpClient()

    def mock_getaddrinfo(host, port, proto=0):
        # Resolves to link-local / cloud metadata (169.254.169.254)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    connection_attempted = []

    def mock_create_connection(address, timeout=None, source_address=None):
        connection_attempted.append(address)
        return socket.socket()

    monkeypatch.setattr(socket, "create_connection", mock_create_connection)

    with pytest.raises(SSRFValidationError) as exc:
        client.fetch_url("http://metadata.attacker.org/credentials")
    assert "link-local/cloud-metadata IP" in str(exc.value)

    # Proves no socket connection was attempted
    assert len(connection_attempted) == 0


def test_wikipedia_provider_redirect_escapes_host_policy_rejected(monkeypatch):
    """WikipediaSearchProvider rejects redirects attempting to escape allowed Wikimedia hosts."""
    import socket
    import httpx
    from studyos.research.providers.wikipedia import WikipediaSearchProvider

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("208.80.154.224", 443))],
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        # Redirect pointing to unauthorized domain
        return httpx.Response(
            status_code=302,
            headers={"Location": "https://unauthorized-leak.com/exfiltrate"},
        )

    provider = WikipediaSearchProvider(transport=httpx.MockTransport(mock_handler))
    with pytest.raises(SSRFValidationError) as exc:
        provider.search("algorithm")
    assert "Redirect escaped allowed Wikipedia hosts" in str(exc.value)


