"""Unit tests for OllamaEmbeddingProvider and FakeEmbeddingProvider."""

from __future__ import annotations

import httpx
import pytest

from studyos.retrieval.embedding import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    FakeEmbeddingProvider,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEmbeddingProvider,
)


class TestOllamaEmbeddingProvider:
    def test_protocol_conformance(self):
        """Verify OllamaEmbeddingProvider implements EmbeddingProvider protocol."""
        provider = OllamaEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)
        assert provider.model_name == "all-minilm"
        assert provider.dimension == 384

    def test_successful_batch_embedding_modern_api(self):
        """Verify successful batch embedding parsing via modern /api/embed."""
        import json
        dummy_vec = [0.1] * 384

        def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/embed"
            data = json.loads(request.read().decode("utf-8"))
            assert data["model"] == "all-minilm"
            inputs = data.get("input", [])
            return httpx.Response(200, json={"embeddings": [dummy_vec for _ in inputs]})

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = OllamaEmbeddingProvider(client=client, dimension=384)

        results = provider.embed_documents(["hello", "world"])
        assert len(results) == 2
        assert results[0] == dummy_vec
        assert results[1] == dummy_vec

        # embed_query
        q_res = provider.embed_query("search query")
        assert len(q_res) == 384
        assert q_res == dummy_vec

    def test_fallback_to_legacy_embeddings_api(self):
        """When /api/embed returns 404 (endpoint not found), fallback to /api/embeddings."""
        dummy_vec = [0.05] * 384

        def mock_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/embed":
                return httpx.Response(404, text="404 page not found")
            if request.url.path == "/api/embeddings":
                return httpx.Response(200, json={"embedding": dummy_vec})
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = OllamaEmbeddingProvider(client=client, dimension=384)

        results = provider.embed_documents(["fallback test"])
        assert len(results) == 1
        assert results[0] == dummy_vec

    def test_connection_error(self):
        """When daemon is unreachable, raise OllamaConnectionError."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = OllamaEmbeddingProvider(client=client)

        with pytest.raises(OllamaConnectionError, match="Cannot connect to local Ollama service"):
            provider.embed_query("test")

    def test_timeout_error(self):
        """When Ollama times out, raise EmbeddingTimeoutError."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Timed out")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = OllamaEmbeddingProvider(client=client, timeout_seconds=1.0)

        with pytest.raises(EmbeddingTimeoutError, match="timed out"):
            provider.embed_query("test")

    def test_model_not_found_error(self):
        """When model is missing, raise ModelNotFoundError with pull instructions."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model 'all-minilm' not found"})

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = OllamaEmbeddingProvider(client=client, model_name="all-minilm")

        with pytest.raises(ModelNotFoundError, match="ollama pull all-minilm"):
            provider.embed_query("test")

    def test_dimension_mismatch_error(self):
        """When returned vector dimension does not match configured dimension, raise EmbeddingDimensionError."""
        bad_vec = [0.1] * 768  # 768 returned when 384 expected

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [bad_vec]})

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = OllamaEmbeddingProvider(client=client, dimension=384)

        with pytest.raises(EmbeddingDimensionError, match="dimension 768, but configured dimension is 384"):
            provider.embed_query("test")

    def test_malformed_empty_vector_response(self):
        """When Ollama returns empty vector or invalid json, raise EmbeddingResponseError."""
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[]]})

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        provider = OllamaEmbeddingProvider(client=client, dimension=384)

        with pytest.raises(EmbeddingResponseError, match="empty vector"):
            provider.embed_query("test")

    def test_empty_input_list(self):
        """Empty input list returns empty list without making HTTP calls."""
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
        provider = OllamaEmbeddingProvider(client=client)
        assert provider.embed_documents([]) == []


class TestFakeEmbeddingProvider:
    def test_protocol_conformance(self):
        provider = FakeEmbeddingProvider(dimension=384)
        assert isinstance(provider, EmbeddingProvider)
        assert provider.model_name == "fake-minilm"
        assert provider.dimension == 384

    def test_deterministic_and_normalized(self):
        provider = FakeEmbeddingProvider(dimension=384)
        vec1 = provider.embed_query("algorithms and data structures")
        vec2 = provider.embed_query("algorithms and data structures")
        vec3 = provider.embed_query("computer networks and tcp")

        assert len(vec1) == 384
        assert vec1 == vec2  # Exact determinism
        assert vec1 != vec3  # Distinct content produces distinct vector

        # Verify unit length (Euclidean norm approx 1.0)
        norm = sum(x * x for x in vec1)
        assert pytest.approx(norm, 1e-5) == 1.0


class TestEndpointValidation:
    """Deterministic tests for local loopback Ollama endpoint validation and security boundaries."""

    def test_default_loopback_endpoint(self):
        """Default configuration resolves to safe local loopback."""
        from studyos.core.config import AppConfig
        cfg = AppConfig()
        assert cfg.ollama_base_url == "http://127.0.0.1:11434"

        provider = OllamaEmbeddingProvider()
        assert provider._base_url == "http://127.0.0.1:11434"

    def test_supported_loopback_host_forms(self):
        """Supported loopback host forms (127.0.0.1, localhost, [::1], 127.0.0.2, https) are accepted."""
        from studyos.core.config import AppConfig, validate_ollama_base_url

        valid_urls = [
            ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
            ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
            ("http://localhost:11434", "http://localhost:11434"),
            ("http://[::1]:11434", "http://[::1]:11434"),
            ("http://127.0.0.2:11434", "http://127.0.0.2:11434"),
            ("https://localhost:11434", "https://localhost:11434"),
            ("http://127.0.0.1", "http://127.0.0.1"),
        ]

        for input_url, expected in valid_urls:
            # 1. Direct validator
            normalized = validate_ollama_base_url(input_url)
            assert normalized == expected

            # 2. AppConfig integration
            cfg = AppConfig(ollama_base_url=input_url)
            assert cfg.ollama_base_url == expected

            # 3. Provider integration
            provider = OllamaEmbeddingProvider(base_url=input_url)
            assert provider._base_url == expected

    def test_external_hostname_rejected(self):
        """External hostnames are strictly rejected with an actionable loopback privacy error."""
        from studyos.core.config import AppConfig

        external_hosts = [
            "http://api.openai.com",
            "http://api.openai.com/v1",
            "http://example.com:11434",
            "http://ollama.remote.lan:11434",
            "http://localhost.evil.com:11434",
        ]

        for ext_url in external_hosts:
            with pytest.raises(ValueError, match="not a permitted local loopback address"):
                AppConfig(ollama_base_url=ext_url)

            with pytest.raises(ValueError, match="not a permitted local loopback address"):
                OllamaEmbeddingProvider(base_url=ext_url)

    def test_public_ip_rejected(self):
        """Public IP addresses are strictly rejected."""
        from studyos.core.config import AppConfig

        public_ips = [
            "http://8.8.8.8:11434",
            "http://1.1.1.1:11434",
            "http://142.250.190.46:11434",
        ]

        for pub_url in public_ips:
            with pytest.raises(ValueError, match="not a permitted local loopback address"):
                AppConfig(ollama_base_url=pub_url)

            with pytest.raises(ValueError, match="not a permitted local loopback address"):
                OllamaEmbeddingProvider(base_url=pub_url)

    def test_private_lan_ip_rejected(self):
        """Private LAN non-loopback IPs (192.168.x, 10.x, 172.16.x, 0.0.0.0) are strictly rejected."""
        from studyos.core.config import AppConfig

        lan_ips = [
            "http://192.168.1.50:11434",
            "http://10.0.0.1:11434",
            "http://172.16.0.1:11434",
            "http://0.0.0.0:11434",
        ]

        for lan_url in lan_ips:
            with pytest.raises(ValueError, match="not a permitted local loopback address"):
                AppConfig(ollama_base_url=lan_url)

            with pytest.raises(ValueError, match="not a permitted local loopback address"):
                OllamaEmbeddingProvider(base_url=lan_url)

    def test_malformed_url_values_and_unsupported_schemes_rejected(self):
        """Malformed URLs, unsupported schemes, user-info, query params, and fragments are rejected."""
        from studyos.core.config import AppConfig

        invalid_urls = [
            ("ftp://127.0.0.1:11434", "scheme must be 'http' or 'https'"),
            ("file:///etc/passwd", "scheme must be 'http' or 'https'"),
            ("127.0.0.1:11434", "scheme must be 'http' or 'https'"),
            ("http://user:pass@127.0.0.1:11434", "must not contain user credentials"),
            ("http://127.0.0.1:11434?query=1", "must not contain query parameters"),
            ("http://127.0.0.1:11434#section", "must not contain URL fragments"),
            ("http://127.0.0.1:99999", "port is invalid|Port out of range|port must be between 1 and 65535"),
            ("", "must be a non-empty string"),
            ("   ", "must be a non-empty string"),
        ]

        for bad_url, match_str in invalid_urls:
            with pytest.raises(ValueError, match=match_str):
                AppConfig(ollama_base_url=bad_url)

            with pytest.raises(ValueError, match=match_str):
                OllamaEmbeddingProvider(base_url=bad_url)

    def test_rejected_configuration_causes_no_http_request(self):
        """Proof that attempting to initialize with a rejected URL performs zero network requests."""
        calls = []

        def recording_handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200)

        mock_client = httpx.Client(transport=httpx.MockTransport(recording_handler))

        # Attempt to create provider with external URL
        with pytest.raises(ValueError, match="not a permitted local loopback address"):
            OllamaEmbeddingProvider(base_url="http://external-leak.com:11434", client=mock_client)

        # Confirm zero HTTP requests were initiated
        assert len(calls) == 0

    def test_ollama_embedding_ignores_environment_proxy(self, monkeypatch):
        """Proof that HTTP_PROXY, HTTPS_PROXY, and ALL_PROXY are ignored by trust_env=False."""
        monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:8080")
        monkeypatch.setenv("HTTPS_PROXY", "http://192.0.2.1:8080")
        monkeypatch.setenv("ALL_PROXY", "http://192.0.2.1:8080")

        provider = OllamaEmbeddingProvider()
        client = provider._get_client()
        try:
            assert client.trust_env is False
            assert client.follow_redirects is False
            assert len(client._mounts) == 0
        finally:
            client.close()

    def test_ollama_embedding_does_not_follow_redirects(self):
        """Verify that HTTP redirects (301/302) are rejected and never followed to external hosts."""
        def redirect_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"Location": "http://external-leak.com/api/embed"})

        mock_client = httpx.Client(
            transport=httpx.MockTransport(redirect_handler),
            trust_env=False,
            follow_redirects=False,
        )
        provider = OllamaEmbeddingProvider(client=mock_client)

        with pytest.raises(EmbeddingResponseError, match="HTTP error 302"):
            provider.embed_query("test query")


