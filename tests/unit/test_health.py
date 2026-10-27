"""Unit tests for Ollama pre-flight health diagnostics."""

from __future__ import annotations

import httpx
import pytest

from studyos.core.config import AppConfig
from studyos.core.health import HealthStatus, check_ollama_health


def test_ollama_health_success():
    """When Ollama responds with version and matching model tags, report HEALTHY."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.40.7"})
        if url.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={"models": [{"name": "all-minilm:latest"}, {"name": "qwen2.5:7b"}]},
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    config = AppConfig(embedding_model="all-minilm")
    report = check_ollama_health(config, client=client)

    assert report.status == HealthStatus.HEALTHY
    assert report.is_ready is True
    assert report.version == "0.40.7"
    assert "all-minilm:latest" in report.installed_models


def test_ollama_health_model_missing():
    """When Ollama is online but requested embedding model is absent, report MODEL_MISSING."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3:latest"}]},
            )
        return httpx.Response(200, json={"version": "0.40.7"})

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    config = AppConfig(embedding_model="all-minilm")
    report = check_ollama_health(config, client=client)

    assert report.status == HealthStatus.MODEL_MISSING
    assert report.is_ready is False
    assert "ollama pull all-minilm" in (report.actionable_remediation or "")


def test_ollama_health_connection_error():
    """When daemon is unreachable, report OFFLINE with actionable startup command."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused to 127.0.0.1:11434")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    report = check_ollama_health(client=client)
    assert report.status == HealthStatus.OFFLINE
    assert report.is_ready is False
    assert "ollama serve" in (report.actionable_remediation or "")


def test_ollama_health_timeout():
    """When request times out, report OFFLINE with overload advisory."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Timed out connecting to daemon")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    report = check_ollama_health(timeout_seconds=0.1, client=client)
    assert report.status == HealthStatus.OFFLINE
    assert report.is_ready is False
    assert "timed out" in report.message.lower()


def test_ollama_health_ignores_environment_proxy(monkeypatch):
    """Verify check_ollama_health constructs an HTTPX client with trust_env=False ignoring proxy env vars."""
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://192.0.2.1:8080")
    monkeypatch.setenv("ALL_PROXY", "http://192.0.2.1:8080")

    created_clients = []
    orig_client_init = httpx.Client.__init__

    def tracking_init(self, *args, **kwargs):
        orig_client_init(self, *args, **kwargs)
        created_clients.append(self)

    monkeypatch.setattr(httpx.Client, "__init__", tracking_init)

    # Calling check_ollama_health without providing a client uses default factory
    report = check_ollama_health(timeout_seconds=0.01)

    assert len(created_clients) == 1
    client = created_clients[0]
    assert client.trust_env is False
    assert client.follow_redirects is False
    assert len(client._mounts) == 0


def test_ollama_health_does_not_follow_redirects():
    """Verify that HTTP redirects (301/302) are not followed by check_ollama_health."""
    def redirect_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://external-leak.com/api/tags"})

    mock_client = httpx.Client(
        transport=httpx.MockTransport(redirect_handler),
        trust_env=False,
        follow_redirects=False,
    )
    report = check_ollama_health(client=mock_client)

    assert report.status == HealthStatus.ERROR
    assert "302" in report.message

