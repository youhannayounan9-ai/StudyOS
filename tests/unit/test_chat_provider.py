"""Unit tests for local Ollama chat provider and inference contracts."""

from __future__ import annotations

import httpx
import pytest

from studyos.core.config import AppConfig
from studyos.inference.fake import FakeChatProvider
from studyos.inference.models import ChatMessage, ChatRequest, ChatResponse, ChatRole
from studyos.inference.ollama import OllamaChatProvider
from studyos.inference.protocol import (
    ChatMalformedOutputError,
    ChatModelNotFoundError,
    ChatResponseError,
    ChatTimeoutError,
    OllamaChatConnectionError,
)


def test_ollama_chat_success():
    """Verify standard successful chat response extraction from Ollama /api/chat."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = request.read().decode("utf-8")
        assert "qwen2.5:7b" in payload
        assert "Explain Dijkstra" in payload

        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:7b",
                "created_at": "2026-10-27T12:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "Dijkstra's algorithm finds shortest paths [E1].",
                },
                "done": True,
                "total_duration": 123456789,
                "prompt_eval_count": 42,
                "eval_count": 18,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(
        base_url="http://127.0.0.1:11434",
        model_name="qwen2.5:7b",
        client=client,
    )

    req = ChatRequest(
        model="qwen2.5:7b",
        messages=[
            ChatMessage(role=ChatRole.SYSTEM, content="You are a tutor."),
            ChatMessage(role=ChatRole.USER, content="Explain Dijkstra"),
        ],
        temperature=0.2,
    )
    resp = provider.chat(req)

    assert isinstance(resp, ChatResponse)
    assert resp.model == "qwen2.5:7b"
    assert resp.content == "Dijkstra's algorithm finds shortest paths [E1]."
    assert resp.message.role == "assistant"
    assert resp.eval_count == 18
    assert resp.done is True


def test_ollama_chat_model_not_found():
    """When Ollama returns 404 with missing model message, raise ChatModelNotFoundError with pull advice."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error": "model \'missing-model:7b\' not found"}')

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(
        base_url="http://127.0.0.1:11434",
        model_name="missing-model:7b",
        client=client,
    )

    req = ChatRequest(
        model="missing-model:7b",
        messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
    )

    with pytest.raises(ChatModelNotFoundError) as exc_info:
        provider.chat(req)

    msg = str(exc_info.value)
    assert "missing-model:7b" in msg
    assert "ollama pull missing-model:7b" in msg


def test_ollama_chat_connection_error():
    """When daemon is unreachable, raise OllamaChatConnectionError with restart guidance."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(base_url="http://127.0.0.1:11434", client=client)

    req = ChatRequest(
        model="qwen2.5:7b",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
    )

    with pytest.raises(OllamaChatConnectionError) as exc_info:
        provider.chat(req)

    assert "Cannot connect to local Ollama service" in str(exc_info.value)
    assert "ollama serve" in str(exc_info.value)


def test_ollama_chat_timeout():
    """When request times out, raise ChatTimeoutError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timed out waiting for response")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(
        base_url="http://127.0.0.1:11434",
        timeout_seconds=5.0,
        client=client,
    )

    req = ChatRequest(
        model="qwen2.5:7b",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
    )

    with pytest.raises(ChatTimeoutError) as exc_info:
        provider.chat(req)

    assert "timed out after 5.0s" in str(exc_info.value)


def test_ollama_chat_server_error_status():
    """When Ollama returns non-200 HTTP status, raise ChatResponseError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal server error")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(base_url="http://127.0.0.1:11434", client=client)

    req = ChatRequest(
        model="qwen2.5:7b",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
    )

    with pytest.raises(ChatResponseError) as exc_info:
        provider.chat(req)

    assert "500" in str(exc_info.value)


def test_ollama_chat_malformed_json_response():
    """When Ollama returns non-JSON text, raise ChatMalformedOutputError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not valid json")

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(base_url="http://127.0.0.1:11434", client=client)

    req = ChatRequest(
        model="qwen2.5:7b",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
    )

    with pytest.raises(ChatMalformedOutputError) as exc_info:
        provider.chat(req)

    assert "Failed to parse Ollama JSON" in str(exc_info.value)


def test_ollama_chat_missing_message_payload():
    """When Ollama returns JSON without 'message' dictionary, raise ChatMalformedOutputError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(base_url="http://127.0.0.1:11434", client=client)

    req = ChatRequest(
        model="qwen2.5:7b",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
    )

    with pytest.raises(ChatMalformedOutputError) as exc_info:
        provider.chat(req)

    assert "missing 'message' dictionary" in str(exc_info.value)


def test_ollama_chat_missing_content_field():
    """When Ollama returns message without 'content', raise ChatMalformedOutputError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant"}})

    client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    provider = OllamaChatProvider(base_url="http://127.0.0.1:11434", client=client)

    req = ChatRequest(
        model="qwen2.5:7b",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello")],
    )

    with pytest.raises(ChatMalformedOutputError) as exc_info:
        provider.chat(req)

    assert "missing 'content' field" in str(exc_info.value)


def test_ollama_chat_rejects_non_loopback():
    """Base URL validation must reject public or LAN endpoints."""
    with pytest.raises(ValueError, match="not a permitted local loopback address"):
        OllamaChatProvider(base_url="http://192.168.1.50:11434")

    with pytest.raises(ValueError, match="not a permitted local loopback address"):
        OllamaChatProvider(base_url="https://api.openai.com")


def test_ollama_chat_client_settings():
    """Default client created internally must enforce trust_env=False and follow_redirects=False."""
    provider = OllamaChatProvider(base_url="http://127.0.0.1:11434")
    client = provider._get_client()

    assert client.trust_env is False
    assert client.follow_redirects is False
    client.close()


def test_fake_chat_provider_behavior():
    """Verify FakeChatProvider supports queueing responses, custom functions, and error injection."""
    fake = FakeChatProvider(default_response="Default response")

    req = ChatRequest(
        model="fake",
        messages=[ChatMessage(role=ChatRole.USER, content="Test")],
    )

    # 1. Default response
    r1 = fake.chat(req)
    assert r1.content == "Default response"
    assert len(fake.requests) == 1

    # 2. Queued response
    fake.queue_response("Queued answer [E1]")
    r2 = fake.chat(req)
    assert r2.content == "Queued answer [E1]"
    assert len(fake.requests) == 2

    # 3. Next one falls back to default
    r3 = fake.chat(req)
    assert r3.content == "Default response"

    # 4. Error injection
    fake.set_error(RuntimeError("Injected failure"))
    with pytest.raises(RuntimeError, match="Injected failure"):
        fake.chat(req)

    # 5. Clear
    fake.clear()
    assert len(fake.requests) == 0
