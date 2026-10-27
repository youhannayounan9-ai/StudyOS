"""Ollama local HTTP chat inference provider."""

from __future__ import annotations

import logging
from typing import Any
import httpx

from studyos.core.config import AppConfig, validate_ollama_base_url
from studyos.inference.models import ChatMessage, ChatRequest, ChatResponse, ChatRole
from studyos.inference.protocol import (
    ChatMalformedOutputError,
    ChatModelNotFoundError,
    ChatResponseError,
    ChatTimeoutError,
    OllamaChatConnectionError,
)

logger = logging.getLogger(__name__)


class OllamaChatProvider:
    """Local chat completion provider connecting to Ollama HTTP API (/api/chat).

    Features:
    - Enforces local loopback endpoints via validate_ollama_base_url.
    - Uses HTTPX with trust_env=False and follow_redirects=False for strict privacy.
    - Validates response schemas, status codes, timeouts, and missing model errors.
    - Avoids logging private prompt text or document passages.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "qwen2.5:7b",
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = validate_ollama_base_url(base_url)
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._client = client

    @classmethod
    def from_config(cls, config: AppConfig) -> OllamaChatProvider:
        """Construct OllamaChatProvider from application configuration."""
        return cls(
            base_url=config.ollama_base_url,
            model_name=config.tutor_model,
            timeout_seconds=config.tutor_timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            timeout=self._timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute chat completion request against local Ollama daemon."""
        client = self._get_client()
        should_close = self._client is None

        model_to_use = request.model or self._model_name
        url = f"{self._base_url}/api/chat"

        # Build options dictionary
        options: dict[str, Any] = {
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        payload: dict[str, Any] = {
            "model": model_to_use,
            "messages": [
                {"role": m.role_str, "content": m.content}
                for m in request.messages
            ],
            "stream": False,
            "options": options,
        }

        if request.format is not None:
            payload["format"] = request.format

        logger.debug(
            "Sending chat request to Ollama: model=%s, message_count=%d, temperature=%.2f",
            model_to_use,
            len(request.messages),
            request.temperature,
        )

        try:
            try:
                resp = client.post(url, json=payload)
            except (httpx.ConnectError, httpx.NetworkError) as e:
                raise OllamaChatConnectionError(
                    f"Cannot connect to local Ollama service at {self._base_url}. "
                    f"Ensure Ollama is running ('ollama serve')."
                ) from e
            except httpx.TimeoutException as e:
                raise ChatTimeoutError(
                    f"Chat request to Ollama at {self._base_url} timed out after {self._timeout_seconds}s."
                ) from e

            # Handle 404 (model missing or unknown endpoint)
            if resp.status_code == 404:
                resp_text = resp.text.lower()
                if "model" in resp_text:
                    raise ChatModelNotFoundError(
                        f"Chat model '{model_to_use}' not found in Ollama at {self._base_url}. "
                        f"Run 'ollama pull {model_to_use}' to install it locally."
                    )
                raise ChatResponseError(f"Ollama endpoint not found (HTTP 404): {resp.text}")

            if not resp.is_success:
                raise ChatResponseError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                )

            try:
                data = resp.json()
            except Exception as e:
                raise ChatMalformedOutputError(f"Failed to parse Ollama JSON response: {e}") from e

            if not isinstance(data, dict):
                raise ChatMalformedOutputError("Ollama returned non-dictionary JSON response.")

            message_data = data.get("message")
            if not isinstance(message_data, dict):
                raise ChatMalformedOutputError(
                    "Ollama response payload missing 'message' dictionary."
                )

            content = message_data.get("content")
            if content is None:
                raise ChatMalformedOutputError(
                    "Ollama response 'message' dictionary missing 'content' field."
                )

            role = message_data.get("role", "assistant")

            logger.debug(
                "Received chat response from Ollama: model=%s, eval_count=%s",
                data.get("model", model_to_use),
                data.get("eval_count"),
            )

            return ChatResponse(
                model=data.get("model", model_to_use),
                message=ChatMessage(role=role, content=content),
                done=data.get("done", True),
                total_duration_ns=data.get("total_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                eval_count=data.get("eval_count"),
            )
        finally:
            if should_close:
                client.close()
