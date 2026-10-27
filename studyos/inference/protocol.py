"""Protocols and exception hierarchies for local chat inference."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from studyos.inference.models import ChatRequest, ChatResponse


class InferenceError(Exception):
    """Base exception for all local model inference errors."""


class OllamaChatConnectionError(InferenceError):
    """Raised when the local Ollama daemon cannot be reached."""


class ChatModelNotFoundError(InferenceError):
    """Raised when the specified chat model is not installed locally in Ollama."""


class ChatTimeoutError(InferenceError):
    """Raised when a chat inference request to Ollama times out."""


class ChatResponseError(InferenceError):
    """Raised when Ollama returns a non-200 status code or error payload."""


class ChatMalformedOutputError(InferenceError):
    """Raised when Ollama returns an empty or unparseable response structure."""


@runtime_checkable
class ChatProvider(Protocol):
    """Protocol for local model chat providers."""

    @property
    def model_name(self) -> str:
        """Configured model name."""
        ...

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a chat completion against the local model."""
        ...
