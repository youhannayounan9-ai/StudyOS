"""Deterministic fake chat provider for fast, offline unit and integration tests."""

from __future__ import annotations

from typing import Callable

from studyos.inference.models import ChatMessage, ChatRequest, ChatResponse, ChatRole


class FakeChatProvider:
    """Mock local chat provider that returns deterministic responses without network calls or LLM daemon."""

    def __init__(
        self,
        model_name: str = "fake-tutor",
        default_response: str = "This is a helpful explanation grounded in [E1].",
        response_fn: Callable[[ChatRequest], str] | None = None,
    ) -> None:
        self._model_name = model_name
        self.default_response = default_response
        self.response_fn = response_fn
        self.requests: list[ChatRequest] = []
        self._queued_responses: list[str] = []
        self._error_to_raise: Exception | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def queue_response(self, text: str) -> None:
        """Queue a specific response for the next chat invocation."""
        self._queued_responses.append(text)

    def set_error(self, error: Exception | None) -> None:
        """Configure an exception to be raised on the next chat invocation."""
        self._error_to_raise = error

    def clear(self) -> None:
        """Reset call history, queued responses, and error state."""
        self.requests.clear()
        self._queued_responses.clear()
        self._error_to_raise = None

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Record the request and return the queued or default response."""
        self.requests.append(request)

        if self._error_to_raise is not None:
            err = self._error_to_raise
            self._error_to_raise = None
            raise err

        if self._queued_responses:
            content = self._queued_responses.pop(0)
        elif self.response_fn is not None:
            content = self.response_fn(request)
        else:
            content = self.default_response

        return ChatResponse(
            model=self._model_name,
            message=ChatMessage(role=ChatRole.ASSISTANT, content=content),
            done=True,
            eval_count=len(content.split()),
            prompt_eval_count=sum(len(m.content.split()) for m in request.messages),
        )
