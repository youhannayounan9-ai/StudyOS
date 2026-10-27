"""Typed data models for local chat inference."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """Role associated with a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single chat message exchanged with the local model."""

    role: ChatRole | str
    content: str

    @property
    def role_str(self) -> str:
        if isinstance(self.role, ChatRole):
            return self.role.value
        return str(self.role)


class ChatRequest(BaseModel):
    """Request payload for local model chat generation."""

    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    format: str | dict[str, Any] | None = Field(
        default=None,
        description="Optional output schema or format (e.g. 'json' or JSON schema dictionary)",
    )


class ChatResponse(BaseModel):
    """Response payload returned by the local model chat provider."""

    model: str
    message: ChatMessage
    done: bool = True
    total_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    @property
    def content(self) -> str:
        """Convenience property for accessing the assistant message text."""
        return self.message.content
