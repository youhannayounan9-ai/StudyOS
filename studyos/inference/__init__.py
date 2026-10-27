"""StudyOS local inference package for model execution and protocols."""

from studyos.inference.fake import FakeChatProvider
from studyos.inference.models import ChatMessage, ChatRequest, ChatResponse, ChatRole
from studyos.inference.ollama import OllamaChatProvider
from studyos.inference.protocol import (
    ChatMalformedOutputError,
    ChatModelNotFoundError,
    ChatProvider,
    ChatResponseError,
    ChatTimeoutError,
    InferenceError,
    OllamaChatConnectionError,
)

__all__ = [
    "ChatMalformedOutputError",
    "ChatMessage",
    "ChatModelNotFoundError",
    "ChatProvider",
    "ChatRequest",
    "ChatResponse",
    "ChatResponseError",
    "ChatRole",
    "ChatTimeoutError",
    "FakeChatProvider",
    "InferenceError",
    "OllamaChatConnectionError",
    "OllamaChatProvider",
]
