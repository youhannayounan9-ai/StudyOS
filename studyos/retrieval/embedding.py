"""Local embedding providers, protocols, and Ollama integration."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable
import httpx

from studyos.core.config import AppConfig, validate_ollama_base_url


class EmbeddingError(Exception):
    """Base exception for all embedding generation errors."""


class OllamaConnectionError(EmbeddingError):
    """Raised when the local Ollama daemon cannot be reached."""


class ModelNotFoundError(EmbeddingError):
    """Raised when the specified embedding model is not installed locally in Ollama."""


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding request to Ollama times out."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when Ollama returns an empty or malformed embedding payload."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when the returned embedding vector dimension does not match the configured dimension."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for local dense vector embedding providers."""

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Dense vector dimensionality."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding vector for a single query string."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of document chunk texts."""
        ...


class OllamaEmbeddingProvider:
    """Local embedding provider connecting to Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model_name: str = "all-minilm",
        dimension: int = 384,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = validate_ollama_base_url(base_url)
        self._model_name = model_name
        self._dimension = dimension
        self._timeout_seconds = timeout_seconds
        self._client = client

    @classmethod
    def from_config(cls, config: AppConfig) -> OllamaEmbeddingProvider:
        """Instantiate an Ollama embedding provider from application configuration."""
        return cls(
            base_url=config.ollama_base_url,
            model_name=config.embedding_model,
            dimension=config.embedding_dimension,
            timeout_seconds=config.embedding_timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            timeout=self._timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )

    def _execute_request(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Validate texts
        cleaned_texts = [t if t.strip() else " " for t in texts]

        client = self._get_client()
        should_close = self._client is None

        try:
            # 1. Try modern Ollama /api/embed batch endpoint
            url = f"{self._base_url}/api/embed"
            payload = {"model": self._model_name, "input": cleaned_texts}

            try:
                resp = client.post(url, json=payload)
            except (httpx.ConnectError, httpx.NetworkError) as e:
                raise OllamaConnectionError(
                    f"Cannot connect to local Ollama service at {self._base_url}. "
                    f"Ensure Ollama is running ('ollama serve')."
                ) from e
            except httpx.TimeoutException as e:
                raise EmbeddingTimeoutError(
                    f"Embedding request to Ollama at {self._base_url} timed out after {self._timeout_seconds}s."
                ) from e

            # Handle 404 (endpoint not found or model not found)
            if resp.status_code == 404:
                resp_text = resp.text.lower()
                if "model" in resp_text:
                    raise ModelNotFoundError(
                        f"Embedding model '{self._model_name}' not found in Ollama at {self._base_url}. "
                        f"Run 'ollama pull {self._model_name}' to install it locally."
                    )
                # Fallback to legacy single /api/embeddings endpoint
                return self._execute_legacy_request(client, cleaned_texts)

            if resp.status_code != 200:
                raise EmbeddingResponseError(
                    f"Ollama returned HTTP error {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(cleaned_texts):
                raise EmbeddingResponseError(
                    f"Expected {len(cleaned_texts)} embeddings from Ollama, got {data}"
                )

            # Validate each embedding
            for idx, vec in enumerate(embeddings):
                if not isinstance(vec, list) or len(vec) == 0:
                    raise EmbeddingResponseError(f"Ollama returned empty vector at index {idx}.")
                if len(vec) != self._dimension:
                    raise EmbeddingDimensionError(
                        f"Ollama returned vector with dimension {len(vec)}, but configured dimension is {self._dimension}."
                    )

            return embeddings

        finally:
            if should_close:
                client.close()

    def _execute_legacy_request(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        """Fallback for older Ollama versions supporting only /api/embeddings."""
        url = f"{self._base_url}/api/embeddings"
        results: list[list[float]] = []

        for text in texts:
            payload = {"model": self._model_name, "prompt": text}
            try:
                resp = client.post(url, json=payload)
            except (httpx.ConnectError, httpx.NetworkError) as e:
                raise OllamaConnectionError(
                    f"Cannot connect to local Ollama service at {self._base_url}."
                ) from e
            except httpx.TimeoutException as e:
                raise EmbeddingTimeoutError(
                    f"Embedding request to Ollama timed out after {self._timeout_seconds}s."
                ) from e

            if resp.status_code == 404:
                raise ModelNotFoundError(
                    f"Embedding model '{self._model_name}' not found in Ollama. "
                    f"Run 'ollama pull {self._model_name}'."
                )
            if resp.status_code != 200:
                raise EmbeddingResponseError(
                    f"Ollama legacy embedding endpoint failed with HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            vec = data.get("embedding")
            if not isinstance(vec, list) or len(vec) == 0:
                raise EmbeddingResponseError("Ollama legacy embedding endpoint returned empty vector.")
            if len(vec) != self._dimension:
                raise EmbeddingDimensionError(
                    f"Ollama returned vector with dimension {len(vec)}, expected {self._dimension}."
                )
            results.append(vec)

        return results

    def embed_query(self, text: str) -> list[float]:
        res = self._execute_request([text])
        if not res:
            raise EmbeddingResponseError("Empty embedding returned for query.")
        return res[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._execute_request(texts)


class FakeEmbeddingProvider:
    """Deterministic, local fake embedding provider for fast unit and integration tests.

    Produces mathematically consistent unit vectors derived from content hashing,
    enabling full vector search testing without external dependencies, models, or network calls.
    """

    def __init__(self, model_name: str = "fake-minilm", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> list[float]:
        # Hash text to generate a deterministic float vector
        cleaned = text.strip().lower()
        h = hashlib.sha256(cleaned.encode("utf-8")).digest()
        vec: list[float] = []

        # Build raw components
        for i in range(self._dimension):
            byte_val = h[(i * 7) % len(h)]
            # Spread pseudo-randomly centered around 0.0
            component = (byte_val - 128.0) / 128.0 + ((i % 11) - 5) * 0.02
            vec.append(component)

        # Normalize to unit length (Euclidean L2 norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0

        return vec

    def embed_query(self, text: str) -> list[float]:
        return self._generate_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]
