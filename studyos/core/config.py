"""Configuration management for StudyOS."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import urllib.parse
from pydantic import BaseModel, Field, field_validator


def validate_ollama_base_url(url: str) -> str:
    """Validate that the Ollama base URL strictly targets a local loopback endpoint.

    Requirements:
    - Must be a non-empty string.
    - Scheme must be explicitly 'http' or 'https'.
    - Host must be an explicitly permitted loopback destination: '127.0.0.1', 'localhost', '::1',
      or an IPv4 loopback address (127.0.0.0/8).
    - Remote hostnames, public IPs, and LAN IPs (e.g. 192.168.x, 10.x, 172.16.x) are strictly rejected.
    - User credentials (username, password, '@') are rejected to prevent hostname spoofing.
    - Query parameters and URL fragments are rejected.
    - No DNS lookups are performed for security and offline determinism.
    - Returns normalized base URL without trailing slash.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise ValueError("ollama_base_url must be a non-empty string.")

    raw = url.strip()

    # Reject user-info / credentials which could be used for hostname confusion (e.g. http://127.0.0.1@evil.com)
    if "@" in raw:
        raise ValueError("ollama_base_url must not contain user credentials or '@' characters.")

    try:
        parsed = urllib.parse.urlsplit(raw)
    except Exception as e:
        raise ValueError(f"ollama_base_url is malformed: {e}") from e

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"ollama_base_url scheme must be 'http' or 'https', got '{parsed.scheme or '<none>'}'. "
            f"Example: 'http://127.0.0.1:11434'"
        )

    if parsed.query:
        raise ValueError("ollama_base_url must not contain query parameters.")

    if parsed.fragment:
        raise ValueError("ollama_base_url must not contain URL fragments.")

    if parsed.username is not None or parsed.password is not None:
        raise ValueError("ollama_base_url must not contain authentication credentials.")

    if not parsed.hostname:
        raise ValueError("ollama_base_url must contain a valid host (e.g. 127.0.0.1 or localhost).")

    host = parsed.hostname.lower()

    # Check loopback strictly without DNS resolution
    is_loopback = False
    if host == "localhost":
        is_loopback = True
    else:
        try:
            ip = ipaddress.ip_address(host)
            is_loopback = ip.is_loopback
        except ValueError:
            # Not a valid IP literal and not 'localhost'
            is_loopback = False

    if not is_loopback:
        raise ValueError(
            f"ollama_base_url host '{host}' is not a permitted local loopback address. "
            f"Only local loopback endpoints ('127.0.0.1', 'localhost', '::1') are permitted "
            f"to guarantee zero data egress."
        )

    try:
        port = parsed.port
    except ValueError as e:
        raise ValueError(f"ollama_base_url port is invalid: {e}") from e

    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"ollama_base_url port must be between 1 and 65535, got {port}.")

    host_str = f"[{host}]" if ":" in host else host
    port_str = f":{port}" if port is not None else ""
    path = parsed.path.rstrip("/")

    return f"{scheme}://{host_str}{port_str}{path}".rstrip("/")


class AppConfig(BaseModel):
    """Application configuration for StudyOS."""

    data_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("STUDYOS_DATA_DIR", str(Path.home() / ".studyos"))
        ).expanduser().resolve()
    )
    db_path: Path | None = None
    max_file_size_bytes: int = Field(
        default=100 * 1024 * 1024,  # 100 MiB
        description="Maximum permitted file size in bytes for document ingestion",
        ge=1024,
    )
    chunk_size_tokens: int = Field(
        default=500,
        description="Target chunk size in estimated tokens",
        ge=100,
        le=2000,
    )
    chunk_overlap_tokens: int = Field(
        default=75,
        description="Overlap between consecutive chunks in estimated tokens",
        ge=0,
        le=500,
    )
    min_chunk_size_tokens: int = Field(
        default=50,
        description="Minimum chunk size in estimated tokens before merging or keeping intact",
        ge=10,
    )

    # Stage 2: Retrieval & Vector Store Configuration
    chroma_dir: Path | None = None
    chroma_collection_name: str = Field(
        default_factory=lambda: os.getenv("STUDYOS_CHROMA_COLLECTION", "studyos_chunks"),
        description="ChromaDB persistent collection name",
    )
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("STUDYOS_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        description="Local Ollama service HTTP base URL",
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("STUDYOS_EMBEDDING_MODEL", "all-minilm"),
        description="Ollama embedding model name",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Expected dense embedding vector dimension (384 for all-minilm)",
        ge=1,
    )
    embedding_timeout_seconds: float = Field(
        default=30.0,
        description="HTTP timeout for embedding generation requests in seconds",
        ge=1.0,
    )
    rrf_k: int = Field(
        default=60,
        description="Reciprocal Rank Fusion smoothing constant (standard default: 60)",
        ge=1,
    )
    anonymized_telemetry: bool = Field(
        default=False,
        description="Disable external ChromaDB telemetry to maintain 100% local privacy",
    )

    # Stage 3: Local Tutor & Chat Model Configuration
    tutor_model: str = Field(
        default_factory=lambda: os.getenv("STUDYOS_TUTOR_MODEL", "qwen2.5:7b"),
        description="Local Ollama chat model for evidence-grounded tutoring",
    )
    tutor_temperature: float = Field(
        default=0.2,
        description="Sampling temperature for tutor explanations (low temperature for factual precision)",
        ge=0.0,
        le=1.0,
    )
    tutor_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate for tutor answers",
        ge=64,
        le=4096,
    )
    tutor_timeout_seconds: float = Field(
        default=60.0,
        description="HTTP timeout in seconds for local chat requests",
        ge=1.0,
    )
    tutor_evidence_k: int = Field(
        default=5,
        description="Maximum number of evidence chunks selected for tutor context",
        ge=1,
        le=20,
    )
    tutor_min_relevance_score: float = Field(
        default=0.01,
        description="Minimum RRF relevance score threshold before tutor considers evidence insufficient",
        ge=0.0,
    )

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size_tokens")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(f"chunk_overlap_tokens ({v}) must be strictly less than chunk_size_tokens ({chunk_size})")
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        return validate_ollama_base_url(v)

    def model_post_init(self, __context) -> None:
        if self.db_path is None:
            self.db_path = self.data_dir / "studyos.db"
        else:
            self.db_path = Path(self.db_path).expanduser().resolve()

        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma"
        else:
            self.chroma_dir = Path(self.chroma_dir).expanduser().resolve()

    def ensure_directories(self) -> None:
        """Create required local directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.db_path and self.db_path.parent != self.data_dir:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.chroma_dir:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> AppConfig:
    """Return default application configuration."""
    return AppConfig()
