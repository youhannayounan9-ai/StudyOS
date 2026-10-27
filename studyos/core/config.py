"""Configuration management for StudyOS."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


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

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size_tokens")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(f"chunk_overlap_tokens ({v}) must be strictly less than chunk_size_tokens ({chunk_size})")
        return v

    def model_post_init(self, __context) -> None:
        if self.db_path is None:
            self.db_path = self.data_dir / "studyos.db"
        else:
            self.db_path = Path(self.db_path).expanduser().resolve()

    def ensure_directories(self) -> None:
        """Create required local directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.db_path and self.db_path.parent != self.data_dir:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)


def get_default_config() -> AppConfig:
    """Return default application configuration."""
    return AppConfig()
