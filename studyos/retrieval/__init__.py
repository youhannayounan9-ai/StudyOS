"""StudyOS Retrieval Engine: Lexical (FTS5), Semantic (ChromaDB), and Hybrid RRF search."""

from studyos.retrieval.embedding import (
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
    FakeEmbeddingProvider,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEmbeddingProvider,
)
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.lexical import LexicalSearchService
from studyos.retrieval.models import (
    RetrievalMode,
    RetrievalResponse,
    RetrievedChunk,
    SyncStats,
)
from studyos.retrieval.semantic import SemanticSearchService
from studyos.retrieval.vector_store import (
    ChromaVectorStore,
    IncompatibleCollectionError,
    VectorStoreError,
)

__all__ = [
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingResponseError",
    "EmbeddingTimeoutError",
    "FakeEmbeddingProvider",
    "ModelNotFoundError",
    "OllamaConnectionError",
    "OllamaEmbeddingProvider",
    "ChromaVectorStore",
    "IncompatibleCollectionError",
    "VectorStoreError",
    "LexicalSearchService",
    "SemanticSearchService",
    "HybridRetrievalService",
    "RetrievalMode",
    "RetrievalResponse",
    "RetrievedChunk",
    "SyncStats",
]
