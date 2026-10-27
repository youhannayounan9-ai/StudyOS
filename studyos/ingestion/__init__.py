"""Document ingestion, parsing, and normalization engine for StudyOS."""

from studyos.ingestion.models import (
    CorruptFileError,
    Course,
    Document,
    DocumentChunk,
    DuplicateDocumentError,
    EmptyFileError,
    EncryptedFileError,
    ExtractionEmptyError,
    FileTooLargeError,
    IngestionError,
    IngestionResult,
    PageElement,
    ParsedDocument,
    UnsupportedFileTypeError,
)
from studyos.ingestion.parsers import (
    DOCXParser,
    DocumentParser,
    PDFParser,
    PPTXParser,
    TXTParser,
    get_parser_for_file,
)

__all__ = [
    "CorruptFileError",
    "Course",
    "DOCXParser",
    "Document",
    "DocumentChunk",
    "DocumentParser",
    "DuplicateDocumentError",
    "EmptyFileError",
    "EncryptedFileError",
    "ExtractionEmptyError",
    "FileTooLargeError",
    "IngestionError",
    "IngestionResult",
    "IngestionService",
    "PDFParser",
    "PPTXParser",
    "PageElement",
    "ParsedDocument",
    "TXTParser",
    "UnsupportedFileTypeError",
    "get_parser_for_file",
]


def __getattr__(name: str):
    if name == "IngestionService":
        from studyos.ingestion.service import IngestionService

        return IngestionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
