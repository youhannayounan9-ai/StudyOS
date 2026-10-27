"""Unit tests for format-specific document parsers."""

from __future__ import annotations

from pathlib import Path
import pytest

from studyos.ingestion.models import (
    CorruptFileError,
    EmptyFileError,
    EncryptedFileError,
    ExtractionEmptyError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from studyos.ingestion.parsers import (
    DOCXParser,
    PDFParser,
    PPTXParser,
    TXTParser,
    get_parser_for_file,
)
from tests.fixtures.synthetic.build_fixtures import generate_all_fixtures


@pytest.fixture(scope="module")
def synthetic_fixtures(tmp_path_factory) -> dict[str, Path]:
    fixtures_dir = tmp_path_factory.mktemp("fixtures")
    return generate_all_fixtures(fixtures_dir)


class TestPDFParser:
    def test_parse_valid_synthetic_pdf(self, synthetic_fixtures):
        parser = PDFParser()
        pdf_path = synthetic_fixtures["pdf"]
        parsed = parser.parse(pdf_path)

        assert parsed.filename == "synthetic_lecture.pdf"
        assert parsed.file_type == "pdf"
        assert len(parsed.elements) == 2

        # Page 1 assertions
        p1 = parsed.elements[0]
        assert p1.location_type == "page"
        assert p1.location_index == 1
        assert "Foundations of Algorithms" in p1.text
        assert "Big-O notation" in p1.text

        # Page 2 assertions
        p2 = parsed.elements[1]
        assert p2.location_type == "page"
        assert p2.location_index == 2
        assert "Graph Traversal" in p2.text
        assert "Dijkstra's algorithm" in p2.text

    def test_pdf_empty_file_rejected(self, tmp_path):
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        parser = PDFParser()
        with pytest.raises(EmptyFileError, match="empty"):
            parser.parse(empty_pdf)

    def test_pdf_corrupt_file_rejected(self, tmp_path):
        corrupt_pdf = tmp_path / "corrupt.pdf"
        corrupt_pdf.write_bytes(b"%PDF-1.4\nCorrupt file content not valid PDF stream")
        parser = PDFParser()
        with pytest.raises(CorruptFileError):
            parser.parse(corrupt_pdf)

    def test_pdf_file_too_large(self, synthetic_fixtures):
        parser = PDFParser(max_file_size_bytes=100)  # Very small limit
        with pytest.raises(FileTooLargeError, match="exceeds limit"):
            parser.parse(synthetic_fixtures["pdf"])

    def test_scanned_pdf_warning(self, tmp_path):
        # Create a PDF with only 10 characters per page (below 50 threshold)
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        sparse_pdf = tmp_path / "sparse.pdf"
        c = canvas.Canvas(str(sparse_pdf), pagesize=letter)
        c.drawString(100, 100, "Tiny")
        c.showPage()
        c.save()

        parser = PDFParser()
        parsed = parser.parse(sparse_pdf)
        assert len(parsed.warnings) == 1
        assert "ScannedDocumentWarning" in parsed.warnings[0]


    def test_encrypted_pdf_rejected_with_specific_error(self, tmp_path, synthetic_fixtures):
        import pypdf

        # Create encrypted PDF using pypdf writer
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.encrypt("super_secret_password")
        enc_path = tmp_path / "encrypted.pdf"
        with open(enc_path, "wb") as f:
            writer.write(f)

        # Confirm file is recognized as encrypted by pypdf
        reader = pypdf.PdfReader(str(enc_path))
        assert reader.is_encrypted is True

        parser = PDFParser()
        # Verify parser raises EncryptedFileError and does not return empty text or report success
        with pytest.raises(EncryptedFileError, match="password-protected|encrypted"):
            parser.parse(enc_path)

        # Verify normal unencrypted PDF continues to parse correctly
        normal_pdf = synthetic_fixtures["pdf"]
        parsed_normal = parser.parse(normal_pdf)
        assert len(parsed_normal.elements) == 2
        assert "Foundations of Algorithms" in parsed_normal.elements[0].text


class TestDOCXParser:
    def test_parse_valid_synthetic_docx(self, synthetic_fixtures):
        parser = DOCXParser()
        docx_path = synthetic_fixtures["docx"]
        parsed = parser.parse(docx_path)

        assert parsed.filename == "synthetic_notes.docx"
        assert parsed.file_type == "docx"
        assert len(parsed.elements) >= 2

        # Check headings and content
        sec1 = parsed.elements[0]
        assert sec1.location_type == "section"
        assert "Relational Database Internals" in sec1.text
        assert "ACID properties" in sec1.text

        # Check table contents preserved
        sec2 = parsed.elements[1]
        assert "B+ Tree" in sec2.text
        assert "O(log n)" in sec2.text
        assert "|" in sec2.text  # Table delimiter

    def test_docx_empty_file_rejected(self, tmp_path):
        empty_docx = tmp_path / "empty.docx"
        empty_docx.write_bytes(b"")
        parser = DOCXParser()
        with pytest.raises(EmptyFileError):
            parser.parse(empty_docx)

    def test_docx_corrupt_file_rejected(self, tmp_path):
        corrupt_docx = tmp_path / "corrupt.docx"
        corrupt_docx.write_bytes(b"PK\x03\x04corrupt zip content")
        parser = DOCXParser()
        with pytest.raises(CorruptFileError):
            parser.parse(corrupt_docx)


class TestPPTXParser:
    def test_parse_valid_synthetic_pptx(self, synthetic_fixtures):
        parser = PPTXParser()
        pptx_path = synthetic_fixtures["pptx"]
        parsed = parser.parse(pptx_path)

        assert parsed.filename == "synthetic_slides.pptx"
        assert parsed.file_type == "pptx"
        assert len(parsed.elements) == 4

        # Slide 1
        s1 = parsed.elements[0]
        assert s1.location_type == "slide"
        assert s1.location_index == 1
        assert s1.location_title == "Introduction to Data Structures"
        assert "Arrays provide O(1)" in s1.text

        # Slide 2
        s2 = parsed.elements[1]
        assert s2.location_index == 2
        assert s2.location_title == "Tree Architectures and Balancing"
        assert "AVL and Red-Black" in s2.text

        # Slide 3
        s3 = parsed.elements[2]
        assert s3.location_index == 3
        assert s3.location_title == "Hash Tables and Collision Resolution"
        assert "linear probing" in s3.text

        # Slide 4 (Table shape)
        s4 = parsed.elements[3]
        assert s4.location_type == "slide"
        assert s4.location_index == 4
        assert s4.location_title == "Algorithm Complexity Summary"
        assert "Algorithm" in s4.text
        assert "QuickSort" in s4.text
        assert "MergeSort" in s4.text
        assert "| QuickSort | O(n log n) | O(log n) |" in s4.text
        assert "| MergeSort | O(n log n) | O(n) |" in s4.text

    def test_pptx_table_with_empty_cells(self, tmp_path):
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = "Sparse Table Slide"
        t_shape = slide.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(2))
        tbl = t_shape.table
        tbl.cell(0, 0).text = "Filled"
        pptx_file = tmp_path / "sparse_table.pptx"
        prs.save(str(pptx_file))

        parser = PPTXParser()
        parsed = parser.parse(pptx_file)
        assert len(parsed.elements) == 1
        assert "| Filled |  |" in parsed.elements[0].text

    def test_pptx_empty_file_rejected(self, tmp_path):
        empty_pptx = tmp_path / "empty.pptx"
        empty_pptx.write_bytes(b"")
        parser = PPTXParser()
        with pytest.raises(EmptyFileError):
            parser.parse(empty_pptx)


class TestTXTParser:
    def test_parse_valid_synthetic_txt(self, synthetic_fixtures):
        parser = TXTParser()
        txt_path = synthetic_fixtures["txt"]
        parsed = parser.parse(txt_path)

        assert parsed.filename == "synthetic_doc.txt"
        assert parsed.file_type == "txt"
        assert len(parsed.elements) == 2

        sec1 = parsed.elements[0]
        assert sec1.location_title == "Computer Systems: Memory Hierarchy"
        assert "L1, L2, and L3 caches" in sec1.text

        sec2 = parsed.elements[1]
        assert sec2.location_title == "Virtual Memory and Paging"
        assert "Translation Lookaside Buffers" in sec2.text
        assert "Δt = α · tc" in sec2.text  # Unicode math retained

    def test_txt_with_utf8_bom(self, tmp_path):
        bom_file = tmp_path / "bom.txt"
        bom_file.write_bytes(b"\xef\xbb\xbf# Header with BOM\nContent goes here.")
        parser = TXTParser()
        parsed = parser.parse(bom_file)
        assert len(parsed.elements) == 1
        assert "Content goes here" in parsed.elements[0].text

    def test_txt_corrupt_non_utf8(self, tmp_path):
        bad_file = tmp_path / "bad.txt"
        bad_file.write_bytes(b"\xff\xfe\x00\x00Invalid UTF-8 binary bytes \x80\x81")
        parser = TXTParser()
        with pytest.raises(CorruptFileError, match="UTF-8"):
            parser.parse(bad_file)


class TestParserFactory:
    def test_get_parser_supported(self):
        assert isinstance(get_parser_for_file(Path("doc.pdf")), PDFParser)
        assert isinstance(get_parser_for_file(Path("doc.docx")), DOCXParser)
        assert isinstance(get_parser_for_file(Path("doc.pptx")), PPTXParser)
        assert isinstance(get_parser_for_file(Path("doc.txt")), TXTParser)
        assert isinstance(get_parser_for_file(Path("doc.md")), TXTParser)

    def test_get_parser_unsupported_extensions(self):
        with pytest.raises(UnsupportedFileTypeError, match="Unsupported file format"):
            get_parser_for_file(Path("file.xyz"))

        with pytest.raises(UnsupportedFileTypeError, match="Legacy .doc"):
            get_parser_for_file(Path("file.doc"))

        with pytest.raises(UnsupportedFileTypeError, match="Legacy .ppt"):
            get_parser_for_file(Path("file.ppt"))
