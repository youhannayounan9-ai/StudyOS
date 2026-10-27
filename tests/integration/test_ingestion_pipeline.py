"""Integration tests for the complete StudyOS ingestion pipeline across all document formats."""

from __future__ import annotations

from pathlib import Path
import pytest

from studyos.core.config import AppConfig
from studyos.ingestion.service import IngestionService
from studyos.storage.db import Database
from tests.fixtures.synthetic.build_fixtures import generate_all_fixtures


class TestIngestionPipelineIntegration:
    def test_full_pipeline_multi_format_and_persistence(self, tmp_path):
        """
        Ingest all four synthetic document formats into a disk-backed SQLite database,
        verify trigger-based FTS5 indexing for format-specific keywords, and ensure
        data integrity persists across database reconnects.
        """
        fixtures = generate_all_fixtures(tmp_path / "fixtures")
        db_path = tmp_path / "studyos_integration.db"

        # 1. Initialize persistent database and migrate
        db = Database(db_path)
        db.migrate()

        config = AppConfig(data_dir=tmp_path, db_path=db_path)
        service = IngestionService(db=db, config=config)

        # 2. Create courses
        cs_course = db.create_course(
            name="Foundations of Computer Science",
            code="CS100",
            description="Introductory curriculum",
        )
        ds_course = db.create_course(
            name="Database Systems",
            code="DS200",
            description="Relational engines",
        )

        # 3. Ingest all 4 formats
        # PDF -> CS Course
        res_pdf = service.ingest_file(fixtures["pdf"], course_id=cs_course.id)
        assert res_pdf.status == "ready"
        assert res_pdf.element_count == 2
        assert res_pdf.chunk_count == 2

        # PPTX -> CS Course
        res_pptx = service.ingest_file(fixtures["pptx"], course_id=cs_course.id)
        assert res_pptx.status == "ready"
        assert res_pptx.element_count == 4
        assert res_pptx.chunk_count == 4

        # DOCX -> DS Course
        res_docx = service.ingest_file(fixtures["docx"], course_id=ds_course.id)
        assert res_docx.status == "ready"
        assert res_docx.element_count >= 2
        assert res_docx.chunk_count >= 2

        # TXT -> CS Course
        res_txt = service.ingest_file(fixtures["txt"], course_id=cs_course.id)
        assert res_txt.status == "ready"
        assert res_txt.element_count == 2
        assert res_txt.chunk_count == 2

        total_chunks = (
            res_pdf.chunk_count
            + res_pptx.chunk_count
            + res_docx.chunk_count
            + res_txt.chunk_count
        )
        assert db.count_fts_rows() == total_chunks

        # 4. Verify FTS5 keyword queries across each document format
        # PDF term
        pdf_matches = db.query_fts("Dijkstra", course_id=cs_course.id)
        assert len(pdf_matches) >= 1
        assert pdf_matches[0]["document_id"] == res_pdf.document_id
        assert "[Source: synthetic_lecture.pdf | Page: 2" in pdf_matches[0]["content"]

        # PPTX terms: regular text and table cells
        pptx_matches = db.query_fts("AVL", course_id=cs_course.id)
        assert len(pptx_matches) >= 1
        assert pptx_matches[0]["document_id"] == res_pptx.document_id
        assert "[Source: synthetic_slides.pptx | Slide: 2" in pptx_matches[0]["content"]

        table_matches = db.query_fts("QuickSort", course_id=cs_course.id)
        assert len(table_matches) >= 1
        assert table_matches[0]["document_id"] == res_pptx.document_id
        assert "[Source: synthetic_slides.pptx | Slide: 4" in table_matches[0]["content"]
        assert "| QuickSort |" in table_matches[0]["content"]

        # DOCX term
        docx_matches = db.query_fts("Atomicity", course_id=ds_course.id)
        assert len(docx_matches) >= 1
        assert docx_matches[0]["document_id"] == res_docx.document_id
        assert "[Source: synthetic_notes.docx | Section: 1" in docx_matches[0]["content"]

        # TXT term
        txt_matches = db.query_fts("TLB", course_id=cs_course.id)
        assert len(txt_matches) >= 1
        assert txt_matches[0]["document_id"] == res_txt.document_id
        assert "[Source: synthetic_doc.txt | Section: 2" in txt_matches[0]["content"]

        # 5. Course isolation test: Searching DOCX term in CS course must return 0 results
        isolated = db.query_fts("Atomicity", course_id=cs_course.id)
        assert len(isolated) == 0

        # 6. Document selection filter test
        filtered = db.query_fts("Dijkstra", document_ids=[res_pptx.document_id])
        assert len(filtered) == 0

        # 7. Persistence verification: close DB, create fresh connection from disk
        db.close()
        db_reopened = Database(db_path)

        assert db_reopened.count_fts_rows() == total_chunks
        courses = db_reopened.list_courses()
        assert len(courses) == 2

        cs_docs = db_reopened.list_documents(cs_course.id)
        assert len(cs_docs) == 3
        ds_docs = db_reopened.list_documents(ds_course.id)
        assert len(ds_docs) == 1

        reopened_matches = db_reopened.query_fts("Dijkstra")
        assert len(reopened_matches) >= 1
