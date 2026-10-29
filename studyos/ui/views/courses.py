"""Courses and document materials management, upload, and ingestion view."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import streamlit as st

from studyos.ingestion.models import (
    DuplicateDocumentError,
    EmptyFileError,
    FileTooLargeError,
    IngestionError,
)
from studyos.ui.state import UIServices, clean_temp_file, get_active_course


def render_courses(services: UIServices) -> None:
    """Render Course and Document Management view."""
    st.title("📁 Courses & Study Materials")
    st.markdown("Manage courses, upload course documents (.pdf, .docx, .pptx, .txt), and inspect ingested chunks.")

    active_course = get_active_course(services)
    courses = services.db.list_courses()

    tab_docs, tab_create = st.tabs(["📚 Ingest Materials & View Documents", "➕ Create New Course"])

    # ---------------------------------------------------------------------------
    # Tab 1: Ingest Materials & View Documents
    # ---------------------------------------------------------------------------
    with tab_docs:
        if not courses:
            st.info("No courses created yet. Please create a course first in the 'Create New Course' tab.")
            return

        # Course Selector for this view
        course_names = {c.id: f"{c.name} ({c.code})" for c in courses}
        current_idx = 0
        if active_course and active_course.id in course_names:
            current_idx = list(course_names.keys()).index(active_course.id)

        selected_id = st.selectbox(
            "Select Course for Documents:",
            options=list(course_names.keys()),
            format_func=lambda cid: course_names[cid],
            index=current_idx,
            key="courses_view_course_select",
        )
        if selected_id != st.session_state.get("selected_course_id"):
            st.session_state["selected_course_id"] = selected_id
            st.rerun()

        selected_course = services.db.get_course(selected_id)
        if not selected_course:
            return

        st.markdown("---")

        # Ingestion Uploader
        st.subheader(f"📤 Ingest Document into {selected_course.name}")
        u1, u2 = st.columns([3, 1])
        with u1:
            uploaded_file = st.file_uploader(
                "Upload a lecture slide, reading, or syllabus:",
                type=["pdf", "docx", "pptx", "txt"],
                key="course_doc_uploader",
                help="Supports PDF, Word (.docx), PowerPoint (.pptx), and plain text (.txt) up to 50MB.",
            )
        with u2:
            st.write("")
            st.write("")
            force_reindex = st.checkbox(
                "Force Re-index",
                value=False,
                help="If checked, overwrites existing document if already present in this course.",
            )

        if uploaded_file is not None:
            if st.button("🚀 Process & Ingest Document", type="primary", key="btn_ingest_doc"):
                suffix = Path(uploaded_file.name).suffix
                temp_file = None
                try:
                    # Write in-memory bytes to safe temporary file with correct extension
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                        temp_file = tf.name
                        tf.write(uploaded_file.getvalue())

                    with st.spinner(f"Parsing, chunking, and indexing '{uploaded_file.name}'..."):
                        result = services.ingestion_service.ingest_file(
                            file_path=temp_file,
                            course_id=selected_course.id,
                            force_reindex=force_reindex,
                        )

                    if result.status == "duplicate":
                        st.warning(
                            f"⚠️ **Duplicate Document Detected**: This file has already been ingested into this course (Document ID: `{result.document_id}`).\n\n"
                            "Tip: Check **Force Re-index** above if you wish to re-parse and replace it."
                        )
                    else:
                        st.success(
                            f"✅ Successfully ingested **{uploaded_file.name}**!\n\n"
                            f"- Document ID: `{result.document_id}`\n"
                            f"- Elements extracted: `{result.element_count}`\n"
                            f"- Chunks indexed: `{result.chunk_count}`"
                        )

                    if result.warnings and result.status != "duplicate":
                        for w in result.warnings:
                            st.warning(f"⚠️ {w}")

                except DuplicateDocumentError as e:
                    st.error(f"❌ Duplicate Document: {e}\n\nTip: Check 'Force Re-index' above to replace it.")
                except EmptyFileError as e:
                    st.error(f"❌ Empty File: {e}")
                except FileTooLargeError as e:
                    st.error(f"❌ File Too Large: {e}")
                except IngestionError as e:
                    st.error(f"❌ Ingestion Error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error during ingestion: {type(e).__name__}: {e}")
                finally:
                    clean_temp_file(temp_file)

        st.markdown("---")

        # Document Library for Selected Course
        st.subheader(f"📑 Documents in {selected_course.name}")
        docs = services.db.list_documents(selected_course.id)

        if not docs:
            st.info(f"No documents have been ingested for {selected_course.name} yet.")
        else:
            for doc in docs:
                with st.expander(f"📄 {doc.filename} — {doc.chunk_count} chunks ({doc.file_type.upper()})", expanded=False):
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.markdown(f"**Document ID:** `{doc.id}`")
                        st.markdown(f"**Type:** `{doc.file_type.upper()}`")
                    with d2:
                        size_kb = doc.file_size_bytes / 1024
                        st.markdown(f"**Size:** `{size_kb:.1f} KB`")
                        st.markdown(f"**Total Pages/Slides:** `{doc.total_pages_or_slides}`")
                    with d3:
                        st.markdown(f"**Status:** `{doc.status.upper()}`")
                        st.markdown(f"**Created:** `{doc.created_at[:16] if doc.created_at else ''}`")

                    # Delete action
                    if st.button("🗑️ Delete Document", key=f"del_doc_{doc.id}"):
                        deleted = services.db.delete_document(doc.id)
                        if deleted:
                            st.success(f"Deleted {doc.filename}.")
                            st.rerun()

    # ---------------------------------------------------------------------------
    # Tab 2: Create New Course
    # ---------------------------------------------------------------------------
    with tab_create:
        st.subheader("➕ Create a New Course")
        with st.form("create_course_form"):
            c_name = st.text_input("Course Name *", placeholder="e.g. Data Structures & Algorithms")
            c_code = st.text_input("Course Code *", placeholder="e.g. CS201")
            c_desc = st.text_area("Description (optional)", placeholder="Core concepts, textbook readings, and labs.")

            submitted = st.form_submit_button("Create Course", type="primary")
            if submitted:
                if not c_name.strip() or not c_code.strip():
                    st.error("Both Course Name and Course Code are required.")
                else:
                    existing = services.db.get_course_by_code(c_code.strip())
                    if existing:
                        st.error(f"Course with code '{c_code.strip()}' already exists ({existing.name}).")
                    else:
                        new_course = services.db.create_course(
                            name=c_name.strip(),
                            code=c_code.strip(),
                            description=c_desc.strip() if c_desc else None,
                        )
                        st.success(f"Course **{new_course.name}** (`{new_course.code}`) created successfully!")
                        st.session_state["selected_course_id"] = new_course.id
                        st.rerun()
