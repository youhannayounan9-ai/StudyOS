"""AI Tutor view providing grounded study Q&A with validated citation cards."""

from __future__ import annotations

import streamlit as st

from studyos.inference.protocol import (
    ChatModelNotFoundError,
    ChatTimeoutError,
    InferenceError,
    OllamaChatConnectionError,
)
from studyos.retrieval.models import RetrievalMode
from studyos.tutoring.models import ResponseStyle, TutorRequest
from studyos.ui.components.citations import render_citations_list
from studyos.ui.state import UIServices, get_active_course


def render_tutor(services: UIServices) -> None:
    """Render the AI Tutor interface."""
    st.title("💬 Grounded AI Study Tutor")
    st.markdown("Ask conceptual questions strictly grounded in your ingested course notes with verified citations.")

    courses = services.db.list_courses()
    if not courses:
        st.warning("No courses available. Please create a course and upload materials in **Courses & Materials** first.")
        return

    # Configuration Bar
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        course_options = {c.id: f"{c.name} ({c.code})" for c in courses}
        active_c = get_active_course(services)
        current_idx = 0
        if active_c and active_c.id in course_options:
            current_idx = list(course_options.keys()).index(active_c.id)

        selected_course_id = st.selectbox(
            "Target Course:",
            options=list(course_options.keys()),
            format_func=lambda cid: course_options[cid],
            index=current_idx,
            key="tutor_course_select",
        )
        if selected_course_id != st.session_state.get("selected_course_id"):
            st.session_state["selected_course_id"] = selected_course_id
            st.rerun()

    with col2:
        # Document filtering within course
        course_docs = services.db.list_documents(selected_course_id)
        doc_options = {d.id: d.filename for d in course_docs}
        selected_doc_ids = st.multiselect(
            "Filter Documents (optional):",
            options=list(doc_options.keys()),
            format_func=lambda did: doc_options[did],
            default=[],
            key="tutor_doc_filter",
            help="Leave empty to search across all documents in this course.",
        )

    with col3:
        style_choice = st.selectbox(
            "Explanation Style:",
            options=["balanced", "concise", "detailed"],
            index=0,
            key="tutor_style_select",
        )

    st.markdown("---")

    # Chat history display
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    h_top1, h_top2 = st.columns([5, 1])
    with h_top2:
        if st.session_state["chat_history"]:
            if st.button("🗑️ Clear Chat", key="btn_clear_tutor_chat"):
                st.session_state["chat_history"] = []
                st.rerun()

    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                # Display modality badge
                mode = msg.get("retrieval_mode")
                if mode == RetrievalMode.HYBRID.value or mode == "hybrid":
                    st.markdown("<span class='badge badge-green'>Hybrid Search (Vector + BM25)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge badge-amber'>Degraded Lexical Search (BM25 only)</span>", unsafe_allow_html=True)

                if msg.get("unsupported"):
                    st.warning("⚠️ **Tutor Abstained:** Retrieved materials do not contain sufficient evidence to answer this question reliably.")

                st.markdown(msg["content"])

                # Citations
                citations = msg.get("citations", [])
                if citations:
                    render_citations_list(citations)

                warnings = msg.get("warnings", [])
                for w in warnings:
                    st.caption(f"⚠️ {w}")

    # Prompt Input
    user_query = st.chat_input("Ask a question about your course materials...")
    if user_query:
        cleaned_query = user_query.strip()
        if not cleaned_query:
            return

        if len(cleaned_query) > 2000:
            st.error("Query is too long (maximum 2,000 characters).")
            return

        # Pre-check Ollama Health
        if services.tutor_service is None or not services.ollama_ready:
            st.error(
                "❌ **Local AI Chat Service Unavailable**\n\n"
                f"Ollama is offline or unreachable at `{services.config.ollama_base_url}`.\n\n"
                "To start the tutor:\n"
                "```powershell\n"
                "ollama serve\n"
                f"ollama pull {services.config.tutor_model}\n"
                "```"
            )
            return

        # Record user message in history
        st.session_state["chat_history"].append({"role": "user", "content": cleaned_query})

        # Process response
        req = TutorRequest(
            query=cleaned_query,
            course_id=selected_course_id,
            document_ids=selected_doc_ids if selected_doc_ids else None,
            style=ResponseStyle(style_choice),
        )

        with st.spinner("Searching course materials and generating grounded explanation..."):
            try:
                response = services.tutor_service.answer(req)
                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "citations": response.citations,
                        "unsupported": response.unsupported,
                        "retrieval_mode": response.retrieval_mode.value,
                        "warnings": response.warnings,
                    }
                )
                st.rerun()
            except OllamaChatConnectionError as e:
                st.error(f"❌ Connection Error: Could not connect to Ollama at {e.base_url}. Ensure 'ollama serve' is running.")
            except ChatModelNotFoundError as e:
                st.error(f"❌ Model Not Found: Ollama model '{e.model_name}' is not installed. Run 'ollama pull {e.model_name}'.")
            except ChatTimeoutError as e:
                st.error(f"❌ Inference Timeout: The local AI model took too long to reply ({e.timeout_seconds}s).")
            except InferenceError as e:
                st.error(f"❌ Inference Error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {type(e).__name__}: {e}")

