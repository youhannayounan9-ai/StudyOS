"""External Research view: strictly opt-in search queries with verified web citations."""

from __future__ import annotations

import streamlit as st

from studyos.research import ResearchRepository
from studyos.ui.components.citations import render_external_sources
from studyos.ui.state import UIServices, get_active_course


def render_research(services: UIServices) -> None:
    """Render External Research interface with mandatory explicit opt-in."""
    st.title("🔬 External Research")
    st.markdown(
        "Augment your study materials with public academic and reference sources (e.g., Wikipedia). "
        "Strictly separated from local course materials."
    )

    # Privacy boundary callout
    st.markdown(
        """
        <div class="studyos-callout">
            🔒 <strong>Strict Local Privacy Boundary:</strong> StudyOS is 100% local by default.
            Your uploaded lecture slides and personal documents are <em>never</em> sent across the network.
            External research performs read-only queries against public web APIs and requires your explicit opt-in for each query.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_query, tab_history = st.tabs(["🔍 New External Research", "📜 Past Research History"])

    # ---------------------------------------------------------------------------
    # Tab 1: New External Research
    # ---------------------------------------------------------------------------
    with tab_query:
        active_course = get_active_course(services)
        courses = services.db.list_courses()
        course_opts = {"none": "None (General Inquiry)"}
        for c in courses:
            course_opts[c.id] = f"{c.name} ({c.code})"

        curr_course_key = active_course.id if active_course and active_course.id in course_opts else "none"

        r1, r2 = st.columns([3, 1])
        with r1:
            res_query = st.text_input(
                "External Research Query:",
                placeholder="e.g. How does the Raft consensus algorithm handle leader election?",
                key="input_research_query",
            )
        with r2:
            course_choice = st.selectbox(
                "Associate with Course (optional):",
                options=list(course_opts.keys()),
                format_func=lambda cid: course_opts[cid],
                index=list(course_opts.keys()).index(curr_course_key),
                key="research_course_select",
            )

        c1, c2 = st.columns([1, 1])
        with c1:
            max_results = st.slider("Max Search Results:", min_value=1, max_value=10, value=5)
        with c2:
            fetch_pages = st.checkbox("Fetch and inspect full article text", value=False)

        # Mandatory Opt-in Checkbox
        opt_in = st.checkbox(
            "✅ I explicitly authorize StudyOS to perform an external network search query for this request.",
            value=False,
            key="research_explicit_opt_in",
            help="Required by StudyOS privacy policy before any external request is made.",
        )

        if st.button("🚀 Run External Research", type="primary", disabled=not opt_in, key="btn_run_research"):
            if not res_query or not res_query.strip():
                st.error("Please enter a research query.")
            elif not opt_in:
                st.error("External research requires your explicit opt-in confirmation.")
            elif services.research_service is None:
                st.error("Research service is not configured.")
            else:
                cid_param = None if course_choice == "none" else course_choice
                with st.spinner("Executing external web search and synthesizing cited summary..."):
                    try:
                        summary = services.research_service.research(
                            query=res_query.strip(),
                            course_id=cid_param,
                            max_results=max_results,
                            fetch_pages=fetch_pages,
                            explicit_opt_in=True,
                        )
                        st.session_state["last_research_result"] = summary
                        st.success("Research completed successfully!")
                    except Exception as e:
                        st.error(f"External research failed: {e}")

        # Display Last Research Result
        last_res = st.session_state.get("last_research_result")
        if last_res:
            st.markdown("---")
            st.subheader(f"Research Report: “{last_res.query}”")

            # Disclaimer banner
            st.info(f"ℹ️ {last_res.disclaimer}")

            # Synthesized summary
            st.markdown("#### 📝 Synthesized Summary")
            st.markdown(last_res.summary_text)

            # Verified External Sources
            if last_res.sources:
                render_external_sources(last_res.sources)

            if getattr(last_res, "invalid_citations_rejected", None):
                st.warning(
                    f"⚠️ Rejected hallucinated citations: {', '.join(last_res.invalid_citations_rejected)}"
                )

    # ---------------------------------------------------------------------------
    # Tab 2: Past Research History
    # ---------------------------------------------------------------------------
    with tab_history:
        st.subheader("Previous Research Queries")
        repo = ResearchRepository(services.db)
        records = repo.list_research_records(limit=25)

        if not records:
            st.info("No past external research queries found.")
        else:
            for rec in records:
                time_str = rec.created_at[:16] if rec.created_at else "Unknown"
                with st.expander(f"🔍 “{rec.query}” — {time_str} ({rec.status.value.upper()})", expanded=False):
                    st.markdown(f"**Query:** `{rec.query}`")
                    st.markdown(f"**Status:** `{rec.status.value}`")
                    st.markdown(f"**Executed At:** `{time_str}`")
                    if rec.summary_text:
                        st.markdown("**Saved Summary:**")
                        st.markdown(rec.summary_text)

