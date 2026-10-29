"""Learning Progress view: concept catalog, Laplace mastery, and spaced repetition."""

from __future__ import annotations

import streamlit as st

from studyos.learner.models import (
    ConceptReviewStatus,
    MasteryCategory,
    ReviewRating,
    ReviewStatus,
)
from studyos.ui.state import UIServices, format_mastery_badge, get_active_course


def render_progress(services: UIServices) -> None:
    """Render Learning Progress and Knowledge Graph overview."""
    st.title("📈 Learning Progress & Mastery")
    st.markdown("Track knowledge concept mastery ($M=(k+1)/(n+2)$), review schedules, and discover concepts from course texts.")

    courses = services.db.list_courses()
    if not courses:
        st.warning("No courses available. Please create a course in **Courses & Materials** first.")
        return

    active_course = get_active_course(services)
    course_options = {c.id: f"{c.name} ({c.code})" for c in courses}
    current_idx = 0
    if active_course and active_course.id in course_options:
        current_idx = list(course_options.keys()).index(active_course.id)

    selected_course_id = st.selectbox(
        "Select Course for Progress Tracking:",
        options=list(course_options.keys()),
        format_func=lambda cid: course_options[cid],
        index=current_idx,
        key="prog_course_select",
    )
    if selected_course_id != st.session_state.get("selected_course_id"):
        st.session_state["selected_course_id"] = selected_course_id
        st.rerun()

    tab_overview, tab_concepts, tab_reviews, tab_discover = st.tabs(
        ["📊 Progress Summary", "📚 Concept Catalog", "⏰ Due Reviews", "🔍 Discover Concepts"]
    )

    summary = services.learner_service.get_progress_summary(course_id=selected_course_id)

    # ---------------------------------------------------------------------------
    # Tab 1: Progress Summary
    # ---------------------------------------------------------------------------
    with tab_overview:
        st.subheader("Mastery Distribution")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.markdown(f"<span class='badge badge-green'>Strong Evidence</span><br><h3>{summary.strong_count}</h3>", unsafe_allow_html=True)
            st.caption("Score >= 0.70")
        with b2:
            st.markdown(f"<span class='badge badge-blue'>Moderate</span><br><h3>{summary.moderate_count}</h3>", unsafe_allow_html=True)
            st.caption("0.40 <= Score < 0.70")
        with b3:
            st.markdown(f"<span class='badge badge-red'>Weak Evidence</span><br><h3>{summary.weak_count}</h3>", unsafe_allow_html=True)
            st.caption("Score < 0.40")
        with b4:
            st.markdown(f"<span class='badge badge-amber'>Needs Evidence</span><br><h3>{summary.needs_evidence_count}</h3>", unsafe_allow_html=True)
            st.caption("< 3 attempts")
        with b5:
            st.markdown(f"<span class='badge badge-gray'>Unassessed</span><br><h3>{summary.unassessed_count}</h3>", unsafe_allow_html=True)
            st.caption("0 attempts")

        st.markdown("---")
        st.subheader("Assessment Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Concepts", summary.total_concepts)
        with m2:
            st.metric("Graded Attempts", summary.total_graded_attempts)
        with m3:
            acc_str = f"{summary.accuracy_rate:.1f}%" if summary.accuracy_rate is not None else "N/A"
            st.metric("Overall Accuracy", acc_str)

    # ---------------------------------------------------------------------------
    # Tab 2: Concept Catalog
    # ---------------------------------------------------------------------------
    with tab_concepts:
        st.subheader("Course Knowledge Concepts")
        status_filter = st.selectbox(
            "Filter by Review Status:",
            options=["all", "accepted", "proposed", "rejected"],
            index=0,
            key="concept_filter_status",
        )

        concepts = services.learner_service.repository.list_concepts(course_id=selected_course_id)
        if status_filter != "all":
            concepts = [c for c in concepts if c.review_status.value == status_filter]

        if not concepts:
            st.info("No concepts match the selected filter.")
        else:
            for cpt in concepts:
                mastery = services.learner_service.repository.get_concept_mastery("learner_default", cpt.id)
                cat = mastery.mastery_category if mastery else MasteryCategory.UNASSESSED
                label, bg, fg = format_mastery_badge(cat)
                score_str = f"{mastery.mastery_score:.2f}" if mastery and mastery.observation_count > 0 else "0.00"
                obs_str = f"{mastery.correct_count:.1f}/{mastery.observation_count}" if mastery else "0/0"

                with st.expander(f"{cpt.name} — Mastery: {score_str} ({label})", expanded=False):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        if cpt.description:
                            st.markdown(f"**Description:** {cpt.description}")
                        st.markdown(f"**Concept ID:** `{cpt.id}`")
                        st.markdown(
                            f"**Status:** <span class='badge badge-purple'>{cpt.review_status.value.upper()}</span> "
                            f"**Provenance:** `{cpt.provenance.value}`",
                            unsafe_allow_html=True,
                        )
                        if mastery:
                            st.markdown(f"**Success Credits / Observations:** `{obs_str}`")
                            st.markdown(f"**Uncertainty (σ):** `{mastery.uncertainty:.4f}` | **Stability:** `{mastery.stability:.1f}d`")
                            if mastery.next_review_due:
                                st.markdown(f"**Next Review Due:** `{mastery.next_review_due[:16]}`")
                    with c2:
                        if cpt.review_status == ConceptReviewStatus.PROPOSED:
                            if st.button("Accept Concept", key=f"acc_cpt_{cpt.id}"):
                                services.learner_service.repository.update_concept_status(cpt.id, ConceptReviewStatus.ACCEPTED)
                                st.success(f"Accepted {cpt.name}!")
                                st.rerun()
                            if st.button("Reject Concept", key=f"rej_cpt_{cpt.id}"):
                                services.learner_service.repository.update_concept_status(cpt.id, ConceptReviewStatus.REJECTED)
                                st.rerun()

    # ---------------------------------------------------------------------------
    # Tab 3: Due Reviews (Spaced Repetition Queue)
    # ---------------------------------------------------------------------------
    with tab_reviews:
        st.subheader("Spaced Repetition Review Queue")
        tasks = services.learner_service.list_pending_reviews(
            course_id=selected_course_id,
            only_due=False,
        )

        if not tasks:
            st.info("No scheduled review tasks for this course. You are completely caught up!")
        else:
            for t in tasks:
                status_pill = "<span class='badge badge-red'>OVERDUE</span>" if t.is_overdue else (
                    "<span class='badge badge-amber'>DUE TODAY</span>" if t.is_due else "<span class='badge badge-gray'>Scheduled</span>"
                )
                c_name = t.concept_name or t.concept_id

                with st.expander(f"{c_name} — Due: {t.due_date[:10]}", expanded=t.is_due or t.is_overdue):
                    st.markdown(f"**Status:** {status_pill}", unsafe_allow_html=True)
                    st.markdown(f"**Task ID:** `{t.id}`")
                    st.markdown(f"**Scheduled Date:** `{t.due_date[:16]}`")

                    st.markdown("**Record Manual Review Outcome:**")
                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        if st.button("🔴 Again", key=f"rev_again_{t.id}", use_container_width=True):
                            services.learner_service.record_review_outcome(t.id, ReviewRating.AGAIN)
                            st.rerun()
                    with r2:
                        if st.button("🟠 Hard", key=f"rev_hard_{t.id}", use_container_width=True):
                            services.learner_service.record_review_outcome(t.id, ReviewRating.HARD)
                            st.rerun()
                    with r3:
                        if st.button("🟢 Good", key=f"rev_good_{t.id}", use_container_width=True):
                            services.learner_service.record_review_outcome(t.id, ReviewRating.GOOD)
                            st.rerun()
                    with r4:
                        if st.button("🔵 Easy", key=f"rev_easy_{t.id}", use_container_width=True):
                            services.learner_service.record_review_outcome(t.id, ReviewRating.EASY)
                            st.rerun()

    # ---------------------------------------------------------------------------
    # Tab 4: Direct Concept Discovery
    # ---------------------------------------------------------------------------
    with tab_discover:
        st.subheader("🔍 Automated Concept Discovery")
        st.markdown(
            "Scan course materials directly with the local LLM to extract key concepts and relationship proposals "
            "without having to take exams first."
        )

        if not services.ollama_ready:
            st.error("Local AI inference server (Ollama) is offline. Start Ollama to run concept discovery.")
        else:
            docs = services.db.list_documents(selected_course_id)
            doc_opts = {d.id: d.filename for d in docs}
            selected_disc_doc = st.selectbox(
                "Filter to specific document (optional):",
                options=["all"] + list(doc_opts.keys()),
                format_func=lambda did: "All Course Documents" if did == "all" else doc_opts[did],
                index=0,
                key="disc_doc_select",
            )
            max_chunks_scan = st.slider("Max chunks to inspect:", min_value=5, max_value=30, value=15)
            auto_accept = st.checkbox("Automatically accept discovered concepts", value=False)

            if st.button("🚀 Discover Concepts", type="primary", key="btn_run_discovery"):
                doc_id_param = None if selected_disc_doc == "all" else selected_disc_doc
                with st.spinner("Extracting conceptual topics from course chunks..."):
                    try:
                        disc_result = services.learner_service.discover_concepts(
                            course_id=selected_course_id,
                            document_id=doc_id_param,
                            max_chunks=max_chunks_scan,
                            auto_accept=auto_accept,
                        )
                        st.success(f"Discovered **{len(disc_result.discovered_concepts)}** concept(s) across {disc_result.chunks_scanned} chunks!")
                        for dc in disc_result.discovered_concepts:
                            st.markdown(f"- **{dc.name}**: {dc.description or 'No description'}")
                    except Exception as e:
                        st.error(f"Discovery error: {e}")

