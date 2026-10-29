"""Dashboard view displaying system stats, learning metrics, and quick actions."""

from __future__ import annotations

from datetime import datetime, timezone
import streamlit as st

from studyos.learner.models import MasteryCategory
from studyos.ui.state import UIServices, get_active_course


def render_dashboard(services: UIServices) -> None:
    """Render the central StudyOS Dashboard."""
    st.title("📊 StudyOS Dashboard")
    active_course = get_active_course(services)

    if active_course:
        st.caption(f"Active Course: **{active_course.name}** (`{active_course.code}`)")
    else:
        st.caption("Active Course: **All Courses** (Select a course in the sidebar to narrow focus)")

    # 1. High-Level Metrics
    with services.db.get_connection() as conn:
        course_count = conn.execute("SELECT COUNT(*) as c FROM courses;").fetchone()["c"]
        doc_count = conn.execute("SELECT COUNT(*) as c FROM documents;").fetchone()["c"]
        chunk_count = conn.execute("SELECT COUNT(*) as c FROM chunks;").fetchone()["c"]
        exam_count = conn.execute("SELECT COUNT(*) as c FROM exams;").fetchone()["c"]
        attempt_count = conn.execute("SELECT COUNT(*) as c FROM exam_attempts WHERE status = 'completed';").fetchone()["c"]
        concept_count = conn.execute("SELECT COUNT(*) as c FROM concepts;").fetchone()["c"]

    # Pending / Overdue Reviews
    course_filter = active_course.id if active_course else None
    progress_summary = services.learner_service.get_progress_summary(course_id=course_filter)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{course_count}</div>
                <div class="metric-label">Registered Courses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{doc_count} <span style="font-size:1rem;color:#6b7280;">({chunk_count} chunks)</span></div>
                <div class="metric-label">Ingested Documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value">{exam_count} <span style="font-size:1rem;color:#6b7280;">({attempt_count} attempts)</span></div>
                <div class="metric-label">Practice Exams</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        due_style = "color:#dc2626;" if progress_summary.overdue_reviews_count > 0 else ""
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value" style="{due_style}">{progress_summary.pending_reviews_count}</div>
                <div class="metric-label">Due Reviews ({progress_summary.overdue_reviews_count} overdue)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Quick Actions
    st.subheader("⚡ Quick Launch")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("💬 Ask AI Tutor", use_container_width=True, key="dash_btn_tutor"):
            st.session_state["active_nav"] = "AI Tutor"
            st.rerun()
    with q2:
        if st.button("📝 Take Practice Exam", use_container_width=True, key="dash_btn_exam"):
            st.session_state["active_nav"] = "Practice Exams"
            st.rerun()
    with q3:
        if st.button("🧭 Today's Study Plan", use_container_width=True, key="dash_btn_plan"):
            st.session_state["active_nav"] = "Study Planner"
            st.rerun()
    with q4:
        if st.button("📁 Upload Materials", use_container_width=True, key="dash_btn_upload"):
            st.session_state["active_nav"] = "Courses & Materials"
            st.rerun()

    st.markdown("---")

    # 3. Learning Progress Overview
    st.subheader("📈 Learning Progress & Mastery")
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown(f"**Knowledge Concepts Tracked:** `{progress_summary.total_concepts}`")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.markdown(f"<span class='badge badge-green'>Strong</span><br><h3 style='margin:0.2rem 0;'>{progress_summary.strong_count}</h3>", unsafe_allow_html=True)
        with b2:
            st.markdown(f"<span class='badge badge-blue'>Moderate</span><br><h3 style='margin:0.2rem 0;'>{progress_summary.moderate_count}</h3>", unsafe_allow_html=True)
        with b3:
            st.markdown(f"<span class='badge badge-red'>Weak</span><br><h3 style='margin:0.2rem 0;'>{progress_summary.weak_count}</h3>", unsafe_allow_html=True)
        with b4:
            st.markdown(f"<span class='badge badge-amber'>Needs Evid.</span><br><h3 style='margin:0.2rem 0;'>{progress_summary.needs_evidence_count}</h3>", unsafe_allow_html=True)
        with b5:
            st.markdown(f"<span class='badge badge-gray'>Unassessed</span><br><h3 style='margin:0.2rem 0;'>{progress_summary.unassessed_count}</h3>", unsafe_allow_html=True)

        acc = f"{progress_summary.accuracy_rate:.1f}%" if progress_summary.accuracy_rate is not None else "No observations"
        st.caption(f"Overall Accuracy: **{acc}** across {progress_summary.total_graded_attempts} graded observation(s).")

    with c2:
        st.markdown("**Recent Activity**")
        events = services.learner_service.repository.list_events(
            learner_id="learner_default",
            course_id=course_filter,
            limit=5,
        )
        if not events:
            st.caption("No study events recorded yet. Take an exam or ask questions to build progress!")
        else:
            for ev in events:
                time_str = ev.created_at[:16] if ev.created_at else ""
                outcome_icon = "✅" if ev.is_correct is True else ("❌" if ev.is_correct is False else "📝")
                st.markdown(f"{outcome_icon} **{ev.event_type.replace('_', ' ').capitalize()}** — `{time_str}`")

