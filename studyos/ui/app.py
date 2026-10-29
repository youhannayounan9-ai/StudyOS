"""Main application entry point for StudyOS Streamlit Graphical Interface."""

from __future__ import annotations

import streamlit as st

from studyos.ui.components.health_banner import render_sidebar_health
from studyos.ui.state import get_services, init_session_state
from studyos.ui.theme import inject_custom_css
from studyos.ui.views.courses import render_courses
from studyos.ui.views.dashboard import render_dashboard
from studyos.ui.views.exams import render_exams
from studyos.ui.views.planner import render_planner
from studyos.ui.views.progress import render_progress
from studyos.ui.views.research import render_research
from studyos.ui.views.system_status import render_system_status
from studyos.ui.views.tutor import render_tutor

# Configure wide layout and page metadata
st.set_page_config(
    page_title="StudyOS — Local AI Study Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Navigation sections
NAV_SECTIONS = [
    "📊 Dashboard",
    "📁 Courses & Materials",
    "💬 AI Tutor",
    "📝 Practice Exams",
    "📈 Learning Progress",
    "🧭 Study Planner",
    "🔬 External Research",
    "⚙️ System & Index Status",
]

NAV_MAP = {
    "📊 Dashboard": "Dashboard",
    "📁 Courses & Materials": "Courses & Materials",
    "💬 AI Tutor": "AI Tutor",
    "📝 Practice Exams": "Practice Exams",
    "📈 Learning Progress": "Learning Progress",
    "🧭 Study Planner": "Study Planner",
    "🔬 External Research": "External Research",
    "⚙️ System & Index Status": "System & Index Status",
}

REVERSE_NAV_MAP = {v: k for k, v in NAV_MAP.items()}


def main() -> None:
    """StudyOS main execution flow."""
    # Inject CSS
    inject_custom_css()

    # Initialize session state
    init_session_state()

    # Load backend services
    services = get_services()

    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>📚 StudyOS</h2>
                <p>EVIDENCE-GROUNDED LOCAL AI STUDY ENGINE</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Global Course Selector
        courses = services.db.list_courses()
        course_options = {"all": "All Courses"}
        for c in courses:
            course_options[c.id] = f"{c.name} ({c.code})"

        current_course_id = st.session_state.get("selected_course_id") or "all"
        if current_course_id not in course_options:
            current_course_id = "all"

        course_keys = list(course_options.keys())
        selected_key = st.selectbox(
            "Active Course Context:",
            options=course_keys,
            index=course_keys.index(current_course_id),
            format_func=lambda k: course_options[k],
            key="sidebar_global_course_select",
        )

        if selected_key == "all":
            st.session_state["selected_course_id"] = None
        else:
            st.session_state["selected_course_id"] = selected_key

        st.markdown("---")

        # Navigation Menu
        active_nav = st.session_state.get("active_nav", "Dashboard")
        nav_label = REVERSE_NAV_MAP.get(active_nav, "📊 Dashboard")
        nav_idx = NAV_SECTIONS.index(nav_label) if nav_label in NAV_SECTIONS else 0

        selected_nav_label = st.radio(
            "Navigation:",
            options=NAV_SECTIONS,
            index=nav_idx,
            key="main_sidebar_nav",
        )

        selected_nav = NAV_MAP[selected_nav_label]
        if selected_nav != st.session_state.get("active_nav"):
            st.session_state["active_nav"] = selected_nav
            st.rerun()

        # Service Health Badges
        render_sidebar_health(services)

    # ---------------------------------------------------------------------------
    # Main View Routing
    # ---------------------------------------------------------------------------
    current_nav = st.session_state.get("active_nav", "Dashboard")

    if current_nav == "Dashboard":
        render_dashboard(services)
    elif current_nav == "Courses & Materials":
        render_courses(services)
    elif current_nav == "AI Tutor":
        render_tutor(services)
    elif current_nav == "Practice Exams":
        render_exams(services)
    elif current_nav == "Learning Progress":
        render_progress(services)
    elif current_nav == "Study Planner":
        render_planner(services)
    elif current_nav == "External Research":
        render_research(services)
    elif current_nav == "System & Index Status":
        render_system_status(services)
    else:
        render_dashboard(services)


if __name__ == "__main__":
    main()

