"""Adaptive Study Planner view: transparent, deterministic, time-budgeted schedules."""

from __future__ import annotations

import streamlit as st

from studyos.learner.planner import PlanActionType
from studyos.ui.state import UIServices, format_mastery_badge, get_active_course


def render_planner(services: UIServices) -> None:
    """Render Adaptive Study Planner interface."""
    st.title("🧭 Adaptive Study Planner")
    st.markdown(
        "Generate a deterministic, prioritized daily study plan based on your knowledge graph, "
        "forgetting curves, and time constraints. **100% offline, zero LLM calls.**"
    )

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
        "Course:",
        options=list(course_options.keys()),
        format_func=lambda cid: course_options[cid],
        index=current_idx,
        key="plan_course_select",
    )
    if selected_course_id != st.session_state.get("selected_course_id"):
        st.session_state["selected_course_id"] = selected_course_id
        st.rerun()

    p_col1, p_col2, p_col3 = st.columns([2, 2, 1])

    with p_col1:
        # Target Concept (optional)
        concepts = services.learner_service.repository.list_concepts(course_id=selected_course_id)
        cpt_options = {"none": "No Target (All Active Course Concepts)"}
        for c in concepts:
            cpt_options[c.id] = c.name

        target_cpt_choice = st.selectbox(
            "Target Goal Concept (optional):",
            options=list(cpt_options.keys()),
            format_func=lambda k: cpt_options[k],
            index=0,
            key="plan_target_cpt",
            help="If selected, prioritizes mastering this concept and its prerequisite dependency chain.",
        )
        target_concept_id = None if target_cpt_choice == "none" else target_cpt_choice

    with p_col2:
        # Time Budget
        budget_options = {
            "unlimited": "Unlimited Time Budget",
            "20": "20 minutes (Quick Sprint)",
            "30": "30 minutes (Standard Session)",
            "45": "45 minutes (Deep Session)",
            "60": "60 minutes (Comprehensive Study)",
        }
        budget_choice = st.selectbox(
            "Time Budget:",
            options=list(budget_options.keys()),
            format_func=lambda b: budget_options[b],
            index=2,
            key="plan_time_budget",
        )
        time_budget_val = None if budget_choice == "unlimited" else int(budget_choice)

    with p_col3:
        max_items_val = st.number_input("Max Items:", min_value=1, max_value=25, value=10)

    st.markdown("---")

    if st.button("🚀 Generate Today's Study Plan", type="primary", key="btn_generate_plan"):
        with st.spinner("Analyzing knowledge graph, mastery state, and review urgency..."):
            plan = services.learner_service.generate_adaptive_study_plan(
                course_id=selected_course_id,
                target_concept_id=target_concept_id,
                time_budget_minutes=time_budget_val,
                max_items=int(max_items_val),
            )
            st.session_state["last_study_plan"] = plan

    plan = st.session_state.get("last_study_plan")
    if plan is None:
        st.info("Click 'Generate Today's Study Plan' to build your optimized study schedule.")
    else:
        st.subheader("📋 Recommended Study Agenda")

        b_str = f"{plan.time_budget_minutes} min" if plan.time_budget_minutes else "Unlimited"
        st.caption(f"Allocated: **{plan.allocated_minutes} min** / Budget: **{b_str}** | Actions: **{plan.total_actions}**")

        if not plan.items:
            st.success("🎉 You are completely caught up! No high-priority study actions required right now.")
        else:
            for idx, item in enumerate(plan.items, 1):
                badge_type = "badge-blue"
                if "overdue" in item.action_type.value:
                    badge_type = "badge-red"
                elif "prerequisite" in item.action_type.value:
                    badge_type = "badge-amber"
                elif "due" in item.action_type.value:
                    badge_type = "badge-purple"
                elif "target" in item.action_type.value:
                    badge_type = "badge-green"

                with st.expander(f"{idx}. [{item.action_title}] — {item.concept_name} ({item.estimated_minutes} min)", expanded=True):
                    i1, i2 = st.columns([3, 1])
                    with i1:
                        st.markdown(
                            f"<span class='badge {badge_type}'>Priority {item.priority}: {item.action_type.value.upper()}</span> "
                            f"<span class='badge badge-gray'>{item.estimated_minutes} minutes</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Diagnostic Rationale:** {item.reason}")

                        if item.unmet_prerequisites:
                            st.warning(
                                f"⚠️ **Unmet Prerequisites Detected:** Master prerequisite concepts before advancing: "
                                f"`{', '.join(item.unmet_prerequisites)}`"
                            )
                    with i2:
                        label, bg, fg = format_mastery_badge(item.mastery_category)
                        st.markdown(f"**Mastery:** `{item.mastery_score:.2f}` ({label})")
                        st.markdown(f"**Observations:** `{item.observation_count}`")
