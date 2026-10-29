"""Practice Exams view: grounded exam generation, strictly isolated taking, and grading."""

from __future__ import annotations

import streamlit as st

from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    ExamRequest,
    GradingStatus,
    QuestionType,
    SubmitAnswersRequest,
)
from studyos.ui.components.citations import render_question_sources
from studyos.ui.state import UIServices, get_active_course


def render_exams(services: UIServices) -> None:
    """Render Practice Exams interface."""
    st.title("📝 Practice Exams & Assessments")
    st.markdown("Generate grounded self-assessment exams, take practice tests, and receive immediate verified feedback.")

    courses = services.db.list_courses()
    if not courses:
        st.warning("No courses available. Please create a course and upload materials in **Courses & Materials** first.")
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
        key="exam_course_select",
    )
    if selected_course_id != st.session_state.get("selected_course_id"):
        st.session_state["selected_course_id"] = selected_course_id
        st.session_state["active_exam_public"] = None
        st.session_state["active_attempt_id"] = None
        st.rerun()

    tab_take, tab_generate, tab_results = st.tabs(["✍️ Take Exam", "⚙️ Generate New Exam", "📊 Latest Results"])

    # ---------------------------------------------------------------------------
    # Tab 1: Take Exam (Strict Answer-Key Isolation)
    # ---------------------------------------------------------------------------
    with tab_take:
        public_exam = st.session_state.get("active_exam_public")
        attempt_id = st.session_state.get("active_attempt_id")

        if public_exam is None or attempt_id is None:
            st.subheader("Available Exams for this Course")
            exams = services.assessment_service.list_exams(course_id=selected_course_id)
            if not exams:
                st.info("No practice exams generated for this course yet. Use the 'Generate New Exam' tab to create one!")
            else:
                for ex in exams:
                    with st.container():
                        st.markdown(f"### 📋 {ex.title}")
                        e1, e2, e3 = st.columns([2, 2, 1])
                        with e1:
                            st.caption(f"Difficulty: **{ex.difficulty.value.upper()}** | Questions: **{ex.question_count}**")
                        with e2:
                            topic_str = ex.topic or "Course-wide review"
                            st.caption(f"Topic: **{topic_str}**")
                        with e3:
                            if st.button("Start Exam", key=f"start_exam_{ex.id}", type="primary"):
                                # Strictly load PublicExam without answer keys
                                pub_ex = services.assessment_service.get_exam_for_taking(ex.id)
                                att = services.assessment_service.start_attempt(ex.id)
                                st.session_state["active_exam_public"] = pub_ex
                                st.session_state["active_attempt_id"] = att.id
                                st.session_state["exam_answers"] = {}
                                st.rerun()
                        st.markdown("---")
        else:
            # Active Exam Session using ONLY PublicExam
            st.subheader(f"Exam: {public_exam.title}")
            st.caption(f"Difficulty: {public_exam.difficulty.value.capitalize()} | Questions: {public_exam.question_count}")

            if "exam_answers" not in st.session_state:
                st.session_state["exam_answers"] = {}

            with st.form("exam_taking_form"):
                for idx, q in enumerate(public_exam.questions, 1):
                    st.markdown(f"#### Question {idx} ({q.points} pt{'s' if q.points != 1 else ''})")
                    st.markdown(f"**{q.question_text}**")

                    if q.question_type == QuestionType.MCQ and q.options:
                        # Render Multiple Choice
                        option_texts = [f"{opt.id}: {opt.text}" for opt in q.options]
                        choice = st.radio(
                            label=f"Select an answer for Q{idx}:",
                            options=option_texts,
                            index=None,
                            key=f"q_{q.id}_radio",
                            label_visibility="collapsed",
                        )
                        if choice:
                            opt_id = choice.split(":")[0].strip()
                            st.session_state["exam_answers"][q.id] = opt_id

                    elif q.question_type == QuestionType.TRUE_FALSE:
                        # Render True / False
                        choice = st.radio(
                            label=f"Select True or False for Q{idx}:",
                            options=["True", "False"],
                            index=None,
                            key=f"q_{q.id}_tf",
                            label_visibility="collapsed",
                        )
                        if choice:
                            st.session_state["exam_answers"][q.id] = choice

                    elif q.question_type == QuestionType.SHORT_ANSWER:
                        # Render Free text
                        ans_text = st.text_area(
                            label=f"Write your response for Q{idx}:",
                            key=f"q_{q.id}_text",
                            placeholder="Type your explanation...",
                            label_visibility="collapsed",
                        )
                        if ans_text:
                            st.session_state["exam_answers"][q.id] = ans_text

                    st.markdown("---")

                sub_col1, sub_col2 = st.columns([1, 4])
                with sub_col1:
                    submitted = st.form_submit_button("Submit Exam", type="primary")

            if submitted:
                with st.spinner("Submitting and grading your answers against verified course ground truth..."):
                    req = SubmitAnswersRequest(answers=st.session_state["exam_answers"])
                    result = services.assessment_service.submit_attempt(
                        attempt_id=attempt_id,
                        request=req,
                    )
                    st.session_state["last_exam_result"] = result
                    st.session_state["active_exam_public"] = None
                    st.session_state["active_attempt_id"] = None
                    st.session_state["exam_answers"] = {}
                    st.rerun()

            if st.button("Cancel Attempt", key="btn_cancel_exam"):
                st.session_state["active_exam_public"] = None
                st.session_state["active_attempt_id"] = None
                st.session_state["exam_answers"] = {}
                st.rerun()

    # ---------------------------------------------------------------------------
    # Tab 2: Generate New Exam
    # ---------------------------------------------------------------------------
    with tab_generate:
        st.subheader("⚙️ Generate Grounded Practice Exam")
        if not services.ollama_ready:
            st.error("Local AI inference server (Ollama) is offline. Start Ollama to generate new practice exams.")
        else:
            with st.form("exam_generation_form"):
                topic_focus = st.text_input("Topic or Chapter Focus (optional)", placeholder="e.g. Graph Algorithms or Dynamic Programming")
                q_count = st.slider("Number of Questions:", min_value=1, max_value=20, value=5)
                diff_choice = st.selectbox("Target Difficulty:", options=["medium", "easy", "hard", "mixed"], index=0)
                selected_types = st.multiselect(
                    "Question Types:",
                    options=["mcq", "true_false", "short_answer"],
                    default=["mcq", "true_false"],
                    format_func=lambda t: {"mcq": "Multiple Choice", "true_false": "True / False", "short_answer": "Short Answer"}[t],
                )

                gen_submitted = st.form_submit_button("Generate Practice Exam", type="primary")
                if gen_submitted:
                    if not selected_types:
                        st.error("Please select at least one question type.")
                    else:
                        types = [QuestionType(t) for t in selected_types]
                        req = ExamRequest(
                            course_id=selected_course_id,
                            topic=topic_focus if topic_focus.strip() else None,
                            question_count=q_count,
                            difficulty=Difficulty(diff_choice),
                            question_types=types,
                        )
                        with st.spinner("Retrieving course chunks and generating grounded questions..."):
                            try:
                                new_exam = services.assessment_service.generate_exam(req)
                                st.success(f"Exam '{new_exam.title}' generated successfully with {len(new_exam.questions)} questions!")
                                # Automatically prepare for taking
                                pub_ex = services.assessment_service.get_exam_for_taking(new_exam.id)
                                att = services.assessment_service.start_attempt(new_exam.id)
                                st.session_state["active_exam_public"] = pub_ex
                                st.session_state["active_attempt_id"] = att.id
                                st.session_state["exam_answers"] = {}
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to generate exam: {e}")

    # ---------------------------------------------------------------------------
    # Tab 3: Latest Results
    # ---------------------------------------------------------------------------
    with tab_results:
        res = st.session_state.get("last_exam_result")
        if not res:
            st.info("No completed exam attempt in this session. Take an exam to see your results and feedback!")
        else:
            st.subheader(f"Grading Report: {res.title}")

            # Score Banner
            if res.status == AttemptStatus.COMPLETED and res.score is not None and res.percentage is not None:
                st.success(f"### Final Score: **{res.score:.1f} / {res.max_score:.1f} ({res.percentage:.1f}%)**")
            else:
                prov_pct = f"{res.provisional_percentage:.1f}%" if res.provisional_percentage is not None else "N/A"
                st.warning(
                    f"### Provisional Score: **{res.provisional_score:.1f} / {res.max_score:.1f} ({prov_pct})**\n\n"
                    "⚠️ One or more short-answer questions are awaiting manual review. The provisional score reflects only auto-graded questions."
                )

            st.markdown("---")
            for q_res in res.questions:
                status_badge = ""
                if q_res.status == GradingStatus.PENDING_MANUAL_REVIEW:
                    status_badge = "<span class='badge badge-amber'>Pending Manual Review</span>"
                elif q_res.is_correct is True:
                    status_badge = "<span class='badge badge-green'>Correct</span>"
                else:
                    status_badge = "<span class='badge badge-red'>Incorrect</span>"

                with st.expander(f"Q{q_res.question_index}: {q_res.question_text[:70]}... | {q_res.score or 0.0:.1f}/{q_res.max_score:.1f} pts", expanded=True):
                    st.markdown(f"**Question:** {q_res.question_text}")
                    st.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)
                    st.markdown(f"**Your Answer:** `{q_res.submitted_answer or 'No answer submitted'}`")
                    st.markdown(f"**Authoritative Correct Answer:** `{q_res.correct_answer}`")

                    if q_res.explanation:
                        st.info(f"**Explanation:** {q_res.explanation}")
                    if q_res.feedback:
                        st.caption(f"**Feedback:** {q_res.feedback}")

                    # Render Citations
                    if q_res.citations:
                        render_question_sources(q_res.citations)

