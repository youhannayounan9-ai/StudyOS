"""Citation formatting and interactive evidence cards for StudyOS."""

from __future__ import annotations

import streamlit as st
from studyos.assessment.models import QuestionSourceRef
from studyos.tutoring.models import ValidatedCitation


def render_citation_card(citation: ValidatedCitation, index: int = 1) -> None:
    """Render a single validated citation card with location and excerpt."""
    loc = f"{citation.location_type.capitalize()} {citation.location_index}"
    if citation.location_title:
        loc += f" ({citation.location_title})"

    with st.expander(f"📌 [{citation.marker}] {citation.filename} — {loc}", expanded=False):
        st.markdown(f"**Document:** `{citation.filename}` ({citation.file_type.upper()})")
        st.markdown(f"**Course:** {citation.course_name} (`{citation.course_id}`)")
        st.markdown(f"**Location:** {loc}")
        st.markdown(f"**Chunk ID:** `{citation.chunk_id}`")
        st.markdown("**Authoritative Source Excerpt:**")
        st.info(f"“{citation.excerpt}”")


def render_citations_list(citations: list[ValidatedCitation]) -> None:
    """Render a collection of verified evidence citations."""
    if not citations:
        return

    st.markdown("#### 📚 Verified Source Evidence")
    for i, c in enumerate(citations, 1):
        render_citation_card(c, index=i)


def render_question_sources(sources: list[QuestionSourceRef]) -> None:
    """Render source provenance for an assessment question."""
    if not sources:
        return

    with st.expander("🔍 View Source Evidence Provenance", expanded=False):
        for s in sources:
            loc = f"{s.location_type.capitalize() if s.location_type else 'Location'} {s.location_index or '?'}"
            if s.location_title:
                loc += f" ({s.location_title})"
            st.markdown(f"- **[{s.evidence_marker}]** `{s.filename or 'Document'}` — {loc}")
            if s.excerpt:
                st.caption(f"“{s.excerpt}”")
            if s.is_stale:
                st.warning("⚠️ Source chunk was updated or reindexed after this question was generated.")


def render_external_sources(sources: list[Any]) -> None:
    """Render external research sources table with outbound URLs."""
    if not sources:
        return

    st.markdown("#### 🌐 Verified External Sources")
    for s in sources:
        marker = getattr(s, "citation_marker", "[EXT]")
        title = getattr(s, "title", "External Source")
        url = getattr(s, "url", "#")
        pub = getattr(s, "publisher", "Web Source")
        snippet = getattr(s, "snippet", "")

        with st.expander(f"🔗 {marker} {title} ({pub})", expanded=False):
            st.markdown(f"**Title:** [{title}]({url})")
            st.markdown(f"**Publisher / Domain:** `{pub}`")
            st.markdown(f"**Direct URL:** [{url}]({url})")
            if snippet:
                st.markdown("**Snippet / Summary:**")
                st.caption(snippet)

