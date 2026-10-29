"""Health check status badges and diagnostics for local Ollama and SQLite."""

from __future__ import annotations

import streamlit as st
from studyos.ui.state import UIServices


def render_sidebar_health(services: UIServices) -> None:
    """Render compact service health indicators in the sidebar footer."""
    st.sidebar.markdown("---")
    st.sidebar.caption("SYSTEM HEALTH")

    # SQLite status (always OK if database opened)
    st.sidebar.markdown("🟢 **SQLite:** Connected (WAL mode)")

    # Ollama Chat
    if services.ollama_ready:
        st.sidebar.markdown(f"🟢 **Tutor AI:** Ready (`{services.config.tutor_model}`)")
    else:
        st.sidebar.markdown(f"🔴 **Tutor AI:** Offline (`{services.config.tutor_model}`)")

    # Ollama Embeddings
    if services.embedding_ready:
        st.sidebar.markdown(f"🟢 **Embeddings:** Ready (`{services.config.embedding_model}`)")
    else:
        st.sidebar.markdown(f"🟡 **Embeddings:** Offline (Lexical Fallback)")

    if not services.ollama_ready:
        with st.sidebar.expander("🛠️ Start Local AI", expanded=False):
            st.markdown(
                "Ollama is not responding at "
                f"`{services.config.ollama_base_url}`.\n\n"
                "**Quick Start:**\n"
                "```powershell\n"
                "ollama serve\n"
                f"ollama pull {services.config.tutor_model}\n"
                f"ollama pull {services.config.embedding_model}\n"
                "```\n"
                "Then refresh this page."
            )

