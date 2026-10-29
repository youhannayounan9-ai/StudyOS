"""System & Index Status view: diagnostics, health checks, and vector index sync."""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

from studyos.core.health import check_ollama_health
from studyos.ui.state import UIServices


def render_system_status(services: UIServices) -> None:
    """Render system diagnostics, configuration, and index health view."""
    st.title("⚙️ System & Index Status")
    st.markdown("Inspect local storage configurations, Ollama AI model availability, and ChromaDB vector synchronization.")

    s_tab_health, s_tab_index, s_tab_config = st.tabs(["🩺 AI & Database Health", "🔄 Vector Index Sync", "📁 Storage Paths"])

    # ---------------------------------------------------------------------------
    # Tab 1: AI & Database Health
    # ---------------------------------------------------------------------------
    with s_tab_health:
        st.subheader("Local Inference Engine (Ollama)")

        h1, h2 = st.columns(2)
        with h1:
            chat_health = check_ollama_health(services.config, model_name=services.config.tutor_model)
            if chat_health.is_ready:
                st.success(f"🟢 **Chat Model:** `{services.config.tutor_model}` (Ready)")
            else:
                st.error(f"🔴 **Chat Model:** `{services.config.tutor_model}` ({chat_health.status.value.upper()})")
                st.caption(f"Details: {chat_health.message}")

        with h2:
            embed_health = check_ollama_health(services.config, model_name=services.config.embedding_model)
            if embed_health.is_ready:
                st.success(f"🟢 **Embedding Model:** `{services.config.embedding_model}` (Ready)")
            else:
                st.warning(f"🟡 **Embedding Model:** `{services.config.embedding_model}` ({embed_health.status.value.upper()})")
                st.caption("Degrading gracefully to SQLite FTS5 lexical keyword search.")

        st.caption(f"Ollama Base URL: `{services.config.ollama_base_url}`")

        if not chat_health.is_ready or not embed_health.is_ready:
            st.markdown("---")
            st.subheader("🛠️ Remediation Instructions")
            st.code(
                f"# In a terminal, ensure Ollama is running and models are downloaded:\n"
                f"ollama serve\n\n"
                f"ollama pull {services.config.tutor_model}\n"
                f"ollama pull {services.config.embedding_model}",
                language="powershell",
            )

        st.markdown("---")
        st.subheader("Relational Database (SQLite)")
        with services.db.get_connection() as conn:
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            latest_mig = conn.execute("SELECT MAX(version) as v FROM schema_migrations;").fetchone()["v"]

        st.markdown(f"- **Journal Mode:** `{journal_mode.upper()}` (WAL)")
        st.markdown(f"- **Foreign Keys:** `{'ENABLED' if foreign_keys else 'DISABLED'}`")
        st.markdown(f"- **Latest Migration Applied:** `Version {latest_mig}`")

    # ---------------------------------------------------------------------------
    # Tab 2: Vector Index Sync
    # ---------------------------------------------------------------------------
    with s_tab_index:
        st.subheader("ChromaDB Vector Index Status")

        with services.db.get_connection() as conn:
            sql_chunks = conn.execute("SELECT COUNT(*) as c FROM chunks;").fetchone()["c"]
            sql_docs = conn.execute("SELECT COUNT(*) as c FROM documents;").fetchone()["c"]
            fts_rows = conn.execute("SELECT COUNT(*) as c FROM chunks_fts;").fetchone()["c"]

        chroma_vectors = 0
        if services.vector_store is not None:
            try:
                chroma_vectors = services.vector_store.count()
            except Exception as e:
                st.warning(f"Unable to read Chroma vector count: {e}")

        v1, v2, v3 = st.columns(3)
        with v1:
            st.metric("SQLite Chunks (Ground Truth)", sql_chunks)
        with v2:
            st.metric("FTS5 Indexed Rows", fts_rows)
        with v3:
            st.metric("ChromaDB Embeddings", chroma_vectors)

        if sql_chunks == chroma_vectors:
            st.success("✅ **Vector Index is Fully Synchronized with SQLite Ground Truth.**")
        else:
            diff = abs(sql_chunks - chroma_vectors)
            st.warning(f"⚠️ **Vector Index Desync Detected:** SQLite has {sql_chunks} chunks, but Chroma has {chroma_vectors} vectors ({diff} difference).")

        st.markdown("---")
        st.subheader("Reconcile Vector Index")
        st.markdown("Safely index unindexed chunks into ChromaDB without re-ingesting documents or touching SQLite.")

        if st.button("🔄 Synchronize Vector Index Now", type="primary", key="btn_sync_vectors"):
            if services.vector_store is None:
                st.error("Vector store is not initialized. Ensure embedding model is available.")
            else:
                with st.spinner("Synchronizing embeddings with SQLite..."):
                    try:
                        sync_stats = services.vector_store.sync(services.db)
                        st.success(
                            f"✅ Synchronization complete!\n\n"
                            f"- Vectors Upserted: `{sync_stats.upserted_count}`\n"
                            f"- Vectors Removed: `{sync_stats.deleted_count}`\n"
                            f"- Unchanged: `{sync_stats.unchanged_count}`"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Synchronization failed: {e}")

    # ---------------------------------------------------------------------------
    # Tab 3: Storage Paths
    # ---------------------------------------------------------------------------
    with s_tab_config:
        st.subheader("Local Storage Locations")
        db_path = Path(services.config.db_path)
        chroma_path = Path(services.config.chroma_dir or (services.config.data_dir / "chroma"))

        db_size_mb = 0.0
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)

        st.markdown(f"- **SQLite Database:** `{db_path}` ({db_size_mb:.2f} MB)")
        st.markdown(f"- **ChromaDB Directory:** `{chroma_path}`")
        st.markdown(f"- **App Data Root:** `{services.config.data_dir}`")
        st.markdown(f"- **Embeddings Model:** `{services.config.embedding_model}` ({services.config.embedding_dimension} dimensions)")
        st.markdown(f"- **Tutor & Assessment Model:** `{services.config.tutor_model}`")
