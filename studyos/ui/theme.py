"""Styling, theme definitions, and CSS injection for StudyOS UI."""

from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """Inject polished, modern CSS styling into the Streamlit app."""
    css = """
    <style>
    /* Global application styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric cards */
    .metric-container {
        background-color: var(--background-color, #ffffff);
        border: 1px solid var(--border-color, #e5e7eb);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text-color, #111827);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.825rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        line-height: 1;
        letter-spacing: 0.025em;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-blue {
        background-color: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
    }
    .badge-purple {
        background-color: #f3e8ff;
        color: #6b21a8;
        border: 1px solid #e9d5ff;
    }
    .badge-gray {
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #e5e7eb;
    }

    /* Citation Cards */
    .citation-card {
        background-color: var(--secondary-background-color, #f9fafb);
        border: 1px solid #e5e7eb;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    .citation-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1f2937;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }
    .citation-excerpt {
        font-size: 0.825rem;
        color: #4b5563;
        font-style: italic;
        line-height: 1.4;
        background: #ffffff;
        padding: 0.4rem 0.6rem;
        border-radius: 4px;
        border: 1px dashed #d1d5db;
    }

    /* Question Card in Exams */
    .question-card {
        background-color: var(--secondary-background-color, #f9fafb);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    /* Info Callout */
    .studyos-callout {
        border-left: 4px solid #4f46e5;
        background-color: #eef2ff;
        color: #312e81;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin: 0.75rem 0;
        font-size: 0.875rem;
    }

    /* Sidebar Branding */
    .sidebar-brand {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .sidebar-brand h2 {
        font-size: 1.35rem;
        font-weight: 800;
        color: #4338ca;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .sidebar-brand p {
        font-size: 0.75rem;
        color: #6b7280;
        margin: 0.2rem 0 0 0;
        letter-spacing: 0.02em;
    }

    /* Hide standard Streamlit header decoration */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

