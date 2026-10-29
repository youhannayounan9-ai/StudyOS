"""Session state management and service container for StudyOS Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any
import streamlit as st

from studyos.assessment.models import ExamResult, PublicExam
from studyos.assessment.service import AssessmentService
from studyos.core.config import AppConfig, get_default_config
from studyos.core.health import check_ollama_health
from studyos.inference.fake import FakeChatProvider
from studyos.inference.models import ChatRole
from studyos.inference.ollama import OllamaChatProvider
from studyos.inference.protocol import ChatProvider
from studyos.ingestion.models import Course
from studyos.ingestion.service import IngestionService
from studyos.learner.models import MasteryCategory
from studyos.learner.planner import AdaptiveStudyPlan
from studyos.learner.service import LearnerService
from studyos.research import (
    ExternalResearchService,
    FakeSearchProvider,
    ResearchRepository,
    WikipediaSearchProvider,
)
from studyos.retrieval.embedding import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    OllamaEmbeddingProvider,
)
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database
from studyos.tutoring.service import TutorService

logger = logging.getLogger("studyos.ui")


@dataclass
class UIServices:
    """Container holding instantiated StudyOS backend services."""

    config: AppConfig
    db: Database
    vector_store: ChromaVectorStore | None
    ingestion_service: IngestionService
    hybrid_service: HybridRetrievalService
    tutor_service: TutorService | None
    assessment_service: AssessmentService
    learner_service: LearnerService
    research_service: ExternalResearchService | None
    chat_provider: ChatProvider | None
    embedding_provider: EmbeddingProvider | None
    ollama_ready: bool
    embedding_ready: bool
    ollama_message: str | None


def create_services(
    config: AppConfig | None = None,
    use_fake_inference: bool = False,
) -> UIServices:
    """Create and configure all StudyOS services."""
    cfg = config or get_default_config()
    db = Database(cfg.db_path)
    db.migrate()

    chat_provider: ChatProvider | None = None
    embedding_provider: EmbeddingProvider | None = None
    ollama_ready = False
    embedding_ready = False
    ollama_msg: str | None = None

    if use_fake_inference:
        def _fake_chat_handler(req: Any) -> str:
            user_msg = req.messages[-1].content if req.messages else ""
            user_blob = (user_msg or "").lower()
            sys_msg = req.messages[0].content if len(req.messages) > 1 else ""
            sys_blob = (sys_msg or "").lower()

            # 1. Exam Generation
            if "exam generator" in sys_blob or "exam specification" in user_blob:
                if "permitted question types: short_answer" in user_blob:
                    return json.dumps({
                        "questions": [
                            {
                                "type": "short_answer",
                                "question": "Explain virtual memory.",
                                "difficulty": "medium",
                                "correct_answer": "Virtual memory uses paging to isolate address spaces [E1].",
                                "explanation": "Paging isolates address spaces [E1].",
                                "rubric": {
                                    "criteria": ["paging", "isolation"],
                                    "key_points": ["virtual memory", "paging"],
                                    "max_score": 1.0,
                                },
                                "evidence_markers": ["E1"],
                            },
                            {
                                "type": "short_answer",
                                "question": "How does address translation work?",
                                "difficulty": "medium",
                                "correct_answer": "The MMU translates virtual pages to physical frames [E1].",
                                "explanation": "Paging protects execution [E1].",
                                "rubric": {
                                    "criteria": ["MMU", "translation"],
                                    "key_points": ["pages", "frames"],
                                    "max_score": 1.0,
                                },
                                "evidence_markers": ["E1"],
                            },
                        ]
                    })
                return json.dumps({
                    "questions": [
                        {
                            "type": "mcq",
                            "question": "What is the primary characteristic of Dijkstra's algorithm?",
                            "difficulty": "medium",
                            "options": [
                                {"id": "A", "text": "It handles negative edge weights"},
                                {"id": "B", "text": "It finds single-source shortest paths on non-negative weights"},
                                {"id": "C", "text": "It computes all-pairs shortest paths"},
                                {"id": "D", "text": "It relies on Bellman-Ford relaxations"},
                            ],
                            "correct_answer": "B",
                            "explanation": "Dijkstra solves single-source shortest paths on non-negative graphs [E1].",
                            "evidence_markers": ["E1"],
                        },
                        {
                            "type": "true_false",
                            "question": "Bellman-Ford algorithm can detect negative weight cycles in a graph.",
                            "difficulty": "easy",
                            "correct_answer": "True",
                            "explanation": "If a relaxation succeeds after |V|-1 passes, a negative cycle exists [E1].",
                            "evidence_markers": ["E1"],
                        },
                    ]
                })
            return "This concept is grounded in the provided course notes [E1]."

        chat_provider = FakeChatProvider(
            model_name="fake-tutor",
            default_response="This concept is grounded in the provided course notes [E1].",
            response_fn=_fake_chat_handler,
        )
        embedding_provider = FakeEmbeddingProvider(dimension=cfg.embedding_dimension)
        ollama_ready = True
        embedding_ready = True
        ollama_msg = "Running with offline fake inference provider."
    else:
        # Check Chat Health
        chat_health = check_ollama_health(cfg, model_name=cfg.tutor_model)
        ollama_ready = chat_health.is_ready
        if ollama_ready:
            chat_provider = OllamaChatProvider.from_config(cfg)
        else:
            ollama_msg = chat_health.message

        # Check Embedding Health
        embed_health = check_ollama_health(cfg, model_name=cfg.embedding_model)
        embedding_ready = embed_health.is_ready
        if embedding_ready:
            embedding_provider = OllamaEmbeddingProvider.from_config(cfg)

    # Initialize Vector Store (falls back safely if provider is None)
    vector_store: ChromaVectorStore | None = None
    try:
        vector_store = ChromaVectorStore.from_config(cfg, provider=embedding_provider)
    except Exception as e:
        logger.warning("Could not initialize ChromaVectorStore: %s", e)
        vector_store = None

    # Hybrid Retrieval (degrades to lexical if vector_store or embedding_provider is missing)
    hybrid_service = HybridRetrievalService(
        db=db,
        vector_store=vector_store,
        rrf_k=cfg.rrf_k,
    )

    # Ingestion Service
    ingestion_service = IngestionService(
        db=db,
        config=cfg,
        vector_store=vector_store,
    )

    # Tutor Service
    tutor_service: TutorService | None = None
    if chat_provider is not None:
        tutor_service = TutorService(
            retrieval_service=hybrid_service,
            chat_provider=chat_provider,
            config=cfg,
        )

    # Learner Service
    learner_service = LearnerService(
        db=db,
        chat_provider=chat_provider,
        config=cfg,
    )

    # Assessment Service
    assessment_service = AssessmentService(
        db=db,
        retrieval_service=hybrid_service,
        chat_provider=chat_provider,
        config=cfg,
        learner_service=learner_service,
    )

    # Research Service
    search_provider = FakeSearchProvider() if use_fake_inference else WikipediaSearchProvider()
    research_service = ExternalResearchService(
        db=db,
        provider=search_provider,
        chat_client=chat_provider,
    )

    return UIServices(
        config=cfg,
        db=db,
        vector_store=vector_store,
        ingestion_service=ingestion_service,
        hybrid_service=hybrid_service,
        tutor_service=tutor_service,
        assessment_service=assessment_service,
        learner_service=learner_service,
        research_service=research_service,
        chat_provider=chat_provider,
        embedding_provider=embedding_provider,
        ollama_ready=ollama_ready,
        embedding_ready=embedding_ready,
        ollama_message=ollama_msg,
    )


def get_services(
    config: AppConfig | None = None,
    force_recreate: bool = False,
    use_fake_inference: bool = False,
) -> UIServices:
    """Retrieve singleton UIServices instance from Streamlit session state or cache."""
    # When running inside Streamlit, we cache in session_state
    if "services" not in st.session_state or force_recreate:
        services = create_services(config=config, use_fake_inference=use_fake_inference)
        st.session_state["services"] = services
        return services
    return st.session_state["services"]


def init_session_state() -> None:
    """Ensure all required session state keys are initialized with defaults."""
    defaults: dict[str, Any] = {
        "selected_course_id": None,
        "active_nav": "Dashboard",
        "chat_history": [],
        "active_exam_id": None,
        "active_attempt_id": None,
        "active_exam_public": None,
        "exam_answers": {},
        "last_exam_result": None,
        "last_research_result": None,
        "last_study_plan": None,
        "upload_status": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_active_course(services: UIServices) -> Course | None:
    """Get the currently selected course object from session state."""
    cid = st.session_state.get("selected_course_id")
    if not cid:
        return None
    return services.db.get_course(cid)


def clean_temp_file(file_path: Path | str | None) -> None:
    """Safely delete a temporary file if it exists."""
    if not file_path:
        return
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to remove temp file %s: %s", file_path, e)


def format_mastery_badge(category: MasteryCategory) -> tuple[str, str, str]:
    """Return (display_text, background_color, text_color) for a mastery category."""
    mapping = {
        MasteryCategory.STRONG: ("Strong", "#dcfce7", "#166534"),
        MasteryCategory.MODERATE: ("Moderate", "#e0f2fe", "#075985"),
        MasteryCategory.WEAK: ("Weak", "#fee2e2", "#991b1b"),
        MasteryCategory.NEEDS_EVIDENCE: ("Needs Evidence", "#fef3c7", "#92400e"),
        MasteryCategory.UNASSESSED: ("Unassessed", "#f3f4f6", "#4b5563"),
    }
    return mapping.get(category, ("Unknown", "#f3f4f6", "#4b5563"))
