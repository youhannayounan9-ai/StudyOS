# StudyOS — Staged Development Roadmap & Execution Plan

- **Document Version:** 1.2.0
- **Status:** Approved Baseline (Committed at 1827f1a) | Stage 1 Complete & Approved (Committed at 84db803) | Stage 2 Complete & Approved (Committed at 7026c4c) | Stage 3 Complete & Approved (Committed at 3d9e26e) | Stage 4 Complete & Approved (Committed at 7351c60) | Stage 5 Complete & Approved (Committed at 73d49bf) | Stage 6 Implementation in review
- **System Target:** Local-First Personalized Learning Operating System
- **Hardware Profile:** Windows 11 | NVIDIA RTX 5060 Laptop GPU (8 GB VRAM) | 16 GB System RAM

---

## 1. Roadmap Philosophy & Stage-Gate Principles

StudyOS follows a disciplined, stage-gated engineering methodology:
1. **Incremental Completeness:** Each stage produces a testable, verifiable, and working artifact. No stage begins until the previous stage satisfies its defined acceptance criteria.
2. **First Usable Release Prioritization:** Stages 0 through 4 deliver the foundational, fully functional local study loop (Ingest $\rightarrow$ Retrieve $\rightarrow$ Tutor with Citations $\rightarrow$ Exam & Grade). Stages 5 through 7 enrich the cognitive modeling, adaptation, and system hardening.
3. **Realistic Verification Time:** Hour estimates explicitly include debugging on Windows, edge-case testing, citation auditing, and learning curves.
4. **No Premature Optimization:** Architectural modularity is established early, but complex features (e.g., cross-encoders, multimodal vision, external research) are gated behind stable core baselines.
5. **Explicit Git Commit Checkpoints:** Work for each stage is reviewed and verified in the local working tree before being committed as a permanent baseline. Stage 0 documentation was approved and committed as baseline commit `1827f1a`; Stage 1 ingestion engine is approved and committed as baseline commit `84db803`; Stage 2 hybrid retrieval is committed as baseline commit `7026c4c`; Stage 3 AI tutor is committed as baseline commit `3d9e26e`; Stage 4 assessment engine is approved and committed as baseline commit `7351c60`; Stage 5 learner model and mastery tracking is approved and committed as baseline commit `73d49bf`; Stage 6 adaptive learning, concept graph, and external research is currently implemented and under review.

---

## 2. Stage Breakdown: Specifications & Gates

---

### Stage 0: Project Specification & Architecture Baseline
- **Scope:** Define complete product vision, user personas, hardware constraints, functional requirements (P0/P1/P2) with acceptance criteria, system architecture, module boundaries, data flows, technical decisions, resource budgets, and roadmap. Verify real environment facts using read-only diagnostics.
- **Explicit Non-Goals:** Writing application code, creating UI components, installing new packages, downloading models, or altering system settings.
- **Expected Deliverables:**
  - `README.md` (clean project overview, review status, and verified environment facts).
  - `docs/PROJECT_SPEC.md` (product specification, untrusted input boundaries, and acceptance criteria).
  - `docs/ARCHITECTURE.md` (modular design, evidence ID provenance, FTS5 triggers, and resource budgets).
  - `docs/ROADMAP.md` (this phased implementation blueprint).
  - Stage 0 Architectural Verification & Review Report.
- **Prerequisites:** Clean workspace, access to terminal read-only diagnostic tools.
- **Risks:** Scope creep; ambiguous acceptance criteria; unrealistic hardware assumptions.
- **Verification Commands & Procedures:**
  - Verify workspace isolation: `(Get-Location).Path`.
  - Verify environment facts: `python --version`, `py --list`, `nvidia-smi`, `uv --version`, `ollama --version`.
  - Internal documentation consistency cross-check.
- **Acceptance Criteria:**
  - All four core documentation files completed and consistent.
  - Zero application code written, zero packages installed, and zero models downloaded.
  - Every P0 requirement has a concrete, verifiable acceptance criterion.
  - Stable baseline commit created in Git upon external review sign-off.
- **Estimated Focused Hours:** 6–10 hours (Completed in Stage 0).
- **Dependencies:** None.

---

### Stage 1: Document Ingestion, Parsing & Normalization Engine
- **Scope:** Implement the document ingestion pipeline with clear architectural separation between:
  1. *Parser Extraction:* Raw text and layout extraction from PDF, DOCX, PPTX, and TXT files.
  2. *Text Normalization:* Whitespace cleansing, UTF-8 normalization, and structural tag preservation.
  3. *Structure-Aware Chunking:* Bounded passages (300–600 tokens) with slide/page breadcrumb headers.
  4. *SQLite Persistence & FTS5 Synchronization:* Storing documents and chunks in SQLite, using database triggers (`chunks_ai`, `chunks_ad`, `chunks_au`) as the single synchronization owner for `chunks_fts`.
  Implement SHA-256 deduplication, corrupt file rejection, and transaction rollback recovery.
- **Explicit Non-Goals:** Vector embeddings, ChromaDB integration, Streamlit UI, LLM inference.
- **Expected Deliverables:**
  - `tests/fixtures/synthetic/`: Small, deterministic synthetic test files (`sample_lecture.pdf`, `sample_slides.pptx`, `sample_notes.docx`, `sample_doc.txt`).
  - `studyos.core.config`: Configuration loader.
  - `studyos.storage.db`: SQLite database connection with WAL mode and schema migration scripts (including FTS5 tables and triggers).
  - `studyos.ingestion`: `DocumentParser` interface with `PDFParser`, `DOCXParser`, `PPTXParser`, `TXTParser`.
  - `studyos.chunking`: `StructureAwareChunker` producing standardized chunk objects with context headers.
  - Unit test suite: `tests/unit/test_parsers.py`, `tests/unit/test_chunker.py`, `tests/unit/test_db.py`, `tests/unit/test_fts_sync.py`.
  - CLI ingestion verification script: `scripts/verify_ingestion.py`.
- **Prerequisites:** Stage 0 documentation approved and committed; `uv` environment created with Python 3.12 target and initial parsing libraries (`pypdf`, `python-docx`, `python-pptx`).
- **Risks:**
  - Complex multi-column PDF layouts causing broken sentences.
  - PPTX presentations with empty shapes or unusual slide layouts.
- **Verification Commands & Procedures:**
  - Run test suite: `pytest tests/unit/ -v`.
  - Ingestion CLI test: `python scripts/verify_ingestion.py --file tests/fixtures/synthetic/sample_lecture.pdf --course "CS101"`.
  - FTS5 sync test: `pytest tests/unit/test_fts_sync.py -k "test_cascade_delete or test_rollback_recovery" -v`.
- **Acceptance Criteria:**
  - Ingestion of test synthetic PDF, DOCX, PPTX, and TXT files completes with 100% test pass rate.
  - Chunks preserve page/slide metadata accurately.
  - Duplicate ingestion of the same file is detected via SHA-256 and rejected cleanly.
  - Trigger-based FTS5 synchronization passes tests for chunk insertion, deletion, course cascade deletion, content updates, and transaction rollback recovery.
- **Estimated Focused Hours:** 14–20 hours (Uncertainty: $\pm 20\%$).
- **Dependencies:** Stage 0.

---

### Stage 2: Hybrid Retrieval & Evidence Grounding Engine
- **Status:** Complete & Approved (Committed at `7026c4c`).
- **Scope:** Implement local vector embeddings using ChromaDB (persistent local directory) and keyword search using SQLite FTS5 (BM25). Build the hybrid retrieval service using Reciprocal Rank Fusion (RRF). Enforce document-level filtering (`course_id`, `document_id` selection). Implement pre-flight health diagnostics for local Ollama service.
- **Explicit Non-Goals:** Full conversational chatbot, exam generation, Streamlit UI.
- **Expected Deliverables:**
  - `studyos.core.health`: Diagnostic health check for Ollama daemon connectivity producing actionable user errors if offline.
  - `studyos.retrieval.vector`: ChromaDB wrapper with `EmbeddingProvider` protocol (Ollama `all-minilm` / local ONNX fallback).
  - `studyos.retrieval.fts`: SQLite FTS5 query service for BM25 ranking.
  - `studyos.retrieval.hybrid`: RRF hybrid fusion service with configurable weights and confidence thresholds.
  - Automated retrieval verification harness: `tests/integration/test_hybrid_retrieval.py`.
- **Prerequisites:** Stage 1 complete; SQLite database containing indexed chunks; Ollama running with `all-minilm` or local embedding runner available.
- **Risks:**
  - Discrepancies between BM25 tokenization and vector similarity ranking.
  - ChromaDB Windows file-lock edge cases during rapid writes.
- **Verification Commands & Procedures:**
  - Pre-flight check: `python -m studyos.core.health`.
  - Run integration tests: `pytest tests/integration/test_hybrid_retrieval.py -v`.
  - CLI retrieval test: `python scripts/test_retrieval.py --query "Dijkstra negative weights" --course "CS101" --top-k 5`.
- **Acceptance Criteria:**
  - Health check cleanly detects if Ollama daemon is offline and emits actionable instructions.
  - Hybrid search over test chunks executes in $< 500$ ms for top-10 results.
  - Filtering by selected `document_id` strictly excludes chunks from unselected documents.
  - Exact technical terms (e.g., function names, formulas) correctly rank in the top-3 via BM25 fusion.
- **Estimated Focused Hours:** 12–18 hours (Uncertainty: $\pm 25\%$).
- **Dependencies:** Stage 1.

---

### Stage 3: AI Tutor & Grounded Interactive Answering
- **Status:** Complete & Approved (Committed at `3d9e26e`).
- **Scope:** Build the local model inference wrapper (`studyos.inference`) connecting to Ollama (`qwen2.5:7b`) with pre-flight connectivity checks and local loopback validation. Implement the evidence-grounded tutoring service (`studyos.tutoring`). Build prompt templates that enclose untrusted text in `<evidence_context>` delimiters, assign temporary evidence markers (`[E1]`, `[E2]`), and resolve validated markers to trusted database metadata. Implement operational CLI (`scripts/ask_studyos.py`) with honest degraded-mode reporting and abstention.
- **Explicit Non-Goals:** Exam generation, grading rubrics, automated learner modeling, Streamlit UI (deferred to UI phase).
- **Expected Deliverables:**
  - `studyos.inference.ollama`: Ollama API client with pre-flight health checks, typed models, timeout handling, and local loopback validation.
  - `studyos.inference.fake`: Deterministic fake chat provider for offline test suites.
  - `studyos.tutoring.tutor`: RAG tutoring orchestrator with untrusted content isolation, evidence ID assignment, and resolution.
  - `studyos.tutoring.citations`: Extraction and strict set-membership validation of citation markers against active retrieval results.
  - `scripts/ask_studyos.py`: Operational command-line entrypoint for grounded answering.
  - `tests/integration/test_tutor_grounding.py`: Automated tests asserting refusal on out-of-domain queries, course isolation, and rejection of invalid evidence markers.
- **Prerequisites:** Stage 2 complete; Ollama running with 7B model or offline fake provider.
- **Risks:**
  - Model hallucinating citations not present in retrieved context (mitigated by strict set-membership validation).
  - High generation latency if VRAM threshold is breached.
- **Verification Commands & Procedures:**
  - Run full test suite: `pytest -v`.
  - Grounding integration test: `pytest tests/integration/test_tutor_grounding.py -v`.
  - CLI tutor test: `python scripts/ask_studyos.py --query "Explain Dijkstra algorithm" --course "CS101" --fake-chat --fake-embeddings`.
- **Acceptance Criteria:**
  - Tutor answers reference valid application evidence markers (`[E1]`, `[E2]`) that resolve to verified source chunks from the selected document set; any unmapped or hallucinated marker is rejected.
  - When asked about a concept not present in selected materials or when retrieval returns 0 chunks, tutor explicitly outputs the standard refusal message.
  - 100% offline test suite execution with zero cloud calls or model downloads.
- **Estimated Focused Hours:** 16–24 hours (Completed in Stage 3).
- **Dependencies:** Stage 2.

---

### Stage 4: Exam Generation, Submission & Automated Grading Engine
- **Status:** Complete & Approved (Baseline Committed at `7351c60`).
- **Scope:** Build a dependable, local-first assessment engine that generates customized practice exams (Multiple Choice, True/False, and Short-Answer) from course materials using hybrid retrieval, assigned evidence markers, and local Ollama inference. Enforce answer-key confidentiality via public exam views (`PublicExam`) that strictly omit correct answers, reference answers, explanations, and rubrics. Store exams, questions, and attempts persistently in SQLite using migrations `0003_assessment_engine.sql`, `0004_grading_states_and_citation_snapshots.sql`, and `0005_source_reference_set_null.sql`. Implement deterministic objective grading, skipped-question scoring (0.0), and conservative short-answer manual review/model evaluation. Provide post-submission results grounded in verified SQLite chunk metadata with stale-chunk detection.
- **Explicit Non-Goals:** Web/Streamlit UI (deferred), persistent learner mastery graphs, spaced repetition scheduling, or cross-course skill tracking (reserved for Stage 5).
- **Expected Deliverables:**
  - `studyos.storage.migrations.0003_assessment_engine.sql`: Normalized tables (`exams`, `exam_questions`, `exam_question_sources`, `exam_attempts`, `question_attempts`) with foreign keys, indexes, and cascade behavior.
  - `studyos.storage.migrations.0004_grading_states_and_citation_snapshots.sql`: Upgrades schema with provisional scores, explicit grading statuses, and historical citation snapshots.
  - `studyos.storage.migrations.0005_source_reference_set_null.sql`: Rebuilds exam_question_sources to enforce ON DELETE SET NULL on chunk foreign keys, ensuring historical citations survive chunk deletion.
  - `studyos.assessment.models`: Pydantic domain models for requests, questions, public taking views, attempts, submissions, and grading results.
  - `studyos.assessment.prompt`: Exam generation prompt with XML evidence encapsulation and JSON schema instructions.
  - `studyos.assessment.validator`: Rigorous structural and semantic validator (MCQ single correct choice, distinct choices, boolean normalization, marker validation, deduplication).
  - `studyos.assessment.repository`: Transactional persistence, leak-proof public retrieval, atomic attempt recording, and stale chunk detection.
  - `studyos.assessment.grader`: Deterministic objective grader and conservative short-answer evaluation.
  - `studyos.assessment.generator`: End-to-end exam generator with hybrid retrieval, bounded correction retries, and atomic persistence.
  - `studyos.assessment.service`: High-level assessment service unifying generation, taking, submission, and grading.
  - `scripts/exams.py`: Operational assessment CLI (`generate`, `list`, `take`, `submit`, `results`, `review`).
  - Unit and integration tests: `tests/unit/test_exam_models_and_validation.py`, `tests/unit/test_exam_repository_and_security.py`, `tests/unit/test_exam_grading.py`, `tests/integration/test_assessment_pipeline.py`.
- **Prerequisites:** Stage 3 complete; reliable JSON mode inference from local model or offline fake chat provider.
- **Risks:**
  - Distractor ambiguity or duplicate options (mitigated by strict validation rules).
  - Answer-key leakage prior to attempt submission (mitigated by separate public models).
  - Short-answer scoring hallucination (mitigated by conservative manual review fallback and bounded rubrics).
- **Verification Commands & Procedures:**
  - Run full test suite: `pytest -v`.
  - Assessment unit tests: `pytest tests/unit/test_exam_*.py -v`.
  - Assessment pipeline integration test: `pytest tests/integration/test_assessment_pipeline.py -v`.
  - Operational CLI demo: `python scripts/exams.py --help`.
- **Acceptance Criteria:**
  - Complete, coherent assessment lifecycle without web UI (generate, list, take, submit, grade, view results).
  - Public exam models serialize 0 answer keys, reference answers, explanations, or rubrics.
  - Objective grading (MCQ, True/False) is 100% deterministic; skipped questions score 0.0 with clear feedback.
  - Generated questions cite valid application-assigned evidence markers resolving to SQLite chunk metadata.
  - 100% offline test suite execution with 175 passing tests.
- **Estimated Focused Hours:** 18–26 hours (Completed in Stage 4).
- **Dependencies:** Stage 3.

---

### Stage 5: Persistent Learner Model, Mastery Tracking & Personalized Revision
- **Status:** Implementation in review (212 automated tests passing).
- **Scope:** Build a persistent local learner cognitive model that tracks study activity, assessment history, and concept-level mastery from course materials. Implement a course-scoped normalized concept catalog with question-to-concept and source-chunk mappings. Implement deterministic, explainable Laplace-smoothed mastery estimation, distinguishing lack of evidence from poor understanding and never penalizing pending manual reviews. Implement an offline spaced-repetition review scheduler with due/overdue tracking, rating progression, and explainable recommendations. Build an operational CLI (`scripts/learner.py`) for inspecting progress, listing concepts/reviews, recording review outcomes, and diagnostic explanations.
- **Explicit Non-Goals:** Web or Streamlit UI (deferred to UI phase), external web research, multi-user authentication, cloud APIs, unbounded conversational memory.
- **Expected Deliverables:**
  - `studyos.storage.migrations.0006_learner_model.sql`: Normalized tables (`learner_profile`, `concepts`, `question_concepts`, `concept_chunks`, `concept_mastery`, `learner_events`, `review_tasks`) with foreign keys, indexes, and cascade integrity.
  - `studyos.learner.models`: Pydantic domain models for learner profiles, concepts, mastery, review tasks, progress summaries, and recommendations.
  - `studyos.learner.mastery`: Deterministic Laplace-smoothed mastery estimation with uncertainty tracking and categorization (`unassessed`, `needs_evidence`, `weak`, `moderate`, `strong`).
  - `studyos.learner.scheduler`: Deterministic bounded spaced repetition scheduler (SM-2/Leitner interval policy, due/overdue calculation).
  - `studyos.learner.concept_mapper`: Concept name normalization, trusted source title extraction, and bounded local LLM tagging fallback.
  - `studyos.learner.repository`: Authoritative SQLite persistence for learner events, mastery states, concepts, and review queues with nested transaction safety.
  - `studyos.learner.service`: High-level learner service orchestrating idempotent attempt processing, manual regrading updates, course summaries, and recommendations.
  - `scripts/learner.py`: Operational learner CLI (`progress`, `concepts`, `history`, `reviews`, `record-review`, `recommendations`, `why`).
  - Automated tests: `tests/unit/test_learner_models.py`, `tests/unit/test_learner_mastery.py`, `tests/unit/test_learner_scheduler.py`, `tests/unit/test_concept_mapping.py`, `tests/unit/test_learner_repository.py`, `tests/unit/test_learner_service.py`, `tests/unit/test_learner_migrations.py`, `tests/integration/test_learner_end_to_end.py`.
- **Prerequisites:** Stage 4 complete; SQLite database containing assessment attempts.
- **Risks:**
  - Double-counting attempts on repeat processing (mitigated by idempotent attempt event checks).
  - Conflating pending manual review with zero score (mitigated by explicit nullable scores and status filtering).
  - Premature claims of mastery on small samples (mitigated by Laplace smoothing and explicit `needs_evidence` category for N < 3).
- **Verification Commands & Procedures:**
  - Run full test suite: `pytest -v` (212 passing tests).
  - Learner unit tests: `pytest tests/unit/test_learner_*.py -v`.
  - Learner integration pipeline test: `pytest tests/integration/test_learner_end_to_end.py -v`.
  - Operational CLI demo: `python scripts/learner.py progress`, `python scripts/learner.py concepts`, `python scripts/learner.py recommendations`.
- **Acceptance Criteria:**
  - Concept mastery updates deterministically after exam submission or manual review finalization.
  - Zero observations strictly yields `unassessed` (never zero mastery). Pending manual review questions never penalize mastery.
  - Spaced repetition queue correctly tracks due and overdue dates and advances intervals upon recorded outcomes.
  - Course-level data isolation is strictly enforced across concepts, mastery records, and progress summaries.
  - 100% offline test suite execution with 212 passing tests and zero regressions.
- **Estimated Focused Hours:** 14–20 hours (Completed in Stage 5).
- **Dependencies:** Stage 4.

---

### Stage 6: Advanced Adaptive Learning, Knowledge Graph & External Research
- **Status:** Approved & Baseline Committed at `bac6d78`.
- **Scope:** Provide evidence-backed concept discovery from ingested documents independently of exams, course-scoped concept relationship graph (`prerequisite`, `related`) with deterministic DAG cycle detection, advanced forgetting-aware review scheduling (`local-fsrs-v1` DSR model with power-law retention), transparent adaptive study planner with time budgeting, and explicitly opt-in external research with strict privacy boundaries, SSRF prevention, and verified `[EXT-N]` citations.
- **Explicit Non-Goals:** Unbounded internet browsing, automatic document leakage to external APIs, speculative psychological claims.
- **Expected Deliverables:**
  - `studyos/learner/discovery.py`: `ConceptDiscoveryService` for direct chunk-grounded concept extraction with chunk provenance.
  - `studyos/learner/graph.py`: `ConceptGraph` with BFS cycle detection, Kahn's topological sort, and course isolation.
  - `studyos/learner/scheduler.py`: Advanced forgetting-aware scheduler (`local-fsrs-v1`) with power-law retrievability $R(t,S) = (1 + \frac{19}{81}\frac{t}{S})^{-0.5}$ and bounds $[1.0, 365.0]$d.
  - `studyos/learner/planner.py`: `AdaptivePlannerService` with deterministic priority ranking and bounded time budgets.
  - `studyos/research/`: External research package (`safety.py`, `service.py`, `repository.py`, `WikipediaSearchProvider`, `FakeSearchProvider`) with SSRF guards and citation validation.
  - `scripts/learner.py`: CLI subcommands (`discover-concepts`, `relationships`, `add-relationship`, `prerequisites`, `plan`, `research`, `research-history`).
  - Automated tests: `tests/unit/test_stage6_migrations.py`, `tests/unit/test_advanced_scheduler.py`, `tests/unit/test_concept_graph.py`, `tests/unit/test_concept_discovery.py`, `tests/unit/test_adaptive_planner.py`, `tests/unit/test_external_research.py`, `tests/integration/test_stage6_end_to_end.py`.
- **Prerequisites:** Stage 5 complete and committed.
- **Risks:**
  - Cycle introduction in concept prerequisite relationships (Mitigated by BFS traversal before edge insertion).
  - External search SSRF or DNS rebinding (Mitigated by `SafeURLValidator` blocking loopback, private RFC 1918, link-local, and cloud metadata).
  - Model hallucinated citations (Mitigated by application-assigned `[EXT-N]` indices and source set validation).
- **Verification Commands & Procedures:**
  - Full test suite: `pytest -v` (277 passed).
  - Direct discovery CLI test: `python scripts/learner.py discover-concepts --course <id>`.
  - Concept graph inspection: `python scripts/learner.py relationships --course <id>`.
  - Adaptive study plan: `python scripts/learner.py plan --course <id> --time-budget 30`.
  - External research with opt-in: `python scripts/learner.py research --query "<text>" --external-research`.
- **Acceptance Criteria:**
  - Concepts discovered from raw chunks retain chunk-level provenance in `concept_chunks`.
  - Accepted prerequisite edges are strictly acyclic; cross-course edges are rejected.
  - Forgetting-aware review scheduler calculates valid bounded intervals from graded history without penalizing pending manual reviews.
  - Adaptive study planner produces transparent, time-budgeted plans using deterministic rules without runtime LLM calls.
  - External research requires explicit user opt-in (`--external-research`); default study operations remain 100% offline.
  - Zero private course text, uploaded documents, or study history is sent to external search providers.
- **Estimated Focused Hours:** 16–22 hours (Completed in Stage 6).
- **Dependencies:** Stage 5.

---

### Stage 7: Comprehensive Evaluation, Security, Performance & Portfolio Polish
- **Status:** Implementation in review.
- **Scope:** Build a deterministic evaluation benchmark harness testing retrieval recall, citation resolution, and security isolation; harden document parsers against decompression archive bombs; implement atomic SQLite transactional migrations with rollback; verify backup and recovery integrity; establish GitHub Actions CI workflow with matrix OS testing; and finalize developer documentation.
- **Explicit Non-Goals:** Cloud commercialization, live external network calls during CI, rewriting operational architecture.
- **Expected Deliverables:**
  - `scripts/evaluate.py`: Deterministic evaluation benchmark measuring Retrieval Recall@K, MRR, citation resolution, answer-key isolation, learner mastery updates, and graph invariants.
  - `studyos/ingestion/parsers.py`: Ingestion archive decompression safety validation (`validate_zip_archive_safety`).
  - `studyos/storage/db.py`: Atomic transactional migration runner wrapping every migration in `BEGIN TRANSACTION; ... COMMIT;` with rollback on error.
  - `.github/workflows/ci.yml`: GitHub Actions continuous integration workflow testing on Windows and Ubuntu.
  - `tests/unit/test_stage7_migrations_and_recovery.py`: Migration upgrades, fresh creation, FK integrity, and online SQLite backup/restore test suite.
  - `tests/unit/test_evaluation_harness.py`: Automated pytest test for the benchmark runner.
- **Verification Commands & Procedures:**
  - Deterministic evaluation benchmark: `python scripts/evaluate.py` (and `python scripts/evaluate.py --json`).
  - Full automated test suite: `pytest -v` (287 passed).
  - Lockfile consistency: `uv lock --check`.
  - Packaging verification: `uv build`.
- **Acceptance Criteria:**
  - 100% offline test suite passes with 287 tests and zero regressions.
  - Deterministic evaluation benchmark produces PASS status with 100% Recall@5 and zero answer-key leakage.
  - Migration rollback leaves zero partial tables or uncommitted state on failure.
  - Online SQLite backup restores databases with 100% data fidelity and 0 foreign key violations.
- **Estimated Focused Hours:** 18–24 hours (Completed in Stage 7).
- **Dependencies:** Stages 1–6.

---

## 3. Milestone Summary & Time Estimates

| Stage | Focus Area | Deliverable Milestone | Focused Hours (Est.) | Cumulative Hours |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Architecture & Specification** | Specs, Architecture, Roadmap docs | **6–10 h** *(Done)* | 6–10 h |
| **1** | **Document Ingestion & Storage** | Parsers, Chunking, SQLite DB | **14–20 h** *(Done)* | 20–30 h |
| **2** | **Hybrid Retrieval Engine** | ChromaDB, FTS5 BM25, RRF Fusion | **12–18 h** *(Done)* | 32–48 h |
| **3** | **AI Tutor with Citations** | Ollama 7B/8B, Grounding, Streamlit UI | **16–24 h** *(Done)* | 48–72 h |
| **4** | **Assessment & Exam Engine** | Quiz Gen, Leak Prevention, Grading | **18–26 h** *(Done)* | 66–98 h |
| **5** | **Persistent Learner Model** | Mastery Tracking, Spaced Revision | **14–20 h** *(Done)* | 80–118 h |
| **6** | **Adaptive Learning & Research** | Concept Graph, FSRS, Adaptive Plan, Research | **16–22 h** *(Done)* | 96–140 h |
| **7** | **Evaluation & Portfolio Polish** | Benchmarks, CI, Security, Recoverability | **18–24 h** *(Under Review)* | 114–164 h |
| **Total** | **Full System Lifecycle** | **Production-Grade Personal StudyOS** | **114–164 h** | **~3–5 calendar weeks** |

*Note: Hours represent focused technical work including edge-case debugging, verification, and empirical testing. First usable release (Stages 1–4) is achieved at ~66–98 focused hours.*
