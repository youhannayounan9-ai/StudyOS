# StudyOS — Staged Development Roadmap & Execution Plan

- **Document Version:** 1.0.0
- **Status:** Approved Baseline (Committed at 1827f1a) | Stage 1 Complete & Approved (Committed at 84db803) | Stage 2 Implementation in review
- **System Target:** Local-First Personalized Learning Operating System
- **Hardware Profile:** Windows 11 | NVIDIA RTX 5060 Laptop GPU (8 GB VRAM) | 16 GB System RAM

---

## 1. Roadmap Philosophy & Stage-Gate Principles

StudyOS follows a disciplined, stage-gated engineering methodology:
1. **Incremental Completeness:** Each stage produces a testable, verifiable, and working artifact. No stage begins until the previous stage satisfies its defined acceptance criteria.
2. **First Usable Release Prioritization:** Stages 0 through 4 deliver the foundational, fully functional local study loop (Ingest $\rightarrow$ Retrieve $\rightarrow$ Tutor with Citations $\rightarrow$ Exam & Grade). Stages 5 through 7 enrich the cognitive modeling, adaptation, and system hardening.
3. **Realistic Verification Time:** Hour estimates explicitly include debugging on Windows, edge-case testing, citation auditing, and learning curves.
4. **No Premature Optimization:** Architectural modularity is established early, but complex features (e.g., cross-encoders, multimodal vision, external research) are gated behind stable core baselines.
5. **Explicit Git Commit Checkpoints:** Work for each stage is reviewed and verified in the local working tree before being committed as a permanent baseline. Stage 0 documentation was approved and committed as baseline commit `1827f1a`; Stage 1 ingestion engine is approved and committed as baseline commit `84db803`; Stage 2 hybrid retrieval is currently implemented and under review.

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
- **Scope:** Build the local model inference wrapper (`studyos.inference`) connecting to Ollama (`qwen2.5:7b` / `llama3:8b`) with pre-flight connectivity checks. Implement the evidence-grounded tutoring service (`studyos.tutoring`). Build prompt templates that enclose untrusted text in `<evidence_context>` delimiters, assign temporary evidence markers (`[E1]`, `[E2]`), and resolve validated markers to trusted database metadata. Build initial Streamlit UI for course management, document selection, and tutor chat.
- **Explicit Non-Goals:** Exam generation, grading rubrics, automated learner modeling.
- **Expected Deliverables:**
  - `studyos.inference.ollama`: Ollama API client with pre-flight health checks, streaming, and error handling.
  - `studyos.tutoring.tutor`: RAG tutoring orchestrator with untrusted content isolation, evidence ID assignment, and resolution.
  - `studyos.ui.pages.tutor`: Streamlit interactive chat interface with collapsible citation viewer.
  - `tests/integration/test_tutor_grounding.py`: Automated tests asserting refusal on out-of-domain queries and rejection of invalid evidence markers.
- **Prerequisites:** Stage 2 complete; Ollama running with 7B/8B model.
- **Risks:**
  - Model hallucinating citations not present in retrieved context.
  - High generation latency if VRAM threshold is breached.
- **Verification Commands & Procedures:**
  - Grounding test: `pytest tests/integration/test_tutor_grounding.py -v`.
  - UI interactive test: `streamlit run app.py` $\rightarrow$ select 2 documents, ask technical questions, inspect citation drawers.
- **Acceptance Criteria:**
  - Tutor answers reference valid application evidence markers (`[E1]`, `[E2]`) that resolve to verified source chunks from the selected document set; any unmapped or hallucinated marker is rejected.
  - When asked about a concept not present in selected slides, tutor explicitly outputs the standard refusal message.
  - Time to first token $\le 2.5$ seconds on RTX 5060; peak VRAM $\le 6.5$ GB.
- **Estimated Focused Hours:** 16–24 hours (Uncertainty: $\pm 25\%$).
- **Dependencies:** Stage 2.

---

### Stage 4: Exam Generation, Submission & Automated Grading Engine
- **Scope:** Implement the assessment engine. Enable students to configure exams (question count, difficulty, target documents). Generate Multiple Choice Questions (MCQ) and Short-Answer Questions using constrained JSON output from Ollama. Implement secure backend storage of answer keys (preventing DOM leakage). Build the interactive exam UI, scoring engine, automated explanations, and mistake recording.
- **Explicit Non-Goals:** Spaced repetition scheduling, cross-course mastery decay modeling.
- **Expected Deliverables:**
  - `studyos.assessment.generator`: Exam generator with schema validation.
  - `studyos.assessment.grader`: Deterministic MCQ grader and LLM rubric evaluator for short answers.
  - `studyos.ui.pages.exam`: Streamlit exam builder and live quiz interface.
  - SQLite schema updates: `exam_definitions`, `exam_questions`, `exam_attempts`, `question_attempts`.
  - Automated tests: `tests/unit/test_exam_generator.py`, `tests/unit/test_grader.py`.
- **Prerequisites:** Stage 3 complete; reliable JSON mode inference from local model.
- **Risks:**
  - LLM emitting malformed JSON or invalid schema for questions.
  - Plausibility of distractors being too obvious or ambiguous.
- **Verification Commands & Procedures:**
  - Run exam test suite: `pytest tests/unit/test_exam_generator.py tests/unit/test_grader.py -v`.
  - End-to-end exam flow in UI: Generate 5-question MCQ exam $\rightarrow$ verify answer keys absent from frontend state $\rightarrow$ submit answers $\rightarrow$ verify score and citation feedback.
- **Acceptance Criteria:**
  - 5-question exam generates in under 60 seconds with 100% valid JSON parsing.
  - Correct answers are inaccessible prior to submission.
  - Scoring calculates accurately, persists to SQLite, and displays source citations for every question.
- **Estimated Focused Hours:** 18–26 hours (Uncertainty: $\pm 25\%$).
- **Dependencies:** Stage 3.

---

### Stage 5: Persistent Learner Model & Mistake Analytics
- **Scope:** Build the persistent student cognitive model in SQLite (`concept_mastery`, `mistake_log`). Implement algorithms to estimate topic mastery from question attempts, track historical error trends, and calculate uncertainty intervals. Build the Learner Dashboard in Streamlit with visual progress indicators, mastery heatmaps, and a one-click "Remediation Exam" generator targeting past mistakes.
- **Explicit Non-Goals:** External web research, multi-user analytics.
- **Expected Deliverables:**
  - `studyos.learner.mastery`: Mastery calculation service (exponential moving average / BKT).
  - `studyos.learner.remediation`: Generator for targeted weakness quizzes.
  - `studyos.ui.pages.analytics`: Streamlit mastery dashboard with charts and mistake logs.
  - Automated tests: `tests/unit/test_learner_model.py`.
- **Prerequisites:** Stage 4 complete; historical exam attempt records.
- **Risks:**
  - Mastery score volatility if too few questions are attempted.
  - Unclear topic tagging in source documents.
- **Verification Commands & Procedures:**
  - Run mastery unit tests: `pytest tests/unit/test_learner_model.py -v`.
  - Simulate quiz failures in test script $\rightarrow$ verify mastery score drops for targeted topic and remediation quiz pulls failed questions.
- **Acceptance Criteria:**
  - Topic mastery updates deterministically after every exam submission.
  - System successfully generates a remediation quiz consisting solely of previously missed concepts.
  - Application restart preserves all mastery levels and mistake logs identically.
- **Estimated Focused Hours:** 14–20 hours (Uncertainty: $\pm 20\%$).
- **Dependencies:** Stage 4.

---

### Stage 6: Adaptive Learning, Spaced Repetition & External Research
- **Scope:** Implement spaced repetition scheduling (e.g., modified SM-2 or FSRS algorithm) calculating memory decay and review intervals. Support adaptive exam difficulty (raising question complexity as student mastery increases). Implement an *optional*, strictly controlled external research mode (retrieving academic definitions via Wikipedia / arXiv APIs to clarify prerequisites, clearly partitioned from primary course evidence).
- **Explicit Non-Goals:** Unbounded internet browsing, social sharing features.
- **Expected Deliverables:**
  - `studyos.learner.spaced_repetition`: Retention decay scheduler.
  - `studyos.research.external`: Controlled external documentation fetcher with clear source partitioning.
  - UI notifications for topics due for review.
  - Automated tests: `tests/unit/test_spaced_repetition.py`.
- **Prerequisites:** Stage 5 complete.
- **Risks:**
  - External research polluting course-specific definitions.
- **Verification Commands & Procedures:**
  - Test decay calculations: `pytest tests/unit/test_spaced_repetition.py -v`.
  - Verify external research sources are visually demarcated with distinct warning badges in UI.
- **Acceptance Criteria:**
  - Due review dates are calculated accurately based on past recall scores.
  - External research is explicitly opt-in and visually separated from official course citations.
- **Estimated Focused Hours:** 16–22 hours (Uncertainty: $\pm 30\%$).
- **Dependencies:** Stage 5.

---

### Stage 7: Comprehensive Evaluation, Security, Performance & Portfolio Polish
- **Scope:** Build a comprehensive evaluation harness with a golden benchmark dataset of real CS/DS course materials. Quantify citation precision, recall, and answer faithfulness. Conduct security hardening (sanitize logs, validate file upload paths, test directory traversal guards). Optimize VRAM usage and database indexing. Polish UI styling, documentation, and create developer setup scripts for portfolio presentation.
- **Explicit Non-Goals:** Cloud commercialization.
- **Expected Deliverables:**
  - `benchmarks/`: Golden dataset of 50 course questions, ground truth citations, and evaluation runner.
  - `docs/BENCHMARK_REPORT.md`: Empirical evaluation results.
  - Performance audit report (VRAM, memory, latency).
  - Security review documentation.
  - One-click setup script (`setup.ps1` using `uv`).
- **Prerequisites:** Stages 1–6 complete.
- **Risks:**
  - Benchmarks revealing lower than expected citation precision requiring prompt tuning.
- **Verification Commands & Procedures:**
  - Run full benchmark suite: `python benchmarks/run_eval.py`.
  - Run end-to-end test suite: `pytest tests/ -v --cov=studyos`.
  - Full clean install test on clean machine/directory.
- **Acceptance Criteria:**
  - Full automated test suite passes with $\ge 80\%$ code coverage.
  - Golden benchmark achieves $\ge 85\%$ citation precision and $\ge 90\%$ answer grounding.
  - End-to-end memory and VRAM stay strictly within the documented budgets.
- **Estimated Focused Hours:** 20–28 hours (Uncertainty: $\pm 25\%$).
- **Dependencies:** Stages 1–6.

---

## 3. Milestone Summary & Time Estimates

| Stage | Focus Area | Deliverable Milestone | Focused Hours (Est.) | Cumulative Hours |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Architecture & Specification** | Specs, Architecture, Roadmap docs | **6–10 h** *(Done)* | 6–10 h |
| **1** | **Document Ingestion & Storage** | Parsers, Chunking, SQLite DB | **14–20 h** | 20–30 h |
| **2** | **Hybrid Retrieval Engine** | ChromaDB, FTS5 BM25, RRF Fusion | **12–18 h** | 32–48 h |
| **3** | **AI Tutor with Citations** | Ollama 7B/8B, Grounding, Streamlit UI | **16–24 h** | 48–72 h |
| **4** | **Assessment & Exam Engine** | Quiz Gen, Leak Prevention, Grading | **18–26 h** | 66–98 h |
| **5** | **Persistent Learner Model** | Mastery Tracking, Mistake Remediation | **14–20 h** | 80–118 h |
| **6** | **Adaptive Learning & Research** | Spaced Repetition, Scaffolding | **16–22 h** | 96–140 h |
| **7** | **Evaluation & Portfolio Polish** | Benchmarks, Security, Coverage | **20–28 h** | 116–168 h |
| **Total** | **Full System Lifecycle** | **Production-Grade Personal StudyOS** | **116–168 h** | **~3–5 calendar weeks** |

*Note: Hours represent focused technical work including edge-case debugging, verification, and empirical testing. First usable release (Stages 1–4) is achieved at ~66–98 focused hours.*
