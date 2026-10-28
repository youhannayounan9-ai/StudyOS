# StudyOS — Product Specification and Requirements Baseline

- **Document Version:** 1.2.0
- **Status:** Approved Baseline (Committed at 1827f1a) | Stage 1 Complete & Approved (Committed at 84db803) | Stage 2 Complete & Approved (Committed at 7026c4c) | Stage 3 Complete & Approved (Committed at 3d9e26e) | Stage 4 Complete & Approved (Committed at 7351c60) | Stage 5 Complete & Approved (Committed at 73d49bf) | Stage 6 Implementation in review
- **System Target:** Local-First Personalized Learning Operating System
- **Hardware Profile:** Windows 11 | NVIDIA RTX 5060 Laptop GPU (8 GB VRAM) | 16 GB System RAM

---

## 1. Executive Summary & Product Vision

### 1.1 The Problem
Computer science and data science students accumulate hundreds of complex technical documents throughout their academic curriculum: lecture slide decks (PPTX/PDF), textbook chapters (PDF), lab assignments (DOCX/PDF), code walkthroughs, and past exams. Typical generic "Chat with PDF" tools fail students in critical ways:
1. **Shallow Context & Amnesia:** They treat queries as isolated chats, forgetting past mistakes, misunderstood prerequisites, and student skill progression across sessions.
2. **Hallucination & Vagueness:** They generate plausible-sounding explanations without verifiable page- or slide-level evidence, making them dangerous for exam revision.
3. **Passive Interaction:** They encourage passive reading rather than active recall, diagnostic self-testing, and targeted weakness remediation.
4. **Cloud/Cost Enclosure:** Most systems require paid API tokens, exposing personal academic materials to external servers and incurring recurring costs.

### 1.2 The StudyOS Vision
**StudyOS** is a local-first, privacy-preserving, AI-powered personal learning operating system. It transforms passive course materials into an active, personalized cognitive tutor and examination engine running entirely on consumer hardware.

StudyOS is explicitly **not** a generic chatbot. Its architectural mission is to provide:
- **Strict Evidence Grounding:** Answers, explanations, and exam questions are tied to verifiable citations (document, section, page, or slide number). If course material does not contain the answer, the system explicitly refuses to guess.
- **Diagnostic Assessment:** Dynamic exam generation with automated grading, multi-faceted rubrics, and source-linked answer keys that prevent answer leakage before submission.
- **Continuous Learner Modeling:** A persistent student knowledge graph that tracks concept mastery, logs recurring misconceptions, models memory decay, and schedules targeted revision.
- **Zero Cloud Dependence:** 100% local execution using open-weight language models and dense embeddings via local accelerators, ensuring complete data privacy and zero API bills.

---

## 2. Target User, Personas & Environment Profile

### 2.1 Primary Persona: The CS/DS Undergraduate & Independent Scholar
- **Profile:** Third- or fourth-year undergraduate or self-directed graduate student studying Algorithms, Computer Systems, Machine Learning, and Database Architecture.
- **Work Habits:** Ingests 50–150 slides per week per course, along with reference manuals and academic papers; reviews past exam papers under timed conditions; needs conceptual analogies combined with rigorous mathematical or code-level explanations.
- **Pain Points:**
  - Difficulty locating which specific lecture covered a nuanced theoretical point.
  - Overconfidence when reading slides passively, revealed only when failing past exam questions.
  - Lack of immediate, high-fidelity feedback on open-ended diagnostic questions.

### 2.2 Operational Hardware & Software Baseline
Empirically verified on the primary host machine via read-only diagnostic commands:
- **Operating System:** Microsoft Windows 11 Pro (Version 10.0.26300, Build 26300) (Verified).
- **GPU Accelerator:** NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB (~8 GB) dedicated VRAM, Driver 610.88 (Verified).
- **Host Memory:** 15.73 GiB (~16 GB) physical system RAM (Verified).
- **Installed Runtimes & Tooling:**
  - Python 3.13.13 (Global CPython) & Python 3.14.3 (Astral managed) verified installed.
  - Python 3.12 is the intended project development target to be provisioned via `uv` in Stage 1; its availability in the active environment is currently unverified.
  - `uv` package manager v0.10.10 verified installed.
  - Ollama client v0.40.7 verified installed; background daemon service is currently inactive/stopped.
  - Cached local model manifests: `all-minilm:latest`, `qwen2.5:7b`, `llama3:latest`, `llava:latest` verified present in disk storage; runtime execution of these models has not yet been tested in this environment.

---

## 3. Scope, Assumptions, Non-Goals, and Failure Conditions

### 3.1 Assumptions & Core Guarantees
1. **Single-User Local Execution:** The application is deployed locally for a single student on their personal workstation; multi-tenant security and user account isolation are out of scope.
2. **Text-Extractable Formats:** The vast majority of college PDFs contain native digital text layers or embedded font encodings.
3. **Local Inference Capacity:** Quantized 7B/8B parameter models (e.g., Qwen 2.5 7B Q4_K_M or Llama 3 8B Q4_K_M) fit comfortably within the 8 GB VRAM budget alongside dense embedding generation when loaded sequentially or bounded carefully.
4. **Offline & Local-Only Guarantees:** Core workflows (ingestion, retrieval, tutoring, examination, mastery tracking) do not depend on external network services and make zero external network requests. Local HTTP loopback communication (e.g., `http://127.0.0.1:11434` for Ollama or port 8501 for Streamlit) is standard and explicitly distinguished from external internet access.

### 3.2 Explicit Non-Goals (Scope Boundaries)
- **Non-Goal 1: Cloud-Hosted Multi-Tenant SaaS.** No cloud databases, auth tokens, external telemetry, or SaaS subscription tiers.
- **Non-Goal 2: Paid Commercial API Integration.** The core system will never require OpenAI, Anthropic, or proprietary cloud endpoints to function.
- **Non-Goal 3: Generic Unbounded Web Agent.** The tutor will not autonomously crawl the web or retrieve unvetted third-party blog posts during core study sessions. An optional external research mode in Stage 6 will make external HTTP queries only when explicitly toggled on by the user.
- **Non-Goal 4: High-Throughput Scanned Document OCR in v1.** Low-quality cell-phone photos of handwritten notes or scanned bitmap PDFs without OCR layers are unsupported in Stage 1 and will trigger a clean warning.
- **Non-Goal 5: Multi-Agent Autonomous Swarms.** No redundant multi-agent negotiation frameworks (e.g., CrewAI, AutoGen) that burn VRAM and introduce unpredictable latency. All workflows use deterministic application orchestration.

### 3.3 System Limitations & Failure Conditions
1. **GPU Out-of-Memory (OOM):** If an external process consumes VRAM while Ollama runs an 8B model with an 8k context window, execution will degrade to system RAM CPU offload. The system must monitor VRAM and set conservative context caps (e.g., 4k–8k tokens).
2. **Scanned / Image-Only PDFs:** If an ingested PDF yields zero characters across all pages, extraction must fail gracefully with `ExtractionEmptyError`, notifying the user that OCR is required.
3. **Ungrounded Retrieval Context:** When a query targets a topic not present in the indexed documents, semantic search may return low-similarity chunks. The system must detect similarity below an empirical threshold and explicitly declare: *"The selected course documents do not contain sufficient evidence to answer this question."*
4. **Grading Discrepancies on Open-Ended Questions:** Subjective student answers may use valid synonyms or non-standard notation not reflected in the generated rubric. The student must always have the ability to contest an automated mark or view the full rubric and source citations.

### 3.4 Untrusted Document Content & Multi-Layered Prompt Injection Defense
Uploaded documents, slide texts, and retrieved passages are untrusted external inputs. While structural delimiters (e.g., `<evidence_context>...</evidence_context>`) are used during evidence assembly to separate source text from instructions, delimiters alone do not provide a complete security guarantee against prompt injection. StudyOS implements a multi-layered defense:
1. **Structural Delimitation:** Passages are bounded in tagged containers to demarcate evidence from prompt directives.
2. **Instruction Precedence:** System prompts explicitly instruct the model that enclosed document text is reference data only and cannot alter operational guidelines, modify answer keys, or grant permissions.
3. **Restricted Output Syntax:** The model is constrained to referencing only assigned evidence markers (`[E1]`, `[E2]`).
4. **Application-Level Token Validation:** Free-form citations in model output are ignored; only validated tokens matching active retrieved evidence resolve to trusted database citations.

---

## 4. Functional Requirements Analysis & Prioritization

Requirements are categorized into:
- **P0 (Must-Have for First Usable Release):** The minimal viable learning loop (multi-format document ingestion, hybrid vector + FTS5 retrieval, grounded tutoring with verifiable citations, dynamic quiz generation, and persistent basic learning history).
- **P1 (Core Personalized Learning Experience):** Open-ended conceptual exams, automated rubric grading, topic mastery tracking, and weakness remediation quizzes.
- **P2 (Advanced / Future Enhancements):** Multimodal image extraction from slides, spaced repetition scheduler, adaptive study roadmaps, and optional external academic research.

---

### Category A: Document Ingestion & Validation

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-A1** | **P0** | **Multi-Format Ingestion:** Ingest PDF, DOCX, PPTX, and UTF-8 TXT files into the system. | **AC-A1:** Given a valid PDF, DOCX, PPTX, or TXT file up to 100 MB, the system parses the document, creates a unique document record with SHA-256 hash, and extracts text without crashing. |
| **REQ-A2** | **P0** | **Structural Metadata Extraction:** Retain page numbers (PDF), slide indices & titles (PPTX), and section headings (DOCX/TXT). | **AC-A2:** Every extracted text chunk carries metadata fields `document_id`, `source_filename`, `chunk_index`, and `location` (e.g., `{"page": 14}` or `{"slide": 8, "title": "Dynamic Programming"}`). |
| **REQ-A3** | **P0** | **Duplicate & Collision Prevention:** Prevent duplicate ingestion of identical files and handle re-uploads cleanly. | **AC-A3:** Uploading a file with a matching SHA-256 hash detects the existing record, alerts the user, and skips duplicate processing unless "Force Re-index" is chosen. |
| **REQ-A4** | **P0** | **Empty / Corrupt File Validation:** Detect empty, password-protected, or unreadable files and report actionable errors. | **AC-A4:** Uploading a password-protected PDF or a 0-byte file immediately returns a descriptive error message without crashing the ingestion worker. |
| **REQ-A5** | **P1** | **Scanned PDF Fallback Detection:** Detect PDFs with bitmap pages lacking embedded text and tag them as requiring OCR. | Emits `ScannedDocumentWarning` when average character count per page is `< 50`. |
| **REQ-A6** | **P2** | **Multimodal Slide Extraction:** Extract embedded diagrams, charts, and figures from slides and associate them with neighboring text chunks. | Stored image artifacts linked to slide chunk IDs for vision-model inspection. |

---

### Category B: Course & Topic Organization

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-B1** | **P0** | **Course Hierarchy Management:** Create, rename, view, and delete courses (e.g., "CS301: Algorithms", "DS202: Linear Algebra"). | **AC-B1:** User can create a course, assign documents to it, and list all courses and their constituent files via the UI. |
| **REQ-B2** | **P0** | **Document Selection Scope:** Enable user to select one, multiple, or all documents within a course as the active context for study. | **AC-B2:** Queries and exam generation only retrieve chunks matching the user's selected `document_id` set; chunks from unselected documents are strictly excluded. |
| **REQ-B3** | **P0** | **Course Boundary Isolation:** Guarantee strict isolation between courses to prevent topic contamination. | **AC-B3:** Querying "Operating Systems" documents will never return results from "Database Systems" chunks even if both courses are indexed in the vector store. |
| **REQ-B4** | **P1** | **Topic Tagging & Hierarchy:** Allow assigning granular topic tags (e.g., "Graph Theory -> Dijkstra") to documents or individual chapters. | Filter retrieval and exams by specific topic tags within a course. |
| **REQ-B5** | **P2** | **Course Prerequisite Graph:** Define dependency relationships between courses and topics (e.g., "Calculus I" is prerequisite for "Machine Learning"). | Warns student when asking advanced topics if foundational course units show low mastery. |

---

### Category C: Retrieval & Evidence Grounding

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-C1** | **P0** | **Structure-Aware Chunking:** Split documents into semantic passages (300–600 tokens) with configurable overlap (50–100 tokens), preserving sentence boundaries and slide/page headers. | **AC-C1:** Text chunks never break mid-sentence; each chunk includes a prefix header indicating the document title and slide/page number. |
| **REQ-C2** | **P0** | **Local Embedding Generation:** Generate dense vector embeddings locally using a resource-efficient model (e.g., `all-minilm` or `bge-small-en-v1.5`). | **AC-C2:** Ingestion generates 384-dimensional dense vectors stored locally in ChromaDB without any remote HTTP calls. |
| **REQ-C3** | **P0** | **Application-Grounded Source Attribution:** Each retrieved evidence chunk carries an application-assigned evidence marker (e.g., `[E1]`, `[E2]`). The model must reference only provided markers. The application resolves validated markers into citations using database metadata (`Document Title`, `Page/Slide`, `Verbatim Chunk`). Any unmapped, unselected, or hallucinated marker is rejected. | **AC-C3:** The UI displays citations resolved exclusively from application-verified evidence IDs mapped to SQLite chunk records; clicking a citation exposes the exact verbatim passage. Any citation ID not in the retrieved set is rejected. |
| **REQ-C4** | **P0** | **Refusal on Unsupported Queries:** When the retrieved context contains zero chunks above an acceptable relevance score, refuse to fabricate answers. | **AC-C4:** Querying a topic completely absent from the selected documents outputs an explicit grounding refusal statement rather than an ungrounded hallucination. |
| **REQ-C5** | **P0** | **Hybrid Vector + Lexical Search:** Combine dense vector search with SQLite FTS5 (BM25) full-text keyword search using Reciprocal Rank Fusion (RRF). | **AC-C5:** Given a search query targeting an exact technical acronym, variable name, or algorithmic term, the hybrid retrieval engine returns the matching chunk in the top-3 results within 500 ms, even if dense semantic similarity alone ranks it lower. |
| **REQ-C6** | **P2** | **Contextual Compression & Re-ranking:** Re-rank top-20 retrieved candidates using a local cross-encoder model to select top-5 most relevant chunks. | Improves retrieval precision for complex multi-hop academic questions. |

---

### Category D: Personalized Tutoring & Conceptual Scaffolding

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-D1** | **P0** | **Evidence-Backed Answering:** Answer student questions strictly using retrieved course context, synthesizing complex concepts into clear academic prose. | **AC-D1:** Given a conceptual question on selected slides, the tutor outputs a structured explanation citing evidence markers (`[E1]`, `[E2]`) that resolve to verified source chunks from the selected document set. |
| **REQ-D2** | **P0** | **Concrete Examples & Code Analogies:** Generate illustrative examples, code snippets, or analogies tailored to computer science and data science pedagogy. | **AC-D2:** Prompts instruct the LLM to provide at least one practical example or pseudo-code demonstration for algorithmic questions. |
| **REQ-D3** | **P1** | **Configurable Explanation Depth:** Support toggleable explanation levels: "Quick Intuition", "Standard Academic Review", and "Deep Theoretical Derivation". | Adjusts prompt complexity and token budget based on student's current study goal. |
| **REQ-D4** | **P1** | **Socratic Follow-Up Questions:** Conclude explanations with 1–2 diagnostic check-for-understanding questions based on the material just explained. | Prompts active recall immediately after delivering an explanation. |
| **REQ-D5** | **P2** | **Prerequisite Diagnostic Probing:** If a student struggles with a concept (e.g., Backpropagation), detect missing prerequisites (e.g., Chain Rule) and offer scaffolding. | Automatically pulls reference chunks from earlier foundational lectures. |

---

### Category E: Assessment & Exam Engine

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-E1** | **P0** | **Configurable Exam Generation:** Generate customized quizzes specifying question count (e.g., 5, 10, 20) and target documents. | **AC-E1:** User selects 2 documents, requests a 5-question multiple-choice quiz; system generates 5 valid, non-redundant questions grounded in those documents within 60 seconds. |
| **REQ-E2** | **P0** | **Multiple-Choice Questions (MCQ):** Generate 4-option MCQs with plausible distractors, a single correct answer, and source citation. | **AC-E2:** Generated MCQs parse cleanly into structured JSON with fields `question`, `options` (A–D), `correct_option`, `explanation`, and `source_chunk_id`. |
| **REQ-E3** | **P0** | **Answer Leakage Prevention:** Keep answer keys and explanations strictly isolated on the backend until student submits their response. | **AC-E3:** The quiz UI never transmits or exposes the correct answer or explanation in the DOM or frontend state prior to user submission. |
| **REQ-E4** | **P0** | **Immediate Scoring & Explanations:** Upon exam submission, score responses, display percentage/points, reveal correct answers, and provide source evidence. | **AC-E4:** Submitting answers displays immediate score calculation, identifies incorrect choices, and cites the exact slide/page explaining the right answer. |
| **REQ-E5** | **P1** | **Short-Answer & Conceptual Questions:** Support open-ended conceptual and coding questions evaluated via a multi-criteria rubric. | Evaluates student text responses using LLM rubric comparison against gold reference answers. |
| **REQ-E6** | **P1** | **Difficulty Calibration:** Support configurable difficulty settings: "Foundational / Recall", "Application / Analysis", "Challenging / Exam-Level". | Calibrates Bloom's taxonomy level in prompt constraints. |
| **REQ-E7** | **P2** | **Timed Mock Exam Mode:** Full-length simulated exam mode with countdown timer, question flagging, review screen, and comprehensive post-exam report. | Simulates actual college final exam constraints. |

---

### Category F: Persistent Learner Model & Analytics

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-F1** | **P0** | **Relational Learner Profile & Activity History:** Persist local learner profile, chronological activity events, and study history in local SQLite WAL storage with application-generated IDs and foreign keys. | **AC-F1:** Closing and reopening restores learner state; study events (exam submissions, reviews, manual grades) are logged chronologically; repeated processing of the same attempt is idempotent and does not duplicate events. |
| **REQ-F2** | **P0** | **Course-Scoped Concept Catalog & Grounding:** Maintain normalized, course-scoped concepts linked to assessment questions and source chunks, preventing cross-course collision and tracking label provenance. | **AC-F2:** Concepts with identical names across different courses remain strictly isolated; concepts link to questions and chunks; automated extraction uses local chat with bounded validation; stale chunk references are marked without corrupting history. |
| **REQ-F3** | **P0** | **Explainable Concept Mastery Estimation:** Deterministic Laplace-smoothed accuracy $(k+1)/(n+2)$ where $k = \sum p_i$ represents normalized earned-credit successes across $n$ observations, with uniform Beta(1,1) prior and sample-size uncertainty tracking, categorizing concepts into `unassessed`, `needs_evidence`, `weak`, `moderate`, and `strong`. | **AC-F3:** Zero graded observations yields `unassessed`; pending manual-review answers never penalize mastery; corrected grades safely supersede earlier states; output communicates evidence counts and recency. |
| **REQ-F4** | **P1** | **Spaced Repetition Review Scheduling:** Bounded offline interval policy (1 to 180 days, ease factor 1.3 to 3.0) calculating next due dates from graded performance and recorded review outcomes without cloud services. | **AC-F4:** Reschedules dynamically on correct/incorrect/partial outcomes; distinguishes due, overdue, and scheduled states; recording a review updates schedule idempotently; review queue preserves course and concept isolation. |
| **REQ-F5** | **P1** | **Explainable Progress Summaries & Recommendations:** Rule-based progress summaries and ranked study recommendations with explicit reasons (overdue review, weak accuracy, insufficient evidence) without LLM hallucinations. | **AC-F5:** Summary queries support course filtering; recommendations explain specific underlying performance reasons; pending attempts are not presented as final outcomes. |

---

### Category G: Privacy, Reliability & Data Integrity

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-G1** | **P0** | **100% Local Core Operation:** Store all vector embeddings, relational data, and cached documents on the local filesystem. Zero external outbound internet calls are made during core document ingestion, query retrieval, tutoring, or quiz evaluation. Local loopback HTTP calls (`127.0.0.1`) to Ollama are isolated from external networks. | **AC-G1:** Zero external outbound network calls are made during document ingestion, query retrieval, tutoring, or quiz evaluation. |
| **REQ-G2** | **P0** | **Clean Document & Course Deletion:** Deleting a document or course cascades cleanly across SQLite records and ChromaDB vector embeddings. | **AC-G2:** Deleting a course removes all associated document rows, chunks, quiz history, and vector embeddings without leaving orphaned data. |
| **REQ-G3** | **P0** | **Sanitized Logging:** Application logs record operational events (errors, latency, token counts) without leaking raw document text or exam answers. | **AC-G3:** Log files in `logs/` contain zero verbatim student document text or sensitive personal data. |
| **REQ-G4** | **P0** | **Atomic State Transitions:** Ingestion and exam grading use database transactions so that interrupted jobs do not leave corrupted partial state. | **AC-G4:** Killing the application mid-ingestion results in a clean rollback; the document is marked `failed` or removed, preventing broken indexing. |
| **REQ-G5** | **P1** | **Local Database Backup & Export:** One-click backup of SQLite database and vector embeddings to a portable archive. | Generates timestamped `.zip` containing database snapshot and vector index. |

---

### Category H: Evaluation & Quality Assurance

| ID | Priority | Requirement Description | Acceptance Criteria (P0) / Target Behavior |
| :--- | :---: | :--- | :--- |
| **REQ-H1** | **P0** | **Deterministic Parsing Regression Suite:** Automated unit tests verifying extraction accuracy across standard sample PDF, DOCX, PPTX, and TXT files. | **AC-H1:** Running `pytest tests/unit/test_parsers.py` passes 100% on a fixed set of synthetic fixture files. |
| **REQ-H2** | **P0** | **Schema & Migration Verification:** Automated tests verifying that SQLite schema initializes cleanly, enforces foreign keys, and executes CRUD operations. | **AC-H2:** `pytest tests/unit/test_db.py` passes cleanly on fresh and populated databases. |
| **REQ-H3** | **P1** | **Citation Precision Benchmark:** Evaluation harness calculating citation correctness against a curated golden dataset of course questions and ground-truth passages. | Measures percentage of generated citations that match true source chunks (Target: $\ge 85\%$). |
| **REQ-H4** | **P1** | **Faithfulness & Grounding Verification:** Automated test detecting whether the tutor refuses to answer when supplied with ungrounded distractor context. | Asserts that out-of-domain questions trigger standard refusal message. |
| **REQ-H5** | **P2** | **Automated Exam Quality Rubric:** LLM-assisted evaluation of generated quiz questions for ambiguity, grammatical clarity, and distractor plausibility. | Automated checks run on candidate questions before presenting to user. |

---

## 5. Non-Functional Requirements (NFRs)

### 5.1 Performance & Resource Efficiency
- **VRAM Ceiling:** Application + Ollama must stay under 7.0 GB VRAM total under peak load, leaving $\ge 1.0$ GB for OS/desktop compositing on the 8 GB RTX 5060.
- **Host RAM Budget:** System RAM consumption must not exceed 4.0 GB for StudyOS processes (ChromaDB, SQLite, Streamlit), leaving $\ge 8.0$ GB for Windows and background tasks on 16 GB RAM.
- **Retrieval Latency:** Hybrid vector + keyword search over a library of 10,000 chunks must return top-10 candidates in under 500 ms.
- **Inference Latency:** Time to first token for local 7B model must be $\le 2.5$ seconds; sustained generation speed must achieve $\ge 20$ tokens/second on RTX 5060.

### 5.2 Reliability & Fault Tolerance
- **Deterministic State:** Relational state (courses, attempts, scores, mastery) is exclusively managed in SQLite with Write-Ahead Logging (`PRAGMA journal_mode=WAL`), preventing database locking on concurrent reads.
- **Graceful Model Disconnect:** If the Ollama background daemon crashes or is unresponsive, the UI must display a clear status indicator ("Ollama Offline — Click to Reconnect") rather than an unhandled stack trace.

### 5.3 Maintainability & Code Standards
- **Zero Black-Box Orchestrators:** No reliance on opaque agent frameworks. All prompting, chunking, and database queries use plain, readable Python with explicit type annotations (`mypy` compliant).
- **Loose Coupling:** The language model interface is abstracted behind an `LLMProvider` protocol, allowing seamless switching between Ollama, llama.cpp, or vLLM without modifying business logic.
