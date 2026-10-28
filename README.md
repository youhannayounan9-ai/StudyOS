# StudyOS

> **A local-first, privacy-respecting, AI-powered personal learning operating system.**

StudyOS is an intelligent personal study environment designed for computer science and data science students. Unlike generic "chat with PDF" tools, StudyOS transforms passive course materials (lecture slides, textbook chapters, notes, and past exams) into an active, verifiable cognitive tutor and examination engine.

---

## Current Status: Stage 7 — Implementation in review (Production Hardening, Security, Reproducibility & Evaluation)

StudyOS is developed under a disciplined, incremental engineering roadmap. Stages 0 through 6 are complete, verified, and committed to Git history. Stage 7 (Production Hardening, Security, Reproducibility, CI & Deterministic Evaluation) is implemented and verified with 287 automated tests passing, currently undergoing review.

- [x] **Stage 0: Architecture & Specification** *(Approved & Baseline Committed at `1827f1a`)*
- [x] **Stage 1: Document Ingestion, Parsing & Normalization Engine** *(Approved & Baseline Committed at `84db803`)*
- [x] **Stage 2: Hybrid Retrieval & Evidence Grounding Engine** *(Approved & Baseline Committed at `7026c4c`)*
- [x] **Stage 3: AI Tutor & Grounded Interactive Answering** *(Approved & Baseline Committed at `3d9e26e`)*
- [x] **Stage 4: Exam Generation, Submission & Automated Grading Engine** *(Approved & Baseline Committed at `7351c60`)*
- [x] **Stage 5: Learner Model, Mastery Tracking & Personalized Revision** *(Approved & Baseline Committed at `73d49bf`)*
- [x] **Stage 6: Advanced Adaptive Learning, Knowledge Graph & External Research** *(Approved & Baseline Committed at `bac6d78`)*
- [ ] **Stage 7: Production Hardening, Security, Reproducibility & Evaluation** *(Implementation in review)*

*Note: In accordance with Stage 6 scope, StudyOS provides direct concept discovery from raw chunks, course-scoped prerequisite DAGs with deterministic cycle prevention, forgetting-aware review scheduling (local-fsrs-v1), a transparent adaptive study planner, and explicit opt-in external research with SSRF safety and citation validation, without cloud telemetry or data leakage.*

---

## Planned System Architecture & Features (Design Phase)

```
[Student Course Materials]
   (PDF, DOCX, PPTX, TXT)
            │
            ▼
[Ingestion & Normalization] ──► [Structure-Aware Chunking]
                                          │
            ┌─────────────────────────────┴────────────────────────────┐
            ▼                                                          ▼
  [ChromaDB Vector Store]                                  [SQLite FTS5 BM25 Engine]
   (Dense Semantic Search)                                   (Exact Keyword Search)
            │                                                          │
            └─────────────────────────────┬────────────────────────────┘
                                          ▼
                         [Reciprocal Rank Fusion (RRF)]
                                          │
                                          ▼
                           [Strict Evidence Grounding]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Cognitive Tutor Engine]                         [Assessment & Exam Engine]
    - Verifiable inline citations                   - Dynamic MCQ & conceptual exams
    - Analogies & code examples                     - Leak-proof answer key isolation
    - Refusal when unsupported                      - Multi-criteria rubric grading
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                           [Persistent Learner Model]
                            - SQLite WAL storage
                            - Topic mastery tracking (0.0 to 1.0)
                            - Mistake logging & remediation exams
```

---

## Target Hardware & Verified Environment Baseline

StudyOS is designed to run 100% locally on consumer hardware without paid APIs:

| Component | Target Specification | Verified Environment Status |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 | Verified: Microsoft Windows 11 Pro (Version 10.0.26300, Build 26300) |
| **GPU Accelerator** | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM) | Verified: 8,151 MiB VRAM (Driver 610.88) |
| **Host Memory** | 16 GB System RAM | Verified: 15.73 GiB Physical RAM |
| **Package Manager** | `uv` (Fast Python package & environment manager) | Verified: `uv 0.10.10` installed |
| **Python Runtimes** | Python 3.12 (Intended project target) | Verified installed: Python 3.13.13 & Python 3.14.3. Python 3.12 is the intended development target to be provisioned via `uv`. |
| **Local LLM Engine** | Ollama | Verified: Client v0.40.7 installed; cached model manifests (`qwen2.5:7b`, `all-minilm`, `llama3`, `llava`) present on disk. Daemon is currently inactive and model runtime execution has not yet been tested in this environment. |

---

## Project Documentation

Detailed architecture specifications, functional requirements, and execution plans are maintained in `docs/`:

1. [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) — Comprehensive product specification, user personas, prioritized requirements (P0/P1/P2) with verifiable acceptance criteria, assumptions, non-goals, and failure conditions.
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Deep architectural design, component boundaries, end-to-end data flows for 5 core scenarios, SQLite relational schema, technology evaluation matrix, and VRAM/RAM resource allocation budgets.
3. [docs/ROADMAP.md](docs/ROADMAP.md) — 8-stage execution roadmap with scopes, non-goals, deliverables, risk mitigations, verification procedures, acceptance criteria, and focused-hour estimates.

---

## Setup, Operations & Stage 2 Guide

### 1. Prerequisites & Environment Setup
- **Python**: 3.12 managed via `uv`.
- **Local Embedding Provider**: [Ollama](https://ollama.com/) with the `all-minilm:latest` (or `all-minilm`) model installed:
  ```powershell
  ollama pull all-minilm
  ollama serve
  ```
- **Virtual Environment**:
  ```powershell
  uv sync
  .\.venv\Scripts\activate
  ```

---

### 2. Configuration Keys & Defaults
All settings can be configured via environment variables or loaded through `studyos.core.config.AppConfig`:

| Setting | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :--- |
| Database Path | `STUDYOS_DB_PATH` | `studyos.db` | Path to the authoritative SQLite database |
| Data Directory | `STUDYOS_DATA_DIR` | `data` | Base directory for raw files and persistence |
| Chroma Directory | `STUDYOS_CHROMA_DIR` | `.chroma` | Directory for persistent ChromaDB vector index |
| Chroma Collection | `STUDYOS_CHROMA_COLLECTION` | `studyos_chunks` | Name of the Chroma collection |
| Ollama Base URL | `STUDYOS_OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for local Ollama HTTP API |
| Embedding Model | `STUDYOS_EMBEDDING_MODEL` | `all-minilm:latest` | Local embedding model name |
| Embedding Dimension | `STUDYOS_EMBEDDING_DIMENSION`| `384` | Expected vector dimension (validated on responses) |
| Embedding Timeout | `STUDYOS_EMBEDDING_TIMEOUT_SECONDS` | `30.0` | Request timeout for embedding calls in seconds |
| RRF Constant $k$ | `STUDYOS_RRF_K` | `60` | Rank smoothing parameter for Reciprocal Rank Fusion |
| Anonymized Telemetry | `STUDYOS_ANONYMIZED_TELEMETRY` | `False` | External telemetry toggle (always disabled locally) |
| Tutor Chat Model | `STUDYOS_TUTOR_MODEL` | `qwen2.5:7b` | Local Ollama chat model for evidence-grounded tutoring |
| Tutor Temperature | `STUDYOS_TUTOR_TEMPERATURE` | `0.2` | Sampling temperature for factual explanations |
| Tutor Max Tokens | `STUDYOS_TUTOR_MAX_TOKENS` | `1024` | Maximum tokens generated per answer |
| Tutor Timeout | `STUDYOS_TUTOR_TIMEOUT_SECONDS` | `60.0` | Request timeout for local chat inference in seconds |
| Tutor Evidence Limit | `STUDYOS_TUTOR_EVIDENCE_K` | `5` | Maximum evidence chunks provided in prompt context |
| Tutor Relevance Threshold | `STUDYOS_TUTOR_MIN_RELEVANCE` | `0.01` | Minimum RRF relevance score before abstaining |

---

### 3. Stage 2 Architecture & Data Flow
Stage 2 implements a local-first hybrid retrieval engine combining lexical search and dense semantic vector search:

```
                      User Search Query
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [Query Normalization]         [Ollama Embeddings]
       (Escape FTS5 syntax)          (all-minilm: 384-d)
               │                             │
               ▼                             ▼
      [SQLite FTS5 Search]         [ChromaDB Vector Search]
      - bm25(chunks_fts)            - HNSW cosine distance
      - SQL pre-filtering           - Metadata pre-filtering
      - Top-K lexical candidates    - Top-K semantic candidates
               │                             │
               └──────────────┬──────────────┘
                              ▼
                [Reciprocal Rank Fusion (RRF)]
                  RRF = Σ 1 / (60 + rank)
                  Deterministic tie-breaking
                              │
                              ▼
             [Authoritative SQLite Validation]
              - Discard stale Chroma records
              - Enforce course & doc isolation
              - Attach verified page/slide context
                              │
                              ▼
                Validated Evidence Chunks
```

- **SQLite is the Sole Source of Truth**: ChromaDB stores only secondary vectors and record IDs. Chunk text, parent document relationships, course assignments, and page/slide breadcrumbs are always retrieved directly from SQLite.
- **Course & Document Scope Isolation**: Both search components apply scope filters during candidate generation. Chunks returned from Chroma are subsequently validated against SQLite relational rows; mismatched or deleted chunks are discarded.

---

### 4. Operational Commands & Maintenance

#### A. Pre-Flight Ollama Health Check
Verify that the local Ollama daemon is reachable and the embedding model is loaded:
```powershell
.\.venv\Scripts\python.exe -m studyos.core.health
```

#### B. Vector Index Synchronization & Rebuild
Synchronize all SQLite chunks into ChromaDB (idempotent upsert, removes deleted chunks):
```powershell
.\.venv\Scripts\python.exe scripts/sync_index.py --sync
```
Check synchronization status and vector counts:
```powershell
.\.venv\Scripts\python.exe scripts/sync_index.py --status
```
Perform an explicit full rebuild (recreates collection from SQLite without deleting SQLite source data):
```powershell
.\.venv\Scripts\python.exe scripts/sync_index.py --rebuild
```

#### C. Testing Hybrid Retrieval
Run a hybrid search query with optional course and document filters:
```powershell
# Hybrid search across all courses
.\.venv\Scripts\python.exe scripts/test_retrieval.py --query "gradient descent" --top-k 5

# Scoped to a specific course
.\.venv\Scripts\python.exe scripts/test_retrieval.py --query "Dijkstra algorithm" --course "CS101" --top-k 3

# Lexical-only or semantic-only search
.\.venv\Scripts\python.exe scripts/test_retrieval.py --query "binary search" --mode lexical
.\.venv\Scripts\python.exe scripts/test_retrieval.py --query "binary search" --mode semantic

# Offline test using deterministic fake embeddings (no Ollama required)
.\.venv\Scripts\python.exe scripts/test_retrieval.py --query "database index" --fake-embeddings
```

#### D. Asking Questions with the Local AI Tutor
Query the AI Tutor using hybrid retrieval over your course materials, with validated source citations:
```powershell
# Ask a conceptual question grounded in uploaded materials
.\.venv\Scripts\python.exe scripts/ask_studyos.py --query "Explain Bellman-Ford algorithm"

# Filter question answering to a specific course
.\.venv\Scripts\python.exe scripts/ask_studyos.py --query "Explain asymptotic Big-O complexity" --course "CS101"

# Select explanation style (balanced, concise, or detailed)
.\.venv\Scripts\python.exe scripts/ask_studyos.py --query "How does virtual memory work?" --style detailed

# Offline test using deterministic fake chat and embeddings (no Ollama required)
.\.venv\Scripts\python.exe scripts/ask_studyos.py --query "Explain Dijkstra" --fake-chat --fake-embeddings
```

#### E. Generating, Taking, and Grading Practice Exams (`scripts/exams.py`)
StudyOS provides a complete, local-first assessment CLI without requiring a web interface:
```powershell
# 1. Generate an exam grounded in course materials
.\.venv\Scripts\python.exe scripts/exams.py generate --course "CS101" --count 5 --types mcq,true_false,short_answer --difficulty medium --title "CS101 Midterm Prep"

# Scoped to a specific topic or document
.\.venv\Scripts\python.exe scripts/exams.py generate --course "CS101" --topic "Graph Algorithms" --count 3 --types mcq

# 2. List saved exams
.\.venv\Scripts\python.exe scripts/exams.py list --course "CS101"

# 3. Take an exam (public view with ZERO answer-key leakage)
.\.venv\Scripts\python.exe scripts/exams.py take --exam-id "<EXAM_ID>"

# 4. Submit answers from a JSON file (unanswered questions receive 0.0 with clear feedback)
# Example answers JSON format: {"q_1": "A", "q_2": "True", "q_3": "Dynamic programming memoization"}
.\.venv\Scripts\python.exe scripts/exams.py submit --exam-id "<EXAM_ID>" --answers-file answers.json

# 5. View results, score breakdown, explanations, and verified source citations
.\.venv\Scripts\python.exe scripts/exams.py results --attempt-id "<ATTEMPT_ID>"

# Offline test using deterministic fake chat and embeddings (no Ollama required)
.\.venv\Scripts\python.exe scripts/exams.py generate --course "CS101" --count 2 --fake-chat --fake-embeddings
```

#### F. Inspecting Progress, Concepts, Reviews & Recommendations (`scripts/learner.py`)
StudyOS provides an operational, local-first learner model CLI:
```powershell
# 1. View overall learner progress or course-scoped progress
.\.venv\Scripts\python.exe scripts/learner.py progress
.\.venv\Scripts\python.exe scripts/learner.py progress --course "<COURSE_ID>"

# 2. List concepts and their mastery status
.\.venv\Scripts\python.exe scripts/learner.py concepts
.\.venv\Scripts\python.exe scripts/learner.py concepts --course "<COURSE_ID>"

# 3. View chronological study activity history
.\.venv\Scripts\python.exe scripts/learner.py history --limit 10
.\.venv\Scripts\python.exe scripts/learner.py history --course "<COURSE_ID>"

# 4. View scheduled, due, and overdue review tasks
.\.venv\Scripts\python.exe scripts/learner.py reviews
.\.venv\Scripts\python.exe scripts/learner.py reviews --overdue-only

# 5. Record a completed review outcome (ratings: good, pass, fail)
.\.venv\Scripts\python.exe scripts/learner.py record-review --concept-id "<CONCEPT_ID>" --rating good

# 6. View prioritized, explainable study recommendations
.\.venv\Scripts\python.exe scripts/learner.py recommendations
.\.venv\Scripts\python.exe scripts/learner.py recommendations --course "<COURSE_ID>"

# 7. Inspect why a specific concept is recommended
.\.venv\Scripts\python.exe scripts/learner.py why --concept-id "<CONCEPT_ID>"
```

---

### 5. Stage 3 Grounding Architecture & Citation Provenance

```
                     Student Question
                            │
                            ▼
               [Hybrid Retrieval Service]
           (RRF: SQLite FTS5 + ChromaDB Vectors)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      [No Chunks Found]          [Chunks Found]
               │                         │
               ▼                         ▼
       [Honest Refusal]        [Application-Assigned Markers]
      "The uploaded materials   Assign [E1], [E2]... mapped to
       do not provide enough    authoritative SQLite chunks
       evidence to answer..."            │
                                         ▼
                            [Prompt & Evidence Assembly]
                            - Untrusted passages enclosed in
                              <evidence_context><passage id="E1">
                            - System instructions isolated
                            - Strict citation mandate: cite only [E1]
                                         │
                                         ▼
                            [Local Ollama Chat Provider]
                            (qwen2.5:7b, low temperature)
                                         │
                                         ▼
                            [Citation Extraction & Set Check]
                            - Parse [E\d+] tokens from output
                            - Validate: [E_i] ∈ active evidence set
                            - Reject unverified markers (e.g. [E999])
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [Valid Citations]             [Zero Valid Citations]
                         │                               │
                         │                     [Bounded Single Repair]
                         │                     Re-prompt model with
                         │                     citation reminder
                         │                               │
                         └───────────────┬───────────────┘
                                         ▼
                            [Authoritative Metadata Resolution]
                            Resolve [E1] -> SQLite document filename,
                            location (page/slide), chunk excerpt
                                         │
                                         ▼
                                   [TutorResponse]
                            Markdown Answer + Citation Drawer
```

1. **Application-Owned Evidence Markers**: The application maps top retrieved chunks to short temporary markers (`[E1]`, `[E2]`). The model is never permitted to choose the mapping or invent metadata.
2. **Untrusted Content Encapsulation**: Document chunk texts are sanitized (neutralizing any premature `</passage>` or `</evidence_context>` tags) and wrapped in XML containers. System instructions strictly instruct the model to treat evidence text as passive study data, preventing prompt injection instructions from overriding tutor rules.
3. **Strict Set-Membership Validation**: Model output tokens matching `\[(E\d+)\]` are extracted and checked against the active request's evidence map. Any marker outside the active candidate set (such as hallucinated `[E999]`) is rejected and recorded with a warning.
4. **Authoritative Relational Resolution**: Valid citations populate display fields (filename, page/slide index, slide title, chunk excerpt) directly from SQLite relational records. Hallucinated filenames or page numbers emitted in model prose are never trusted as citation metadata.
5. **Honest Grounding & Abstention**: If retrieval yields 0 matching chunks, the tutor abstains immediately with a clear refusal message without making an LLM generation call. If the model determines that retrieved evidence is insufficient, it emits an honest abstention and is marked `unsupported=True`.

---

### 6. Stage 4 Exam Generation & Assessment Engine Architecture

```
                 Exam Generation Request
       (course, topic, question count, difficulty, types)
                           │
                           ▼
              [Hybrid Retrieval & Evidence Selection]
         - Course and document scoping enforced in SQLite & FTS5
         - Assign application-owned markers: [E1], [E2]...
         - XML encapsulation of untrusted passage text
                           │
                           ▼
             [Structured Ollama Generation]
          (qwen2.5:7b with low temperature & JSON schema)
                           │
                           ▼
          [Rigorous Python/Pydantic Validation]
         - MCQ: exactly one valid choice marked correct
         - True/False: boolean correctness normalization
         - Short-Answer: reference solution + multi-criteria rubric
         - Provenance: all cited markers must belong to active evidence
         - Deduplication: reject identical/near-identical stems
         - Bounded single retry on malformed JSON
                           │
                           ▼
           [Atomic SQLite Persistence (WAL)]
        - Store full exam, questions, source refs atomically
        - Never store model-generated database IDs
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 [Public Exam View]                    [Attempt Submission]
- Exclude correct_answer,             - Validate question IDs belong to exam
  explanation, and rubric             - Atomic attempt creation
- Leak-proof taking view              - Deterministic MCQ & True/False grading
                                      - Skipped questions scored 0.0
                                      - Conservative Short-Answer manual review
                                        or local model evaluation
                                              │
                                              ▼
                                 [Grounded Post-Exam Results]
                                - Earned score, percentage, feedback
                                - Explanations resolved to SQLite chunks
                                - Stale-chunk detection if sources deleted
```

- **Answer-Key Confidentiality**: The assessment architecture strictly separates internal question models (`ExamQuestion`) from public presentation models (`PublicQuestion`, `PublicExam`). Methods providing exams for taking (`get_exam_for_taking`) strip `correct_answer`, `reference_answer`, `rubric`, and `explanation`. Serialization tests verify that zero key data appears in JSON outputs or serialized states prior to submission.
- **Deterministic Objective Grading**: MCQ and True/False questions are graded deterministically against the stored answer key. Unanswered questions score 0.0 with clear feedback (`"Question was skipped."`).
- **Truthful Grading States & Pending Manual Review**: Short-answer questions awaiting grading have status `pending_manual_review` and unawarded points set to `None`, never a false score of `0.0`. If any question remains pending review, the overall exam attempt has status `pending_review` with final `score = None` and `percentage = None`. An explicit `provisional_score` and `provisional_percentage` subtotal are exposed for already-graded objective questions.
- **Instructor Manual-Review Completion**: The CLI provides a dedicated `review` subcommand (`python scripts/exams.py review --attempt-id <id> --question-id <id> --score <pts> --feedback "..."`) that records verified scores within question point bounds and transitions attempts to `completed` without permitting modifications to stored questions, answer keys, or rubrics.
- **Historical Citation Lifecycle & Stale Detection**: `exam_question_sources` persists an immutable provenance snapshot (`original_chunk_id`, filename, location type/index/title, excerpt) alongside a nullable foreign key (`chunk_id REFERENCES chunks(id) ON DELETE SET NULL`). While the source chunk exists, current SQLite metadata is authoritative. If the chunk or parent document is deleted, the historical snapshot is preserved, the citation is marked `is_stale = True` (`chunk_id = None`), and the exam questions, answer keys, and past attempt results remain completely uncorrupted.

---

### 7. Stage 5 Learner Model, Mastery Tracking & Spaced Repetition Architecture

```
                 Exam Attempt / Manual Grade / Review
                                 │
                                 ▼
                     [Learner Service Facade]
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
     [Idempotent Event Log]              [Question-to-Concept Mapping]
  - Chronological study history        - Course-scoped normalized concepts
  - Deduplicated by attempt_id         - Evidence-backed chunk links
  - Preserves pending vs. graded       - Reject cross-course collision
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
               [Deterministic Mastery Calculation]
             - Laplace-smoothed empirical accuracy:
                 M = (k + 1) / (n + 2)
             - Uniform Beta(1, 1) prior
             - Explicit categories:
                 unassessed (n=0)
                 needs_evidence (1 <= n < 3)
                 weak (n>=3, M < 0.60)
                 moderate (n>=3, 0.60 <= M < 0.85)
                 strong (n>=3, M >= 0.85)
             - Exclude pending grades completely
             - Corrected grades recompute safely
                                 │
                                 ▼
               [Spaced Repetition Review Queue]
             - Bounded offline interval policy (1 to 180 days)
             - Dynamic ease factor adjustment (1.3 to 3.0)
             - Status: scheduled, due (<= now), overdue (< now - 24h)
             - Rating updates: good (I*EF), pass (I*1.2), fail (I=1)
                                 │
                                 ▼
               [Explainable Progress & Recommendations]
             - Rule-based priority queue (zero hallucinated stats)
             - Priority: Overdue > Weak > Due > Needs Evidence > Unassessed
             - Traceable supporting evidence & explicit reasons
```

- **Separation of Observed Evidence vs. Estimated Mastery**: The model never treats lack of evidence as poor understanding. An unassessed concept ($n=0$) is categorized as `unassessed` ($M=0.0$ default estimate with prior uncertainty $\sigma \approx 0.29$), not a zero-mastery failure. Concepts with $1 \le n < 3$ observations are categorized as `needs_evidence`.
- **Laplace Smoothing Formula & Interpretability**:
  For every valid, completed, graded observation $i$ with earned score $\text{score\_earned}_i \ge 0$ and positive possible score $\text{score\_possible}_i > 0$, the normalized credit is:
  $$p_i = \frac{\text{score\_earned}_i}{\text{score\_possible}_i} \in [0.0, 1.0]$$
  For $n$ eligible observations, the fractional successes (normalized earned credits) and Laplace-smoothed mastery estimate are:
  $$k = \sum_{i=1}^n p_i, \quad M = \frac{k + 1}{n + 2}, \quad \text{Uncertainty } \sigma = \sqrt{\frac{(k + 1)(n - k + 1)}{(n + 2)^2 (n + 3)}}$$
  where $0.0 \le k \le n$. Each eligible observation contributes exactly one unit of evidence to $n$, with partial credit contributing a fractional success $p_i \in [0.0, 1.0]$ rather than raw unnormalized points.
  *What this does and does not prove:* This formula provides a transparent, empirical heuristic under a uniform $\text{Beta}(1, 1)$ prior. It does **not** claim to measure latent cognitive intelligence, neurological retention, or psychological aptitude.
- **Grading State Integrity**: Pending manual-review answers (`pending_manual_review`) and attempts awaiting grading (`pending_review`) are completely excluded from $k$ and $n$. They never lower concept mastery or count as false zeros. Finalizing a manual grade records an event and idempotently recomputes mastery from deduplicated history.
- **Bounded Offline Spaced-Repetition**: The review scheduler calculates intervals $I \in [1, 180]$ days entirely offline without cloud dependencies. Quality scores $q \in \{1, 3, 5\}$ adjust intervals and ease factors ($EF \in [1.3, 3.0]$), with failing reviews immediately resetting intervals to 1 day.
- **Rule-Based Recommendations**: Recommendations are produced deterministically from known relational state, explaining the exact reason (e.g., `"Review overdue (due 2026-10-25)"`, `"Consistently weak accuracy (45.0% across 4 observations)"`, `"Insufficient evidence (1 graded observation)"`). Generative language models are never used to hallucinate progress summaries or study statistics.

---

### 8. Stage 6 Knowledge Graph, FSRS Review Scheduler & External Research Architecture

```
                 Ingested Document Chunks
                            │
                            ▼
               [Direct Concept Discovery]
             - Structural headings & breadcrumbs
             - Bounded local JSON extraction
             - Chunk provenance in concept_chunks
             - Status: proposed / accepted / rejected
                            │
                            ▼
               [Course-Scoped Concept Graph]
             - Prerequisite (A -> B) & Related edges
             - Strict course isolation (same course)
             - BFS / Kahn's DAG cycle detection
                            │
                            ▼
          [Forgetting-Aware Review Scheduler (DSR)]
        - Policy: 'local-fsrs-v1'
        - Stability S, Difficulty D, Retrievability R:
            R(t, S) = (1 + 19/81 * t/S)^(-0.5)
        - Retention target: 90%
        - Bounds: [1.0, 365.0] days
                            │
                            ▼
            [Transparent Adaptive Study Planner]
        - Deterministic ranking:
            1. Overdue Spaced Reviews
            2. Unmet Target Prerequisites
            3. Due Spaced Reviews
            4. Weak Concept Remediation
            5. Target Concept Focus
            6. Gathering Evidence (< 3 obs)
            7. Initial Diagnostics
        - Time-budgeted scheduling (e.g. 20, 30, 60 min)
        - Zero runtime LLM calls
                            │
                            ▼
        [Explicit Opt-In External Research (--external-research)]
      - Strict privacy boundary: Zero document notes sent to search
      - Network SSRF defense: Blocks loopback, RFC 1918, link-local, cloud metadata
      - Free, keyless providers: Wikipedia REST Search API & FakeSearchProvider
      - Application-assigned citations: [EXT-1], [EXT-2], ...
      - Citation validation: Hallucinated [EXT-99] rejected and sanitized
      - Untrusted content envelope for prompt injection defense
```

#### CLI Operations for Stage 6

```powershell
# Direct concept discovery from raw chunks
python scripts/learner.py discover-concepts --course "CS101" --max-chunks 15

# View concept relationships
python scripts/learner.py relationships --course "CS101"

# Add a prerequisite edge (DAG cycle detection enforced)
python scripts/learner.py add-relationship --course "CS101" --source "cpt_arrays" --target "cpt_lists" --type prerequisite

# Topological prerequisite order for a target concept
python scripts/learner.py prerequisites --course "CS101" --concept-id "cpt_dijkstra"

# Generate today's adaptive study plan with a 30-minute time budget
python scripts/learner.py plan --course "CS101" --time-budget 30

# Explicit opt-in external research with Wikipedia and verified citations
python scripts/learner.py research --query "Dijkstra algorithm priority queue" --course "CS101" --external-research
```

---

### 9. Partial Failure Handling & Index Recovery
- **Non-blocking Indexing**: Ingestion transactions in SQLite commit independently. If vector embedding or ChromaDB indexing fails post-commit (e.g., if Ollama is paused or out of memory), the SQLite document and chunk data remain intact.
- **Actionable Warnings**: The ingestion pipeline logs a partial failure warning with the exact recovery command.
- **Idempotent Repair**: Running `python scripts/sync_index.py --sync` safely catches up unsynced chunks without duplicating or corrupting existing embeddings.
- **Degraded Lexical Answering**: If Ollama or ChromaDB is offline, the tutor answers using available lexical FTS5 evidence when adequate, honestly preserving the `DEGRADED_LEXICAL` mode indicator and diagnostic warnings.

---

### 10. Automated Testing & Continuous Integration Without Ollama
The entire test suite is 100% deterministic and runs completely offline without requiring a running Ollama daemon or downloading models:
- **Fake Embeddings**: Unit and integration tests inject `FakeEmbeddingProvider`, generating deterministic 384-dimensional vectors from SHA-256 hashes.
- **Fake Chat Provider**: Tutor, exam, concept discovery, and research tests inject `FakeChatProvider`, supporting queued responses, citation verification, and error simulation.
- **Mock HTTP Layer**: Ollama API and external network tests simulate responses, timeouts, connection errors, SSRF attempts, socket connection pinning, and malformed JSON using mocked HTTP transports.
- **Temporary Storage**: All tests run against ephemeral SQLite databases and temporary directories for ChromaDB.
- **Continuous Integration**: GitHub Actions workflow (`.github/workflows/ci.yml`) runs across Windows and Linux matrix configurations, checking lockfile consistency (`uv lock --check`), dependency sync (`uv sync --frozen`), compilation, evaluation benchmark, and the complete pytest suite.

Run the complete test suite (287 passing tests):
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

### 11. Deterministic Evaluation Benchmark Harness
StudyOS includes a dedicated, reproducible quality and security benchmark (`scripts/evaluate.py`) running against labeled academic fixtures:

```powershell
# Run the complete deterministic evaluation benchmark
.\.venv\Scripts\python.exe scripts/evaluate.py

# Output structured JSON metrics for automated CI reporting
.\.venv\Scripts\python.exe scripts/evaluate.py --json

# Optional: include live Ollama model availability check
.\.venv\Scripts\python.exe scripts/evaluate.py --live-model
```

#### Measured Benchmark Dimensions:
1. **Retrieval Quality (RRF $k=60$):** Measures Top-1 accuracy, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR) against ground-truth target chunks.
2. **Citation Integrity & Hallucination Defense:** Confirms 100% resolution of authentic citations and 100% rejection/sanitization of hallucinated markers (`[E99]`).
3. **Tutor Grounding & Immediate Abstention:** Verifies that unanswerable or unindexed queries trigger immediate refusal (`unsupported=True`) without hallucinating content.
4. **Assessment Security & Answer-Key Isolation:** Evaluates public exam presentations (`PublicExam`) to guarantee zero leakage of answer keys, rubrics, or explanations prior to grading.
5. **Learner Mastery & Forgetting-Aware Scheduling:** Validates Laplace-smoothed posterior mastery estimates ($M = (k+1)/(n+2)$) and power-law memory decay retrievability under FSRS (`local-fsrs-v1`).
6. **Concept Graph Constraints:** Enforces database-level rejection of prerequisite cycles and cross-course relationship edges.

---

### 12. Local Data Management, Backup & Restore
All user data resides in `~/.studyos` (`studyos.db` for relational entities and `chroma/` for persistent dense vector indexes).

#### A. Hot Online Backup (Live Database)
Because StudyOS uses SQLite WAL mode, you can safely take a non-blocking, transactionally consistent live backup at any time:
```powershell
# Python hot online backup using SQLite's native backup API
.\.venv\Scripts\python.exe -c "
import sqlite3
from pathlib import Path
src = sqlite3.connect(Path.home() / '.studyos' / 'studyos.db')
dst = sqlite3.connect(Path.home() / '.studyos' / 'backup_studyos.db')
src.backup(dst)
dst.close()
src.close()
print('Hot backup completed successfully.')
"
```

#### B. Cold File Backup & Restore
To copy the database file directly, first checkpoint the write-ahead log:
```powershell
# Checkpoint WAL log into the main database file
.\.venv\Scripts\python.exe -c "
import sqlite3
from pathlib import Path
conn = sqlite3.connect(Path.home() / '.studyos' / 'studyos.db')
conn.execute('PRAGMA wal_checkpoint(TRUNCATE);')
conn.close()
"

# Copy the database file
Copy-Item "$HOME\.studyos\studyos.db" "$HOME\.studyos\studyos_backup.db"
```

#### C. Database Migration Integrity
All migrations (0001 through 0007) execute inside atomic transaction blocks (`BEGIN TRANSACTION; ... COMMIT;`). If any migration statement fails, the transaction is immediately rolled back, leaving zero partial tables and zero schema corruption.

---

### 13. Current Known Limitations
- **Citation Provenance vs. Claim Entailment**: The application strictly validates citation reference provenance (guaranteeing that every citation resolves to a real SQLite chunk retrieved for that query and rejecting invented markers). However, syntactic citation validation does not mathematically prove logical entailment of the generated claim. Formal natural language inference (NLI) evaluation is reserved for Stage 7.
- **Assessment Structure vs. Pedagogical Validity**: The assessment engine deterministically validates question structure, schema, distractor uniqueness, single-correct-answer constraints, and evidence marker provenance. However, automated generation cannot guarantee pedagogical perfection or replace human instructor review.
- **Local FSRS Model vs. Neurological Memory**: The review scheduling algorithm implements a documented, bounded DSR policy (`local-fsrs-v1`) calculating power-law retrievability $R(t,S)$. While mathematically principled, it is an empirical scheduling heuristic and does not prove neurological retention across all individuals.
- **Course-Specific Concept Graphs vs. Universal Laws**: The prerequisite DAG represents course-specific instructional ordering derived from lecture materials or instructor input; it does not claim to define universal epistemology.
- **External Web Sources vs. Course Truth**: External research findings are explicitly partitioned from course notes and tagged with `[EXT-N]`. A retrieved Wikipedia article never overwrites course notes or changes student mastery scores.
- **Prompt Injection Defense in Depth**: Delimitation, untrusted envelopes, and token extraction provide robust defense against untrusted external web text and document-borne instructions, but prompt-injection defense remains defense-in-depth rather than a mathematical guarantee against language model non-determinism.
- **Conservative Short-Answer Grading**: Open-ended question grading defaults to manual instructor review when offline to prevent hallucinated scoring. Automated model grading is strictly bounded by max points and criterion rubrics.
- **Heuristic Concept Extraction**: Automated concept labeling uses bounded local LLM extraction or fallback topic extraction from question metadata. While deduplication prevents name collisions within courses, subtle synonymy (e.g., "Binary Search" vs. "B-Search") is handled via normalized strings and manual reassignment workflows.
- **Heuristic Spaced Repetition**: The review scheduling algorithm uses bounded SM-2/Leitner intervals. While explainable and dependable offline, it is an empirical study policy rather than a mathematically optimal memory decay model.
- **Ollama Prerequisite for Live Chat & Semantic Search**: Live interactive answering requires a local Ollama daemon serving `qwen2.5:7b` (for tutoring/exams) and `all-minilm:latest` (for dense embeddings). When offline, the CLI operates in degraded mode or via the `--fake-chat` and `--fake-embeddings` flags.
- **Prompt Injection Defense in Depth**: Delimitation, system instruction separation, and token extraction provide robust defense against document-borne instructions, but prompt-injection defense remains defense-in-depth rather than a mathematical guarantee against language model non-determinism.

---

## License & Ownership

Developed by **Youhanna Younan** as an open-source, local-first personal AI learning operating system.
*License:* Planned for release under the MIT License (the license file has not yet been added to the repository and will be provided upon initial project release).
