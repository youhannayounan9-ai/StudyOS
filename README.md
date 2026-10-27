# StudyOS

> **A local-first, privacy-respecting, AI-powered personal learning operating system.**

StudyOS is an intelligent personal study environment designed for computer science and data science students. Unlike generic "chat with PDF" tools, StudyOS transforms passive course materials (lecture slides, textbook chapters, notes, and past exams) into an active, verifiable cognitive tutor and examination engine.

---

## Current Status: Stage 3 — Implementation in review (Tutor & Evidence-Grounded Answer Engine)

StudyOS is developed under a disciplined, incremental engineering roadmap. Stage 0 (Architecture & Specification), Stage 1 (Document Ingestion Engine), and Stage 2 (Hybrid Retrieval & Evidence Grounding Engine) are complete and committed to the baseline. Stage 3 (AI Tutor & Grounded Interactive Answering) is implemented and verified with 144 automated tests passing, currently undergoing review.

- [x] **Stage 0: Architecture & Specification** *(Approved & Baseline Committed at `1827f1a`)*
- [x] **Stage 1: Document Ingestion, Parsing & Normalization Engine** *(Approved & Baseline Committed at `84db803`)*
- [x] **Stage 2: Hybrid Retrieval & Evidence Grounding Engine** *(Approved & Baseline Committed at `7026c4c`)*
- [ ] **Stage 3: AI Tutor & Grounded Interactive Answering** *(Implementation in review)*
- [ ] **Stage 4: Exam Generation, Submission & Automated Grading Engine** *(Upcoming)*
- [ ] **Stage 5: Persistent Learner Model & Mistake Analytics**
- [ ] **Stage 6: Adaptive Learning, Spaced Repetition & External Research**
- [ ] **Stage 7: Comprehensive Evaluation, Security & Portfolio Polish**

*Note: In accordance with Stage 3 scope, the local tutor engine answers student questions grounded strictly in uploaded course materials retrieved via hybrid search (FTS5 + ChromaDB). Citations are bound to application-owned evidence markers (`[E1]`, `[E2]`) and resolved directly to authoritative SQLite metadata (filename, page/slide). If evidence is missing or insufficient, the tutor explicitly abstains.*

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

### 6. Partial Failure Handling & Index Recovery
- **Non-blocking Indexing**: Ingestion transactions in SQLite commit independently. If vector embedding or ChromaDB indexing fails post-commit (e.g., if Ollama is paused or out of memory), the SQLite document and chunk data remain intact.
- **Actionable Warnings**: The ingestion pipeline logs a partial failure warning with the exact recovery command.
- **Idempotent Repair**: Running `python scripts/sync_index.py --sync` safely catches up unsynced chunks without duplicating or corrupting existing embeddings.
- **Degraded Lexical Answering**: If Ollama or ChromaDB is offline, the tutor answers using available lexical FTS5 evidence when adequate, honestly preserving the `DEGRADED_LEXICAL` mode indicator and diagnostic warnings.

---

### 7. Automated Testing Without Ollama
The entire test suite is 100% deterministic and runs completely offline without requiring a running Ollama daemon or downloading models:
- **Fake Embeddings**: Unit and integration tests inject `FakeEmbeddingProvider`, generating deterministic 384-dimensional vectors from SHA-256 hashes.
- **Fake Chat Provider**: Tutor tests inject `FakeChatProvider`, supporting queued responses, citation verification, and error simulation.
- **Mock HTTP Layer**: Ollama API network tests simulate responses, timeouts, connection errors, and malformed JSON using mocked HTTP transports.
- **Temporary Storage**: All tests run against ephemeral SQLite databases and temporary directories for ChromaDB.

Run the complete test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

### 8. Current Known Limitations
- **Citation Provenance vs. Claim Entailment**: The application strictly validates citation reference provenance (guaranteeing that every citation resolves to a real SQLite chunk retrieved for that query and rejecting invented markers). However, syntactic citation validation does not mathematically prove logical entailment of the generated claim. Formal natural language inference (NLI) evaluation is reserved for Stage 7.
- **Ollama Prerequisite for Live Chat & Semantic Search**: Live interactive answering requires a local Ollama daemon serving `qwen2.5:7b` (for tutoring) and `all-minilm:latest` (for dense embeddings). When offline, the CLI operates in degraded mode or via the `--fake-chat` and `--fake-embeddings` flags.
- **Single-Turn Scoped Q&A**: Stage 3 implements stateless, evidence-grounded single-turn question answering. Multi-turn dialogue and persistent learner memory are planned for Stage 5.
- **Prompt Injection Defense in Depth**: Delimitation, system instruction separation, and token extraction provide robust defense against document-borne instructions, but prompt-injection defense remains defense-in-depth rather than a mathematical guarantee against language model non-determinism.

---

## License & Ownership

Developed by **Youhanna Younan** as an open-source, local-first personal AI learning operating system.
*License:* Planned for release under the MIT License (the license file has not yet been added to the repository and will be provided upon initial project release).
