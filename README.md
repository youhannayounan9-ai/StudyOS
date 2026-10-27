# StudyOS

> **A local-first, privacy-respecting, AI-powered personal learning operating system.**

StudyOS is an intelligent personal study environment designed for computer science and data science students. Unlike generic "chat with PDF" tools, StudyOS transforms passive course materials (lecture slides, textbook chapters, notes, and past exams) into an active, verifiable cognitive tutor and examination engine.

---

## Current Status: Stage 2 — Implementation in review (Hybrid Retrieval & Evidence Grounding Engine)

StudyOS is developed under a disciplined, incremental engineering roadmap. Stage 0 (Architecture & Specification) and Stage 1 (Document Ingestion Engine) are complete and committed to the baseline. Stage 2 (Hybrid Retrieval & Evidence Grounding Engine) is implemented and verified with 96 automated tests passing, currently undergoing review.

- [x] **Stage 0: Architecture & Specification** *(Approved & Baseline Committed at `1827f1a`)*
- [x] **Stage 1: Document Ingestion, Parsing & Normalization Engine** *(Approved & Baseline Committed at `84db803`)*
- [ ] **Stage 2: Hybrid Retrieval & Evidence Grounding Engine** *(Implementation in review)*
- [ ] **Stage 3: AI Tutor & Grounded Interactive Answering** *(Upcoming)*
- [ ] **Stage 4: Exam Generation, Submission & Automated Grading Engine**
- [ ] **Stage 5: Persistent Learner Model & Mistake Analytics**
- [ ] **Stage 6: Adaptive Learning, Spaced Repetition & External Research**
- [ ] **Stage 7: Comprehensive Evaluation, Security & Portfolio Polish**

*Note: In accordance with Stage 2 scope, the hybrid retrieval engine combines SQLite FTS5 BM25 lexical ranking with local ChromaDB dense semantic vector search via Reciprocal Rank Fusion (RRF, k=60). All returned chunks are strictly validated and enriched against SQLite authoritative relational records.*

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

---

### 5. Partial Failure Handling & Index Recovery
- **Non-blocking Indexing**: Ingestion transactions in SQLite commit independently. If vector embedding or ChromaDB indexing fails post-commit (e.g., if Ollama is paused or out of memory), the SQLite document and chunk data remain intact.
- **Actionable Warnings**: The ingestion pipeline logs a partial failure warning with the exact recovery command.
- **Idempotent Repair**: Running `python scripts/sync_index.py --sync` safely catches up unsynced chunks without duplicating or corrupting existing embeddings.

---

### 6. Automated Testing Without Ollama
The entire test suite is deterministic and runs offline without requiring a running Ollama daemon or downloading models:
- **Fake Embeddings**: Unit and integration tests inject `FakeEmbeddingProvider`, which generates deterministic 384-dimensional vectors based on input string hashes.
- **Mock HTTP Layer**: Ollama API network tests simulate responses, timeouts, connection errors, and malformed JSON using mocked HTTP transports.
- **Temporary Storage**: All tests run against ephemeral SQLite databases and temporary directories for ChromaDB.

Run the complete test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

### 7. Current Known Limitations
- **Ollama Prerequisite for Live Semantic Retrieval**: Live semantic vector search requires a local Ollama daemon serving `all-minilm:latest`. If Ollama is offline, the hybrid engine operates in honest degraded mode (`DEGRADED_LEXICAL`) with an explicit warning, or fails gracefully if strict semantic retrieval was demanded.
- **Cross-Lingual Embeddings**: The default model `all-minilm` is optimized for English technical prose. Multi-language course notes may exhibit lower semantic recall.
- **In-Memory Query Re-Ranking**: RRF combines lexical and semantic rank lists. Cross-encoder re-ranking is reserved for later optimization stages.

---

## License & Ownership

Developed by **Youhanna Younan** as an open-source, local-first personal AI learning operating system.
*License:* Planned for release under the MIT License (the license file has not yet been added to the repository and will be provided upon initial project release).
