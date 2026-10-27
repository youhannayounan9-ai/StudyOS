# StudyOS

> **A local-first, privacy-respecting, AI-powered personal learning operating system.**

StudyOS is an intelligent personal study environment designed for computer science and data science students. Unlike generic "chat with PDF" tools, StudyOS transforms passive course materials (lecture slides, textbook chapters, notes, and past exams) into an active, verifiable cognitive tutor and examination engine.

---

## Current Status: Stage 1 Complete & Approved (Stage 2: Hybrid Retrieval Next)

StudyOS is developed under a disciplined, incremental engineering roadmap. Stage 0 (Architecture & Specification) and Stage 1 (Document Ingestion, Parsing & Normalization Engine) are complete and approved with 53 automated tests passing. Stage 2 (Hybrid Retrieval & Evidence Grounding Engine) is the next milestone.

- [x] **Stage 0: Architecture & Specification** *(Approved & Baseline Committed at `1827f1a`)*
- [x] **Stage 1: Document Ingestion, Parsing & Normalization Engine** *(Approved & Complete with 53 tests passing)*
- [ ] **Stage 2: Hybrid Retrieval & Evidence Grounding Engine** *(Next Stage)*
- [ ] **Stage 3: AI Tutor & Grounded Interactive Answering**
- [ ] **Stage 4: Exam Generation, Submission & Automated Grading Engine**
- [ ] **Stage 5: Persistent Learner Model & Mistake Analytics**
- [ ] **Stage 6: Adaptive Learning, Spaced Repetition & External Research**
- [ ] **Stage 7: Comprehensive Evaluation, Security & Portfolio Polish**

*Note: In accordance with Stage 1 scope, the ingestion engine, format parsers (PDF with encryption rejection, DOCX, PPTX with slide table extraction, TXT), structure-aware chunker with loop progress guarantees, SQLite storage layer with atomic force-reindex and trigger-based FTS5 sync, and test suites are verified with 53/53 tests passing.*

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

## Intended Setup Approach (Stage 1 Preview)

Once development commences in Stage 1, the environment will be initialized using `uv`:

```powershell
# 1. Clone repository
git clone https://github.com/youhannayounan9-ai/StudyOS.git
cd StudyOS

# 2. Create and activate virtual environment with uv
uv venv --python 3.12
.venv\Scripts\activate

# 3. Install core dependencies
uv pip install -e .

# 4. Start local inference service (Ollama)
ollama serve

# 5. Launch StudyOS
streamlit run app.py
```

---

## License & Ownership

Developed by **Youhanna Younan** as an open-source, local-first personal AI learning operating system.
*License:* Planned for release under the MIT License (the license file has not yet been added to the repository and will be provided upon initial project release).
