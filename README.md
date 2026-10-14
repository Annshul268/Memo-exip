# Memo-exip: Adaptive Temporal Memory Management for LLM Agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange.svg)](https://www.trychroma.com/)
[![SQLite](https://img.shields.io/badge/SQLite-ACID_Source_of_Truth-003B57.svg)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-14_passed-brightgreen.svg)](https://docs.pytest.org/)

**MEMO-EXI** is a research prototype investigating **continual external memory with adaptive temporal management for Large Language Model (LLM) agents**.

The system addresses the fundamental limitation of standard vector-based Retrieval-Augmented Generation (RAG): **vector similarity alone does not establish fact validity, temporal correctness, or obsolescence.** When user facts evolve across conversations, conventional semantic retrievers indiscriminately retrieve stale, contradictory, and superseded facts, inflating prompt tokens and causing hallucinations.

Memo-exip resolves this via **bitemporal record modeling, transactional version lineage, intent-based hybrid retrieval, and utility-based context-budget knapsack packing**.

---

## 1. Core Architecture & Pipeline

Memo-exip decouples semantic indexing from authoritative temporal state through a dual-store architecture:

```
                           +-------------------------------------------------------------+
                           |                     User Query / Task                       |
                           +------------------------------+------------------------------+
                                                          |
                                                          v
                                      +---------------------------------------+
                                      |   Temporal Query Intent Classifier    |
                                      |   (Current vs. Historical vs. At-Time)|
                                      +-------------------+-------------------+
                                                          |
                      +-----------------------------------+-----------------------------------+
                      |                                                                       |
                      v                                                                       v
+-------------------------------------------+                       +---------------------------------------------------+
|       ChromaDB Vector Store               |                       |              SQLite Relational Store              |
| (Semantic Embeddings of Fact Statements)  |                       | (Structured Temporal Metadata, Versions, Lineage) |
+---------------------+---------------------+                       +-------------------------+-------------------------+
                      |                                                                       |
                      +-----------------------------------+-----------------------------------+
                                                          |
                                                          v
                                      +---------------------------------------+
                                      |       Hybrid Temporal Filtering       |
                                      |         & Utility Scoring             |
                                      +-------------------+-------------------+
                                                          |
                                                          v
                                      +---------------------------------------+
                                      |   Context-Budget Knapsack Selector    |
                                      |   (Token-bounded, Redundancy penalty) |
                                      +-------------------+-------------------+
                                                          |
                                                          v
                                      +---------------------------------------+
                                      |           LLM Prompt Context          |
                                      |          & Response Generation        |
                                      +-------------------+-------------------+
                                                          |
                                                          v
                                      +---------------------------------------+
                                      |    Temporal Memory Lifecycle Manager  |
                                      |    (Conflict Detection & Versioning)  |
                                      +---------------------------------------+
```

### Why Dual Storage?
- **SQLite (Authoritative Source of Truth):** Manages relational integrity, bitemporal intervals (`valid_from`, `valid_until`), record transaction timestamps (`created_at`, `updated_at`), lifecycle status (`current`, `superseded`, `disputed`), and parent version pointers (`supersedes_id`).
- **ChromaDB (Semantic Vector Index):** Provides dense semantic similarity search over natural language expressions of facts. If index discrepancy occurs, `reconcile_storage()` resynchronizes ChromaDB directly from SQLite.

---

## 2. Key Capabilities & Research Contributions

1. **Bitemporal Fact Storage & Historical Preservation:** Facts are never silently overwritten. When a fact changes (e.g., deadline moves from Nov 10 to Nov 20), the earlier record is marked `superseded` and bounded by `valid_until`, while the new record becomes `current`.
2. **Temporal Intent Query Disambiguation:** The retriever dynamically routes queries:
   - *Current queries* (*"What is my deadline now?"*) prioritize active records and suppress superseded versions.
   - *Historical queries* (*"What was my deadline before?"*) prioritize superseded records in the version lineage.
   - *Point-in-time queries* (*"What was the status on Nov 3?"*) check interval overlap $t \in [t_{\text{valid\_from}}, t_{\text{valid\_until}}]$.
3. **Conflict & Ambiguity Handling:** Conflicting statements with uncertain validity or conflicting sources are flagged as `disputed` rather than silently replacing confirmed facts.
4. **Context-Budget Knapsack Selection:** A greedy utility-per-token algorithm packs memories into a strict token budget, applying dynamic penalties against semantic redundancy.
5. **Adaptive Tabular UCB Controller:** A lightweight contextual bandit inspired by MEMCON (Jiang et al., 2026) adapts retrieval parameters ($\text{top\_k}$, search depth) with **zero additional LLM calls**.
6. **Zero-API-Key Deterministic Mode:** Full functionality, unit tests, and the evaluation suite run deterministically out of the box using `MockLLMProvider`, without requiring third-party API keys.

---

## 3. Empirical Benchmark Results

Evaluated on a standardized 10-scenario benchmark suite covering fact evolution, historical queries, contradiction handling, redundancy flooding, stable invariants, and context budget pressure (token budget = 500):

| Baseline System | Overall Accuracy (%) | Current Fact Accuracy (%) | Historical Fact Accuracy (%) | Stale-Fact Error Rate (%) | Avg Tokens / Query | Budget Compliance (%) | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: No Persistent Memory** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 100.0% | 0.01 ms |
| **B1: Recent History Window (N=3)** | 50.0% | 42.9% | 50.0% | 14.3% | 30.6 | 100.0% | 0.02 ms |
| **B2: Fixed Semantic Vector Retrieval** | 70.0% | 71.4% | 50.0% | 28.6% | 93.9 | 100.0% | 46.60 ms |
| **B3: Temporal Store (Fixed Rules)** | 100.0% | 100.0% | 100.0% | 0.0% | 119.0 | 100.0% | 48.05 ms |
| **P: Proposed Adaptive Temporal (Ours)** | **100.0%** | **100.0%** | **100.0%** | **0.0%** | **79.1** | **100.0%** | **47.51 ms** |

### Verified Empirical Findings:
- **Elimination of Stale-Fact Errors:** Conventional vector retrieval (B2) suffers a **28.6% stale-fact error rate** because superseded facts possess identical vector embeddings. Our proposed system achieves **0.0% stale-fact error rate**.
- **Historical Query Accuracy:** Historical retrieval accuracy jumps from **50.0%** in B1/B2 to **100.0%** in our system due to explicit version lineage.
- **Context Token Reduction:** The utility-based knapsack selector in **P** reduces prompt token consumption by **33.5%** compared to unconstrained temporal retrieval B3 (79.1 vs. 119.0 tokens) while maintaining 100% accuracy.
- **Relational Overhead:** SQLite temporal validation adds less than **1.5 ms** over raw ChromaDB retrieval.

Raw reproducible data is preserved at `results/benchmark_results.json`.

---

## 4. Repository Structure

```
Memo-exip/
├── app/                        # FastAPI REST Backend
│   ├── main.py                 # Application entrypoint & CORS
│   ├── models.py               # Pydantic request/response schemas
│   └── routes.py               # Endpoints for CRUD, search, agent query
├── memory/                     # Core Temporal Memory Engine
│   ├── schema.py               # Bitemporal MemoryRecord and Enums
│   ├── sqlite_store.py         # Authoritative transactional SQLite store
│   ├── vector_store.py         # ChromaDB semantic index & reconciliation
│   ├── temporal_engine.py      # Versioning, superseding & conflict logic
│   ├── hybrid_retriever.py     # Intent classification & temporal scoring
│   ├── context_budget.py       # Utility knapsack packer & redundancy control
│   ├── adaptive_controller.py  # Tabular UCB contextual bandit controller
│   └── agent_pipeline.py       # End-to-end question answering pipeline
├── llm/                        # LLM Provider Abstraction
│   ├── base.py                 # Abstract Base Class
│   ├── mock_provider.py        # Deterministic mock provider (zero keys)
│   ├── groq_provider.py        # Groq Cloud API integration
│   └── factory.py              # Dynamic provider instantiation
├── evaluation/                 # Benchmark Framework
│   ├── dataset_generator.py    # 10 synthetic temporal challenge scenarios
│   ├── baselines.py            # Implementations of B0, B1, B2, B3, P
│   ├── metrics.py              # Deterministic accuracy and error metrics
│   └── runner.py               # Automated benchmark execution script
├── ui/                         # Streamlit Interactive Web Application
│   └── app.py                  # Multi-page dashboard and live demonstration
├── tests/                      # Automated Test Suite (pytest)
│   ├── test_temporal_engine.py # Tests for ACID updates, versioning, conflicts
│   ├── test_hybrid_retrieval.py# Tests for intent routing and temporal filters
│   ├── test_context_budget.py  # Tests for token estimation and budget limits
│   ├── test_agent_pipeline.py  # End-to-end deadline update integration test
│   ├── test_api.py             # FastAPI TestClient endpoint tests
│   └── test_evaluation.py      # Tests for benchmark execution
├── docs/                       # Comprehensive Academic Documentation
│   ├── architecture.md         # Detailed system architecture & dual-store
│   ├── algorithms.md           # Mathematical formulas & algorithms
│   ├── traceability.md         # Research traceability matrix
│   └── api_reference.md        # REST API endpoint reference
├── results/                    # Machine-readable evaluation outputs
│   └── benchmark_results.json  # Raw benchmark measurements
├── .env.example                # Configuration template (no secrets)
├── requirements.txt            # Dependency manifest
├── pytest.ini                  # Pytest configuration
└── README.md                   # Project overview & documentation
```

---

## 5. Quickstart & Installation

### 5.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/Annshul268/Memo-exip.git
cd Memo-exip

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
By default, `LLM_PROVIDER=mock`, allowing the system to run locally without any external API keys. To connect Groq:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## 6. Running the System

### 6.1 Start the FastAPI Backend
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be live at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 6.2 Launch the Streamlit Interactive UI
In a separate terminal:
```bash
streamlit run ui/app.py --server.port 8501
```
Navigate to `http://localhost:8501` to access:
- **Overview:** System summary, live backend counts, retrieval mode, context budget, and temporal architecture guide.
- **Memory Chat:** Interactive conversation, fact ingestion, historical/current queries, and memory inspection.
- **Memory Explorer:** Searchable memory table, validity intervals, disputed facts, and chronological version lineage.
- **Context Inspector:** Transparent breakdown of candidate retrieval, utility scoring, and knapsack token packing.
- **Evaluation:** Real empirical benchmark results across all 5 baselines with accuracy and token consumption comparisons.

---

## 7. Running Tests & Reproducing Benchmarks

### 7.1 Execute Automated Tests
```bash
pytest -v
```
All 14 unit and integration tests run deterministically with zero external network dependencies.

### 7.2 Reproduce the Benchmark Evaluation
```bash
python3 -m evaluation.runner
```
This executes all 5 baselines across the 10 scenarios and regenerates `results/benchmark_results.json`.

---

## 8. Limitations & Future Work

1. **Natural Language Information Extraction:** Ingesting free-form unstructured paragraphs currently relies on a lightweight heuristic regex/rule extractor when subject-attribute-value triples are not explicitly passed. Future work will integrate an open-source entity-relation tuple extraction model.
2. **Cross-Task Bandit Credit Assignment:** Following the observation in MEMCON (Jiang et al., 2026), maintenance operations (`Consolidate`, `Forget`) primarily yield benefits across future tasks rather than the immediate query. Incorporating temporal-difference learning with cross-task trace decay will improve long-horizon policy stability.
3. **Complex Temporal Reasoning:** Expanding point-in-time filtering to support fuzzy relative intervals (e.g., *"two weeks before last Thanksgiving"*) using formal temporal logic (Allen's Interval Algebra).

---

## 9. License

This project is licensed under the MIT License.