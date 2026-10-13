# Memo-exip System Architecture

## 1. Overview and Core Philosophy

**Memo-exip** implements **Continual External Memory with Adaptive Temporal Management for Large Language Model (LLM) Agents**. The system decouples the static knowledge weights of a frozen foundation model from a dynamic, version-aware external memory store.

Unlike standard vector retrieval architectures that store unstructured chunks and retrieve exclusively by semantic proximity, Memo-exip manages the **temporal lifecycle of evolving facts**.

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

---

## 2. Complementary Storage Roles: Dual-Store Design

Memo-exip relies on a dual-store paradigm where **SQLite** and **ChromaDB** serve distinct, complementary roles:

### 2.1 SQLite (Authoritative Relational Source of Truth)
- **Role:** Structured record ledger, temporal interval indexing, version lineage, and transaction coordinator.
- **ACID Guarantees:** When a fact is updated, the superseding operation occurs transactionally. If an update fails mid-flight, SQLite rolls back, ensuring that no partially updated version history corrupts the store.
- **Relational Lineage:** Each record maintains `supersedes_id`, creating an immutable linked chain representing the chronological evolution of any entity-attribute tuple.

### 2.2 ChromaDB (Semantic Vector Index)
- **Role:** Dense semantic similarity retrieval over natural language representations of stored facts.
- **Reconciliation:** ChromaDB acts as a secondary derived index. If an index corruption or process interruption occurs, `reconcile_with_sqlite()` reads all authoritative records from SQLite, purges stale vector IDs, and re-indexes the collection.

---

## 3. Temporal Semantics: Bitemporal Modeling

Memo-exip explicitly distinguishes between **Transaction Time** and **Real-World Validity Time**:

1. **Transaction Time (Record Time):**
   - `created_at`: The UTC timestamp when the system ingested the memory record.
   - `updated_at`: The UTC timestamp when the record was last modified or touched.
2. **Real-World Validity Time:**
   - `valid_from`: The UTC timestamp when the real-world proposition became true.
   - `valid_until`: The UTC timestamp when the proposition ceased to be true (`None` indicates active into the indefinite future).
3. **Lifecycle Status (`status`):**
   - `current`: Actively valid current truth.
   - `superseded`: Historical fact superseded by a newer verified record.
   - `disputed`: Conflicting statement detected without definitive resolution.
   - `expired`: Fact whose `valid_until` has lapsed.

---

## 4. End-to-End Pipeline Execution Flow

1. **Ingestion / Fact Update:**
   - An assertion is passed to `TemporalMemoryEngine.add_or_update_fact()`.
   - The engine checks for existing records matching `(subject, attribute)`.
   - If identical value $\to$ Reinforces confidence and increments access count.
   - If new value with confirmed correction $\to$ Closes validity of previous record at $t_{\text{valid\_from}}$, assigns `supersedes_id`, and inserts the new record as `current`.
   - If contradictory statement without explicit correction $\to$ Marks new record as `disputed`, preserving the existing truth.
2. **Query Processing:**
   - The query is analyzed by `TemporalIntentClassifier` (categorizing into `CURRENT`, `HISTORICAL`, `AT_TIME`, `TIMELINE`, or `GENERAL`).
   - `HybridTemporalRetriever` queries ChromaDB for top semantic candidates and queries SQLite for authoritative metadata.
   - Temporal relevance scores are assigned based on query intent and record status.
   - `ContextBudgetManager` runs a greedy utility-per-token knapsack algorithm to pack memories into the specified token budget while penalizing redundancy.
   - Prompt context is formulated and sent to the LLM backend (`GroqLLMProvider` or `MockLLMProvider`).
