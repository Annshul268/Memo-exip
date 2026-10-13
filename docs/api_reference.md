# Memo-exip REST API Reference

The FastAPI backend exposes endpoints for temporal memory ingestion, version querying, hybrid retrieval, agent question answering, and storage reconciliation.

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## 1. System Endpoints

### `GET /health`
Returns system diagnostics, storage connectivity, active LLM provider, and total memory count.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "llm_provider": "mock",
  "is_mock_llm": true,
  "sqlite_connected": true,
  "chroma_connected": true,
  "memory_count": 12
}
```

### `POST /api/memories/reconcile`
Resynchronizes ChromaDB vector index from the authoritative SQLite store.

---

## 2. Memory CRUD & Lineage Endpoints

### `POST /api/memories`
Add a new memory or update an existing fact version-aware.

**Request Body:**
```json
{
  "subject": "user",
  "attribute": "deadline",
  "value": "November 20",
  "valid_from": "2026-11-05T12:00:00Z",
  "confidence": 1.0,
  "importance": 0.8,
  "is_explicit_correction": true
}
```

**Response (201 Created):**
```json
{
  "message": "Fact processed with action: superseded_previous_version_mem_87f21a",
  "action": "superseded_previous_version_mem_87f21a",
  "record": { ... },
  "status": "current",
  "supersedes_id": "mem_87f21a"
}
```

### `GET /api/memories`
List all stored records.
- Query parameters: `status` (optional: `current`, `superseded`, `disputed`), `subject` (optional).

### `GET /api/memories/current`
Fetch the active current version of a fact.
- Query parameters: `subject=user&attribute=deadline`

### `GET /api/memories/historical`
Point-in-time lookup for the fact valid at a historical timestamp.
- Query parameters: `subject=user&attribute=deadline&timestamp=2026-11-02T10:00:00Z`

### `GET /api/memories/history`
Retrieve the full chronological version history chain for an attribute.
- Query parameters: `subject=user&attribute=deadline`

---

## 3. Retrieval & Agent Query Endpoints

### `POST /api/memories/search`
Perform hybrid semantic and temporal retrieval.

**Request Body:**
```json
{
  "query": "What is my current deadline?",
  "top_k": 5
}
```

### `POST /api/agent/query`
End-to-end question answering with context budget packing.

**Request Body:**
```json
{
  "query": "What is my current deadline now?",
  "budget_tokens": 512,
  "top_k": 5
}
```

**Response (200 OK):**
```json
{
  "query": "What is my current deadline now?",
  "temporal_intent": "current",
  "response": "[Mock LLM] Currently, the deadline is 'November 20'.",
  "provider": "mock",
  "is_mock": true,
  "used_tokens": 24,
  "budget_tokens": 512,
  "budget_respected": true,
  "candidate_count": 2,
  "selected_count": 1,
  "selected_memories": [ ... ],
  "rejected_memories": [ ... ]
}
```
