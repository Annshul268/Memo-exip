"""
FastAPI route definitions for temporal memory operations.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.models import (
    AddMemoryRequest,
    SearchMemoryRequest,
    AgentQueryRequest,
    AgentQueryResponse,
    HealthResponse
)
from memory.schema import MemoryRecord, MemoryStatus
from memory.agent_pipeline import TemporalMemoryAgentPipeline

router = APIRouter()

# Global pipeline instance initialized by app lifespans or route fallback
_pipeline: Optional[TemporalMemoryAgentPipeline] = None


def get_pipeline() -> TemporalMemoryAgentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TemporalMemoryAgentPipeline()
    return _pipeline


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and diagnostic inspection."""
    pipeline = get_pipeline()
    records = pipeline.sqlite.list_records()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=pipeline.llm.provider_name,
        is_mock_llm=pipeline.llm.is_mock,
        sqlite_connected=True,
        chroma_connected=True,
        memory_count=len(records)
    )


@router.post("/api/memories", response_model=dict, status_code=status.HTTP_201_CREATED, tags=["Memories"])
def add_memory(req: AddMemoryRequest):
    """
    Ingest a fact or update an existing fact version-aware.
    Automatically handles superseding, historical preservation, or dispute flagging.
    """
    pipeline = get_pipeline()
    result = pipeline.ingest_fact(
        subject=req.subject,
        attribute=req.attribute,
        value=req.value,
        valid_from=req.valid_from,
        valid_until=req.valid_until,
        confidence=req.confidence,
        importance=req.importance,
        source=req.source,
        is_explicit_correction=req.is_explicit_correction
    )
    return {
        "message": f"Fact processed with action: {result['action']}",
        "action": result["action"],
        "record": result["record"].model_dump(),
        "status": result["status"],
        "supersedes_id": result["supersedes_id"]
    }


@router.get("/api/memories", response_model=List[dict], tags=["Memories"])
def list_memories(
    status: Optional[MemoryStatus] = None,
    subject: Optional[str] = None
):
    """List all stored memory records with optional status and subject filters."""
    pipeline = get_pipeline()
    records = pipeline.sqlite.list_records(status=status, subject=subject)
    return [r.model_dump() for r in records]


@router.get("/api/memories/current", response_model=dict, tags=["Memories"])
def get_current_fact(subject: str = Query(...), attribute: str = Query(...)):
    """Fetch the confirmed active current version of a fact."""
    pipeline = get_pipeline()
    record = pipeline.engine.get_current_fact(subject, attribute)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active current fact found for '{subject}' -> '{attribute}'"
        )
    return record.model_dump()


@router.get("/api/memories/historical", response_model=dict, tags=["Memories"])
def get_historical_fact(
    subject: str = Query(...),
    attribute: str = Query(...),
    timestamp: datetime = Query(...)
):
    """Point-in-time query: find fact valid at a historical timestamp."""
    pipeline = get_pipeline()
    record = pipeline.engine.get_historical_fact(subject, attribute, timestamp)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No valid fact found for '{subject}' -> '{attribute}' at {timestamp.isoformat()}"
        )
    return record.model_dump()


@router.get("/api/memories/history", response_model=List[dict], tags=["Memories"])
def get_fact_history(subject: str = Query(...), attribute: str = Query(...)):
    """Retrieve full chronological version history lineage for an attribute."""
    pipeline = get_pipeline()
    history = pipeline.engine.get_fact_lineage(subject, attribute)
    return [r.model_dump() for r in history]


@router.post("/api/memories/search", response_model=List[dict], tags=["Retrieval"])
def search_memories(req: SearchMemoryRequest):
    """Execute hybrid retrieval returning candidate memories with scores."""
    pipeline = get_pipeline()
    candidates = pipeline.retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        target_timestamp=req.target_timestamp
    )
    return [{
        "record": c.record.model_dump(),
        "semantic_similarity": c.semantic_similarity,
        "temporal_relevance": c.temporal_relevance,
        "estimated_tokens": c.estimated_tokens
    } for c in candidates]


@router.post("/api/agent/query", response_model=AgentQueryResponse, tags=["Agent"])
def agent_query(req: AgentQueryRequest):
    """
    End-to-end agent query:
    1. Classifies temporal intent
    2. Performs hybrid retrieval
    3. Selects memories under token budget
    4. Generates grounded LLM response
    5. Returns transparent inspection data
    """
    pipeline = get_pipeline()
    result = pipeline.query(
        user_query=req.query,
        budget_tokens=req.budget_tokens,
        target_timestamp=req.target_timestamp,
        top_k=req.top_k
    )
    
    sel = result["selection_result"]
    return AgentQueryResponse(
        query=result["query"],
        temporal_intent=result["temporal_intent"],
        response=result["response"],
        provider=result["provider"],
        is_mock=result["is_mock"],
        used_tokens=result["used_tokens"],
        budget_tokens=result["budget_tokens"],
        budget_respected=result["budget_respected"],
        candidate_count=result["candidate_count"],
        selected_count=result["selected_count"],
        selected_memories=[{
            "record": m.record.model_dump(),
            "utility_score": m.utility_score,
            "estimated_tokens": m.estimated_tokens
        } for m in sel.selected_memories],
        rejected_memories=[{
            "record": m.record.model_dump(),
            "utility_score": m.utility_score,
            "rejection_reason": m.rejection_reason,
            "estimated_tokens": m.estimated_tokens
        } for m in sel.rejected_memories]
    )


@router.post("/api/memories/reconcile", response_model=dict, tags=["System"])
def reconcile_storage():
    """Resynchronize ChromaDB vector index from authoritative SQLite store."""
    pipeline = get_pipeline()
    stats = pipeline.engine.reconcile_storage()
    return {"message": "Storage reconciliation complete", "stats": stats}


@router.post("/api/memories/reset", response_model=dict, tags=["System"])
def reset_memories(confirm: bool = Query(False, description="Must be set to true to execute reset")):
    """Reset all memory storage (SQLite and ChromaDB). Requires explicit confirmation."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset requires explicit confirmation flag 'confirm=true'."
        )
    pipeline = get_pipeline()
    pipeline.reset()
    return {"message": "All memories and vector indices have been purged successfully."}


@router.post("/api/evaluation/run", response_model=dict, tags=["Evaluation"])
def run_evaluation_benchmark(budget_tokens: int = Query(500, ge=100, le=2048)):
    """Execute reproducible benchmark evaluation across all 5 baselines."""
    from evaluation.runner import run_benchmark
    results = run_benchmark(output_dir="results", budget_tokens=budget_tokens)
    return {
        "message": "Benchmark evaluation executed successfully",
        "metadata": results.get("metadata"),
        "baseline_summaries": results.get("baseline_summaries")
    }


@router.get("/api/evaluation/results", response_model=dict, tags=["Evaluation"])
def get_evaluation_results():
    """Retrieve raw benchmark evaluation results from storage."""
    results_file = Path("results/benchmark_results.json")
    if not results_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark results found. Run /api/evaluation/run first."
        )
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)
