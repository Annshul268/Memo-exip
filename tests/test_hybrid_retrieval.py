"""
Unit tests for hybrid retrieval, intent classification, and temporal filtering.
"""

import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from memory.schema import QueryTemporalIntent, MemoryStatus
from memory.sqlite_store import SQLiteTemporalStore
from memory.vector_store import ChromaVectorStore
from memory.temporal_engine import TemporalMemoryEngine
from memory.hybrid_retriever import HybridTemporalRetriever, TemporalIntentClassifier


def test_intent_classification():
    # Current intent
    intent1, _ = TemporalIntentClassifier.classify_intent("What is my current address?")
    assert intent1 == QueryTemporalIntent.CURRENT
    
    intent2, _ = TemporalIntentClassifier.classify_intent("What is the latest deadline now?")
    assert intent2 == QueryTemporalIntent.CURRENT

    # Historical intent
    intent3, _ = TemporalIntentClassifier.classify_intent("What was my previous office before July?")
    assert intent3 == QueryTemporalIntent.HISTORICAL

    intent4, _ = TemporalIntentClassifier.classify_intent("Where did I originally live?")
    assert intent4 == QueryTemporalIntent.HISTORICAL

    # Timeline intent
    intent5, _ = TemporalIntentClassifier.classify_intent("Show me the timeline of my project status")
    assert intent5 == QueryTemporalIntent.TIMELINE


@pytest.fixture
def retrieval_stores():
    temp_dir = Path(tempfile.mkdtemp())
    sqlite_store = SQLiteTemporalStore(db_path=str(temp_dir / "retrieval_test.db"))
    vector_store = ChromaVectorStore(persist_dir=str(temp_dir / "retrieval_chroma"))
    engine = TemporalMemoryEngine(sqlite_store=sqlite_store, vector_store=vector_store)
    retriever = HybridTemporalRetriever(sqlite_store=sqlite_store, vector_store=vector_store)
    
    yield sqlite_store, vector_store, engine, retriever
    
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_current_vs_historical_query_retrieval(retrieval_stores):
    sqlite, vector, engine, retriever = retrieval_stores
    
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    
    # Old address
    engine.add_or_update_fact("user", "address", "100 Maple St", valid_from=t0)
    # New address
    engine.add_or_update_fact("user", "address", "200 Oak Ave", valid_from=t1)
    
    # 1. Query current address
    current_candidates = retriever.retrieve("What is my current address?")
    assert len(current_candidates) >= 1
    # Top candidate should have status == CURRENT
    top_current = current_candidates[0]
    assert top_current.record.status == MemoryStatus.CURRENT
    assert top_current.record.value == "200 Oak Ave"
    assert top_current.temporal_relevance > 0.8
    
    # 2. Query historical address
    historical_candidates = retriever.retrieve("What was my previous address before?")
    assert len(historical_candidates) >= 1
    # Candidates with status == SUPERSEDED should have high temporal relevance
    superseded_candidate = next((c for c in historical_candidates if c.record.status == MemoryStatus.SUPERSEDED), None)
    assert superseded_candidate is not None
    assert superseded_candidate.record.value == "100 Maple St"
    assert superseded_candidate.temporal_relevance >= 0.9
