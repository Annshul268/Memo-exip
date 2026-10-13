"""
Deterministic unit tests for SQLite temporal store and versioning engine.
Tests ACID properties, version lineage, point-in-time queries, and conflict handling.
"""

import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from memory.schema import MemoryRecord, MemoryStatus
from memory.sqlite_store import SQLiteTemporalStore
from memory.vector_store import ChromaVectorStore
from memory.temporal_engine import TemporalMemoryEngine


@pytest.fixture
def temp_stores():
    """Create isolated temporary database and vector index for test execution."""
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test_memory.db"
    chroma_dir = temp_dir / "test_chroma"
    
    sqlite_store = SQLiteTemporalStore(db_path=str(db_path))
    vector_store = ChromaVectorStore(persist_dir=str(chroma_dir))
    engine = TemporalMemoryEngine(sqlite_store=sqlite_store, vector_store=vector_store)
    
    yield sqlite_store, vector_store, engine
    
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_add_and_retrieve_initial_fact(temp_stores):
    sqlite, vector, engine = temp_stores
    
    t0 = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)
    rec, action = engine.add_or_update_fact(
        subject="user",
        attribute="deadline",
        value="November 10",
        valid_from=t0
    )
    
    assert action == "created_initial_fact"
    assert rec.status == MemoryStatus.CURRENT
    assert rec.supersedes_id is None
    
    # Retrieve current
    current = engine.get_current_fact("user", "deadline")
    assert current is not None
    assert current.id == rec.id
    assert current.value == "November 10"
    assert current.status == MemoryStatus.CURRENT


def test_version_aware_update_preserves_history(temp_stores):
    sqlite, vector, engine = temp_stores
    
    # 1. Fact introduced
    t0 = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)
    rec1, action1 = engine.add_or_update_fact(
        subject="user",
        attribute="deadline",
        value="November 10",
        valid_from=t0
    )
    
    # 2. Fact updated later
    t1 = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
    rec2, action2 = engine.add_or_update_fact(
        subject="user",
        attribute="deadline",
        value="November 20",
        valid_from=t1,
        is_explicit_correction=True
    )
    
    assert "superseded_previous_version" in action2
    assert rec2.status == MemoryStatus.CURRENT
    assert rec2.supersedes_id == rec1.id
    
    # Verify old record is NOT deleted, but marked superseded
    archived = sqlite.get_record(rec1.id)
    assert archived is not None
    assert archived.status == MemoryStatus.SUPERSEDED
    assert archived.valid_until == t1
    assert archived.value == "November 10"
    
    # Verify current fact returns November 20
    current = engine.get_current_fact("user", "deadline")
    assert current is not None
    assert current.id == rec2.id
    assert current.value == "November 20"
    
    # Verify point-in-time historical query returns November 10 for date prior to t1
    historical = engine.get_historical_fact("user", "deadline", timestamp=t0 + timedelta(days=1))
    assert historical is not None
    assert historical.id == rec1.id
    assert historical.value == "November 10"


def test_multiple_sequential_updates_lineage(temp_stores):
    sqlite, vector, engine = temp_stores
    
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 2, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)
    
    r1, _ = engine.add_or_update_fact("user", "office", "Building A", valid_from=t0)
    r2, _ = engine.add_or_update_fact("user", "office", "Building B", valid_from=t1)
    r3, _ = engine.add_or_update_fact("user", "office", "Building C", valid_from=t2)
    
    lineage = engine.get_fact_lineage("user", "office")
    assert len(lineage) == 3
    
    values = [m.value for m in lineage]
    assert values == ["Building A", "Building B", "Building C"]
    
    assert lineage[0].status == MemoryStatus.SUPERSEDED
    assert lineage[1].status == MemoryStatus.SUPERSEDED
    assert lineage[2].status == MemoryStatus.CURRENT
    
    assert lineage[1].supersedes_id == lineage[0].id
    assert lineage[2].supersedes_id == lineage[1].id


def test_redundant_fact_reinforces_confidence(temp_stores):
    sqlite, vector, engine = temp_stores
    
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r1, a1 = engine.add_or_update_fact("user", "city", "Seattle", valid_from=t0, confidence=0.8)
    initial_access = r1.access_count
    
    # Re-state exact same fact
    r2, a2 = engine.add_or_update_fact("user", "city", "Seattle", valid_from=t0 + timedelta(days=1))
    assert a2 == "redundant_reinforced_existing_fact"
    assert r2.id == r1.id
    assert r2.access_count == initial_access + 1
    assert r2.confidence > 0.8
    
    # Verify no duplicate rows created
    history = engine.get_fact_lineage("user", "city")
    assert len(history) == 1


def test_ambiguous_conflict_preserves_uncertainty(temp_stores):
    sqlite, vector, engine = temp_stores
    
    t0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
    r1, _ = engine.add_or_update_fact("user", "phone", "555-0100", valid_from=t0)
    
    # Ingestion with ambiguous flag (not confirmed correction)
    r2, action = engine.add_or_update_fact(
        "user", "phone", "555-0999",
        valid_from=t0,
        is_explicit_correction=False
    )
    
    assert "ambiguous_conflict_detected" in action
    assert r2.status == MemoryStatus.DISPUTED
    
    # Current record remains intact and not silently overwritten
    current = engine.get_current_fact("user", "phone")
    assert current.value == "555-0100"
    assert current.status == MemoryStatus.CURRENT
