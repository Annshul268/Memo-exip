"""
Integration test for end-to-end agent pipeline in deterministic mock mode.
Tests the deadline update scenario from Section 2.B of the research specification.
"""

import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from memory.agent_pipeline import TemporalMemoryAgentPipeline
from llm.mock_provider import MockLLMProvider


def test_deadline_update_end_to_end():
    temp_dir = Path(tempfile.mkdtemp())
    sqlite_p = str(temp_dir / "pipeline_test.db")
    chroma_p = str(temp_dir / "pipeline_chroma")
    
    mock_llm = MockLLMProvider()
    pipeline = TemporalMemoryAgentPipeline(
        sqlite_path=sqlite_p,
        chroma_path=chroma_p,
        llm_provider=mock_llm
    )
    
    # 1. User states: deadline is November 10
    t0 = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)
    pipeline.ingest_fact("user", "deadline", "November 10", valid_from=t0)
    
    # Query current deadline
    res1 = pipeline.query("What is my current deadline?")
    assert "November 10" in res1["response"]
    assert res1["is_mock"] is True
    
    # 2. Later, user updates deadline to November 20
    t1 = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
    pipeline.ingest_fact("user", "deadline", "November 20", valid_from=t1)
    
    # 3. Query current deadline -> Must return November 20
    res2 = pipeline.query("What is my current deadline now?")
    assert "November 20" in res2["response"]
    assert "November 10" not in res2["response"]  # Obsolete fact should not be claimed as current
    
    # 4. Query earlier deadline -> Must return November 10
    res3 = pipeline.query("What was my previous deadline before?")
    assert "November 10" in res3["response"]
    
    shutil.rmtree(temp_dir, ignore_errors=True)
