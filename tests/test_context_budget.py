"""
Unit tests for context budget management and token packing.
"""

from datetime import datetime, timezone
from memory.schema import MemoryRecord, MemoryStatus, MemoryCandidate
from memory.context_budget import ContextBudgetManager, estimate_tokens


def test_token_estimation():
    short_text = "user -> address: 123 Main St"
    tokens = estimate_tokens(short_text)
    assert tokens > 0
    assert tokens < 20


def test_budget_enforcement():
    manager = ContextBudgetManager()
    
    # Create 5 candidates with different utilities and token costs
    candidates = []
    for i in range(5):
        rec = MemoryRecord(
            id=f"mem_{i}",
            subject="user",
            attribute=f"attr_{i}",
            value=f"Detailed informative value for attribute {i} that consumes reasonable token space",
            valid_from=datetime.now(timezone.utc),
            importance=0.5 + i * 0.1,
            confidence=1.0,
            status=MemoryStatus.CURRENT
        )
        cand = MemoryCandidate(
            record=rec,
            semantic_similarity=0.8,
            temporal_relevance=1.0,
            utility_score=0.0,
            estimated_tokens=estimate_tokens(rec.to_natural_language())
        )
        candidates.append(cand)

    # Set very small budget: enough for ~2 memories only
    small_budget = 60
    overhead = 10
    result = manager.select_memories(candidates, budget_tokens=small_budget, prompt_overhead_tokens=overhead)
    
    assert result.budget_respected is True
    assert result.used_tokens <= (small_budget - overhead)
    assert len(result.selected_memories) < 5
    assert len(result.rejected_memories) > 0
    
    # Check that rejection reason exists
    for rej in result.rejected_memories:
        assert rej.rejection_reason is not None
        assert "budget_exceeded" in rej.rejection_reason or "redundant" in rej.rejection_reason
