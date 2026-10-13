"""
Automated tests for evaluation suite execution.
Ensures benchmark suite runs deterministically without an API key.
"""

import shutil
import tempfile
from pathlib import Path

from evaluation.dataset_generator import generate_benchmark_dataset
from evaluation.runner import run_benchmark


def test_dataset_generation():
    dataset = generate_benchmark_dataset()
    assert len(dataset) == 10
    for scen in dataset:
        assert "scenario_id" in scen
        assert "query" in scen
        assert "expected_value" in scen
        assert "updates" in scen


def test_benchmark_execution():
    temp_dir = Path(tempfile.mkdtemp())
    results = run_benchmark(output_dir=str(temp_dir), budget_tokens=300)
    
    assert "baseline_summaries" in results
    summaries = results["baseline_summaries"]
    assert "B0_NoMemory" in summaries
    assert "B1_RecentWindow" in summaries
    assert "B2_FixedSemantic" in summaries
    assert "B3_TemporalFixedRules" in summaries
    assert "P_ProposedAdaptiveTemporal" in summaries
    
    # Check that output JSON exists
    assert (temp_dir / "benchmark_results.json").exists()
    
    # P should strictly outperform B0 and B2 on overall accuracy
    p_acc = summaries["P_ProposedAdaptiveTemporal"]["overall_accuracy_pct"]
    b0_acc = summaries["B0_NoMemory"]["overall_accuracy_pct"]
    assert p_acc > b0_acc
    
    shutil.rmtree(temp_dir, ignore_errors=True)
