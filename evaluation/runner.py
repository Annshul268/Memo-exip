"""
Benchmark Evaluation Runner.
Executes all baselines against the 10 benchmark scenarios,
records real measurements, and writes raw results to results/benchmark_results.json.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dataset_generator import generate_benchmark_dataset
from evaluation.baselines import (
    B0_NoMemory,
    B1_RecentWindow,
    B2_FixedSemanticRetrieval,
    B3_TemporalFixedRules,
    P_ProposedSystem
)
from evaluation.metrics import evaluate_response


def run_benchmark(output_dir: str = "results", budget_tokens: int = 500) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = generate_benchmark_dataset()
    baselines = [
        B0_NoMemory(),
        B1_RecentWindow(),
        B2_FixedSemanticRetrieval(),
        B3_TemporalFixedRules(),
        P_ProposedSystem()
    ]

    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "budget_tokens": budget_tokens,
            "scenario_count": len(dataset),
            "baseline_count": len(baselines)
        },
        "baseline_summaries": {},
        "raw_scenario_runs": []
    }

    print(f"Starting benchmark execution across {len(baselines)} baselines and {len(dataset)} scenarios...")

    for b in baselines:
        b_name = b.name
        print(f"--> Running baseline: {b_name}")
        b_runs = []
        
        total_correct = 0
        current_q_correct = 0
        current_q_total = 0
        historical_q_correct = 0
        historical_q_total = 0
        total_stale_errors = 0
        total_tokens = 0
        total_retrieval_latency = 0.0
        total_ingest_latency = 0.0
        ingest_count = 0
        compliant_count = 0

        for scen in dataset:
            b.reset()

            # 1. Ingest updates and distractors
            all_facts = scen["updates"] + scen.get("distractors", [])
            for fact in all_facts:
                lat = b.ingest(fact)
                total_ingest_latency += lat
                ingest_count += 1

            # 2. Query baseline
            query_res = b.query(scen["query"], budget_tokens=budget_tokens)
            total_retrieval_latency += query_res["latency_ms"]
            total_tokens += query_res["used_tokens"]

            # 3. Evaluate response
            eval_res = evaluate_response(
                response=query_res["response"],
                retrieved_memories=query_res["retrieved_memories"],
                expected_value=scen["expected_value"],
                stale_values=scen["stale_values"],
                used_tokens=query_res["used_tokens"],
                budget_tokens=budget_tokens,
                query_intent=scen["query_intent"]
            )

            if eval_res["is_correct"]:
                total_correct += 1

            if scen["query_intent"] == "current":
                current_q_total += 1
                if eval_res["is_correct"]:
                    current_q_correct += 1

            if scen["query_intent"] == "historical":
                historical_q_total += 1
                if eval_res["is_correct"]:
                    historical_q_correct += 1

            if eval_res["is_stale_error"]:
                total_stale_errors += 1

            if eval_res["budget_compliant"]:
                compliant_count += 1

            run_entry = {
                "baseline": b_name,
                "scenario_id": scen["scenario_id"],
                "query": scen["query"],
                "expected": scen["expected_value"],
                "response": query_res["response"],
                "latency_ms": round(query_res["latency_ms"], 2),
                **eval_res
            }
            b_runs.append(run_entry)
            all_results["raw_scenario_runs"].append(run_entry)

        # Aggregate metrics
        num_scenarios = len(dataset)
        summary = {
            "overall_accuracy_pct": round((total_correct / num_scenarios) * 100, 1),
            "current_fact_accuracy_pct": round((current_q_correct / max(1, current_q_total)) * 100, 1),
            "historical_fact_accuracy_pct": round((historical_q_correct / max(1, historical_q_total)) * 100, 1),
            "stale_fact_error_rate_pct": round((total_stale_errors / max(1, current_q_total)) * 100, 1),
            "avg_tokens_per_query": round(total_tokens / num_scenarios, 1),
            "budget_compliance_pct": round((compliant_count / num_scenarios) * 100, 1),
            "avg_retrieval_latency_ms": round(total_retrieval_latency / num_scenarios, 2),
            "avg_update_latency_ms": round(total_ingest_latency / max(1, ingest_count), 2)
        }
        all_results["baseline_summaries"][b_name] = summary

    # Save to JSON
    json_path = out_path / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark completed successfully! Raw results saved to: {json_path}")
    return all_results


if __name__ == "__main__":
    run_benchmark()
