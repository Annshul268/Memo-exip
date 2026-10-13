"""
Synthetic benchmark dataset generator for temporal memory evaluation.
Creates 10 reproducible scenarios with ground truth temporal states and queries.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any


def generate_benchmark_dataset() -> List[Dict[str, Any]]:
    """
    Constructs 10 standardized evaluation scenarios representing distinct
    temporal memory challenges.
    """
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    scenarios = []

    # Scenario 1: Single Fact Remembered Across Long Sequence
    scenarios.append({
        "scenario_id": "scen_01_single_fact",
        "name": "Single Fact Retention",
        "description": "Fact introduced early and queried after long conversational sequence.",
        "updates": [
            {"subject": "user", "attribute": "favorite_editor", "value": "Neovim", "valid_from": base_time}
        ],
        "distractors": [
            {"subject": "environment", "attribute": "weather", "value": "Sunny", "valid_from": base_time + timedelta(hours=1)},
            {"subject": "project", "attribute": "language", "value": "Python", "valid_from": base_time + timedelta(hours=2)},
            {"subject": "user", "attribute": "coffee_preference", "value": "Espresso", "valid_from": base_time + timedelta(hours=3)},
        ],
        "query": "What is my favorite editor?",
        "query_intent": "current",
        "expected_value": "Neovim",
        "stale_values": []
    })

    # Scenario 2 & 3: Multi-Update Evolution & Current-Fact Query
    scenarios.append({
        "scenario_id": "scen_02_fact_evolution_current",
        "name": "Fact Evolution - Current Query",
        "description": "Fact updated 3 times; query targets the latest active version.",
        "updates": [
            {"subject": "project", "attribute": "deadline", "value": "October 15", "valid_from": base_time},
            {"subject": "project", "attribute": "deadline", "value": "November 10", "valid_from": base_time + timedelta(days=10)},
            {"subject": "project", "attribute": "deadline", "value": "November 20", "valid_from": base_time + timedelta(days=20)}
        ],
        "distractors": [],
        "query": "What is the current project deadline now?",
        "query_intent": "current",
        "expected_value": "November 20",
        "stale_values": ["October 15", "November 10"]
    })

    # Scenario 4: Historical-Fact Query
    scenarios.append({
        "scenario_id": "scen_03_historical_query",
        "name": "Historical Fact Retrieval",
        "description": "Fact updated multiple times; query asks for the earlier superseded value.",
        "updates": [
            {"subject": "user", "attribute": "office", "value": "Room 101", "valid_from": base_time},
            {"subject": "user", "attribute": "office", "value": "Room 205", "valid_from": base_time + timedelta(days=30)},
            {"subject": "user", "attribute": "office", "value": "Room 404", "valid_from": base_time + timedelta(days=60)}
        ],
        "distractors": [],
        "query": "What was my original office before I moved?",
        "query_intent": "historical",
        "expected_value": "Room 101",
        "stale_values": []  # For historical query, Room 101 is correct, Room 404 is NOT the previous one
    })

    # Scenario 5: Timeline Progression Query
    scenarios.append({
        "scenario_id": "scen_04_timeline_progression",
        "name": "Timeline Change History",
        "description": "Query requesting the full chronological evolution of a fact.",
        "updates": [
            {"subject": "system", "attribute": "status", "value": "Draft", "valid_from": base_time},
            {"subject": "system", "attribute": "status", "value": "Under Review", "valid_from": base_time + timedelta(days=5)},
            {"subject": "system", "attribute": "status", "value": "Approved", "valid_from": base_time + timedelta(days=12)}
        ],
        "distractors": [],
        "query": "What is the timeline history of the system status?",
        "query_intent": "timeline",
        "expected_value": "Draft",  # All should be in context
        "stale_values": []
    })

    # Scenario 6: Contradictory Statements with Ambiguity
    scenarios.append({
        "scenario_id": "scen_05_contradiction_handling",
        "name": "Contradictory Statements",
        "description": "Two conflicting assertions without clear temporal progression.",
        "updates": [
            {"subject": "server", "attribute": "ip_address", "value": "192.168.1.50", "valid_from": base_time},
            {"subject": "server", "attribute": "ip_address", "value": "10.0.0.1", "valid_from": base_time, "is_explicit_correction": False}
        ],
        "distractors": [],
        "query": "What is the server ip address?",
        "query_intent": "current",
        "expected_value": "192.168.1.50",
        "stale_values": []
    })

    # Scenario 7: Redundant Memory Flooding
    scenarios.append({
        "scenario_id": "scen_06_redundancy_flooding",
        "name": "Redundant Memory Flooding",
        "description": "Identical fact stated repeatedly with slightly different phrasing.",
        "updates": [
            {"subject": "user", "attribute": "diet", "value": "Vegetarian", "valid_from": base_time},
            {"subject": "user", "attribute": "diet", "value": "Vegetarian", "valid_from": base_time + timedelta(days=2)},
            {"subject": "user", "attribute": "diet", "value": "Vegetarian", "valid_from": base_time + timedelta(days=4)},
        ],
        "distractors": [],
        "query": "What is my current dietary preference?",
        "query_intent": "current",
        "expected_value": "Vegetarian",
        "stale_values": []
    })

    # Scenario 8: Stable Invariant Facts
    scenarios.append({
        "scenario_id": "scen_07_stable_invariants",
        "name": "Stable Invariant Facts",
        "description": "Permanent fact queried amidst frequent volatile updates.",
        "updates": [
            {"subject": "user", "attribute": "date_of_birth", "value": "March 15", "valid_from": base_time, "importance": 0.9},
            {"subject": "user", "attribute": "current_task", "value": "Writing report", "valid_from": base_time + timedelta(days=1)},
            {"subject": "user", "attribute": "current_task", "value": "Reviewing PR", "valid_from": base_time + timedelta(days=2)},
            {"subject": "user", "attribute": "current_task", "value": "In a meeting", "valid_from": base_time + timedelta(days=3)},
        ],
        "distractors": [],
        "query": "What is my date of birth?",
        "query_intent": "current",
        "expected_value": "March 15",
        "stale_values": []
    })

    # Scenario 9: Multi-Fact Interdependent Query
    scenarios.append({
        "scenario_id": "scen_08_multi_fact_query",
        "name": "Multi-Fact Composition",
        "description": "Answering requires retrieving two distinct temporal facts.",
        "updates": [
            {"subject": "event", "attribute": "location", "value": "Hall B", "valid_from": base_time},
            {"subject": "event", "attribute": "location", "value": "Auditorium 1", "valid_from": base_time + timedelta(days=1)},
            {"subject": "event", "attribute": "start_time", "value": "09:00 AM", "valid_from": base_time},
            {"subject": "event", "attribute": "start_time", "value": "10:30 AM", "valid_from": base_time + timedelta(days=1)},
        ],
        "distractors": [],
        "query": "Where and at what time does the event take place now?",
        "query_intent": "current",
        "expected_value": "Auditorium 1",
        "stale_values": ["Hall B", "09:00 AM"]
    })

    # Scenario 10: Context Budget Pressure
    distractors_heavy = []
    for i in range(15):
        distractors_heavy.append({
            "subject": "irrelevant_topic",
            "attribute": f"metric_{i}",
            "value": f"Auxiliary telemetry observation number {i} with extensive filler details",
            "valid_from": base_time + timedelta(minutes=i)
        })
    scenarios.append({
        "scenario_id": "scen_09_budget_pressure",
        "name": "Context Budget Pressure",
        "description": "15 competing candidate facts under strict 250-token budget.",
        "updates": [
            {"subject": "user", "attribute": "emergency_contact", "value": "Alice 555-9999", "valid_from": base_time, "importance": 1.0}
        ],
        "distractors": distractors_heavy,
        "query": "What is my emergency contact?",
        "query_intent": "current",
        "expected_value": "Alice 555-9999",
        "stale_values": []
    })

    # Scenario 10b: Historical Ambiguity vs Latest
    scenarios.append({
        "scenario_id": "scen_10_ambiguity_vs_latest",
        "name": "Point-in-Time Historical Verification",
        "description": "Verifying point-in-time value at exact intermediate date.",
        "updates": [
            {"subject": "subscription", "attribute": "tier", "value": "Free", "valid_from": base_time},
            {"subject": "subscription", "attribute": "tier", "value": "Pro", "valid_from": base_time + timedelta(days=30)},
            {"subject": "subscription", "attribute": "tier", "value": "Enterprise", "valid_from": base_time + timedelta(days=60)},
        ],
        "distractors": [],
        "query": "What was my subscription tier before I upgraded to Enterprise?",
        "query_intent": "historical",
        "expected_value": "Pro",
        "stale_values": []
    })

    return scenarios
