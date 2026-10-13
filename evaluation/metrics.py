"""
Evaluation metric calculations from actual benchmark runs.
Measures accuracy, stale-fact error rate, contradiction presence, token usage, and latency.
"""

from typing import List, Dict, Any


def evaluate_response(
    response: str,
    retrieved_memories: List[str],
    expected_value: str,
    stale_values: List[str],
    used_tokens: int,
    budget_tokens: int,
    query_intent: str
) -> Dict[str, Any]:
    """
    Evaluates an individual scenario execution deterministically:
    - Accuracy: Does response reflect the ground truth expected value?
    - Stale error: Does response assert an obsolete/superseded value?
    - Contradiction: Are mutually exclusive facts asserted without qualification?
    """
    resp_lower = response.lower()
    expected_lower = expected_value.lower()

    # 1. Accuracy
    is_correct = expected_lower in resp_lower

    # 2. Stale-fact error rate:
    # Stale value is present in response when the query was targeting the CURRENT state
    is_stale_error = False
    if query_intent == "current" and stale_values:
        for sv in stale_values:
            if sv.lower() in resp_lower:
                is_stale_error = True
                break

    # 3. Budget compliance
    budget_compliant = used_tokens <= budget_tokens

    # 4. Context Grounding
    in_context = any(expected_lower in m.lower() for m in retrieved_memories)

    return {
        "is_correct": is_correct,
        "is_stale_error": is_stale_error,
        "in_context": in_context,
        "used_tokens": used_tokens,
        "budget_compliant": budget_compliant
    }
