"""
Utility-based, context-budget-aware memory selection.
Implements token-bounded knapsack-style selection with redundancy control.
"""

import math
from typing import List, Dict, Set, Optional

from memory.schema import MemoryCandidate, ContextSelectionResult, MemoryRecord


def estimate_tokens(text: str) -> int:
    """
    Conservative token count estimator when exact BPE tokenizer is unavailable.
    Approximation: ~3.8 characters per token, minimum 1 token per word.
    """
    if not text:
        return 0
    char_estimate = math.ceil(len(text) / 3.8)
    word_estimate = math.ceil(len(text.split()) * 1.3)
    return max(1, max(char_estimate, word_estimate))


class ContextBudgetManager:
    """Selects top memories within a strict token budget using utility-per-token."""

    def __init__(
        self,
        w_sim: float = 0.35,
        w_temp: float = 0.35,
        w_imp: float = 0.15,
        w_conf: float = 0.15,
        redundancy_threshold: float = 0.75,
        redundancy_penalty: float = 0.5
    ):
        self.w_sim = w_sim
        self.w_temp = w_temp
        self.w_imp = w_imp
        self.w_conf = w_conf
        self.redundancy_threshold = redundancy_threshold
        self.redundancy_penalty = redundancy_penalty

    def calculate_utility(
        self,
        candidate: MemoryCandidate,
        already_selected_entities: Set[str]
    ) -> float:
        """
        Calculates composite memory utility score in [0.0, 1.0].
        Applies a redundancy penalty if the entity (subject, attribute) was already selected.
        """
        rec = candidate.record
        base_utility = (
            self.w_sim * candidate.semantic_similarity +
            self.w_temp * candidate.temporal_relevance +
            self.w_imp * rec.importance +
            self.w_conf * rec.confidence
        )

        # Redundancy check: if identical subject & attribute is already present
        entity_key = f"{rec.subject}::{rec.attribute}"
        if entity_key in already_selected_entities:
            base_utility = max(0.0, base_utility - self.redundancy_penalty)

        return round(base_utility, 4)

    def select_memories(
        self,
        candidates: List[MemoryCandidate],
        budget_tokens: int,
        prompt_overhead_tokens: int = 150
    ) -> ContextSelectionResult:
        """
        Greedy utility-per-token selection under context budget:
        1. Estimates token footprint for each candidate.
        2. Computes utility score with dynamic redundancy penalty.
        3. Packs candidates into the available token budget.
        4. Categorizes unselected candidates with explicit rejection reasons.
        """
        effective_budget = max(0, budget_tokens - prompt_overhead_tokens)
        
        # 1. Compute tokens and initial scores
        for c in candidates:
            c.estimated_tokens = estimate_tokens(c.record.to_natural_language())

        # 2. Sort by raw utility descending to prioritize best candidates
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                self.w_sim * c.semantic_similarity +
                self.w_temp * c.temporal_relevance +
                self.w_imp * c.record.importance +
                self.w_conf * c.record.confidence
            ) / max(1, c.estimated_tokens),
            reverse=True
        )

        selected: List[MemoryCandidate] = []
        rejected: List[MemoryCandidate] = []
        used_tokens = 0
        selected_entities: Set[str] = set()

        for cand in sorted_candidates:
            # Recompute utility considering currently selected set
            cand.utility_score = self.calculate_utility(cand, selected_entities)
            entity_key = f"{cand.record.subject}::{cand.record.attribute}"

            # Check if this candidate is duplicate/redundant with an already accepted version
            is_redundant = entity_key in selected_entities and cand.utility_score < 0.25

            if is_redundant:
                cand.rejection_reason = "redundant_subsumed_by_higher_utility_version"
                rejected.append(cand)
                continue

            if used_tokens + cand.estimated_tokens <= effective_budget:
                selected.append(cand)
                used_tokens += cand.estimated_tokens
                selected_entities.add(entity_key)
            else:
                cand.rejection_reason = f"budget_exceeded (needed {cand.estimated_tokens}, remaining {effective_budget - used_tokens})"
                rejected.append(cand)

        return ContextSelectionResult(
            selected_memories=selected,
            rejected_memories=rejected,
            total_budget_tokens=budget_tokens,
            used_tokens=used_tokens,
            remaining_tokens=max(0, effective_budget - used_tokens),
            budget_respected=used_tokens <= effective_budget
        )
