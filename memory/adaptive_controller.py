"""
Lightweight tabular contextual bandit with UCB exploration,
inspired by MEMCON (Jiang et al., 2026), for adaptive retrieval parameter selection.
Optional extension that runs on top of the deterministic temporal engine.
"""

import json
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from memory.schema import QueryTemporalIntent


class AdaptiveMemoryController:
    """
    Learns an online policy mapping (query_intent, memory_size_bin, budget_pressure)
    to discrete retrieval presets via Upper Confidence Bound (UCB).
    Requires zero additional LLM inference.
    """

    ACTIONS = [
        {"id": 0, "name": "conservative_current", "top_k": 3, "w_temp": 0.5, "w_sim": 0.3},
        {"id": 1, "name": "balanced_hybrid", "top_k": 5, "w_temp": 0.35, "w_sim": 0.35},
        {"id": 2, "name": "deep_history", "top_k": 8, "w_temp": 0.6, "w_sim": 0.2},
        {"id": 3, "name": "strict_budget", "top_k": 2, "w_temp": 0.4, "w_sim": 0.4},
    ]

    def __init__(self, c_ucb: float = 1.0, alpha: float = 0.1, save_path: str = "data/q_table.json"):
        self.c_ucb = c_ucb
        self.alpha = alpha
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # State-action value table: Q[state_key][action_id]
        self.q_table: Dict[str, List[float]] = {}
        # Visit count table: N[state_key][action_id]
        self.n_table: Dict[str, List[int]] = {}
        
        self.load_policy()

    def _discretize_state(self, intent: QueryTemporalIntent, memory_count: int, budget: int) -> str:
        """Discretize continuous state into coarse hashable key."""
        # Memory size bin
        if memory_count < 10:
            m_bin = "sparse"
        elif memory_count < 50:
            m_bin = "moderate"
        else:
            m_bin = "dense"

        # Budget pressure bin
        b_bin = "tight" if budget < 500 else "spacious"

        return f"{intent.value}:{m_bin}:{b_bin}"

    def select_action(self, intent: QueryTemporalIntent, memory_count: int, budget: int) -> Dict[str, any]:
        """UCB action selection rule."""
        state_key = self._discretize_state(intent, memory_count, budget)
        num_actions = len(self.ACTIONS)

        if state_key not in self.q_table:
            # Warm-start prior: slight preference for balanced_hybrid (action 1)
            self.q_table[state_key] = [0.2, 0.5, 0.1, 0.1]
            self.n_table[state_key] = [1, 1, 1, 1]

        q_vals = self.q_table[state_key]
        n_vals = self.n_table[state_key]
        total_visits = sum(n_vals)

        # UCB formula
        ucb_scores = []
        for a_idx in range(num_actions):
            if n_vals[a_idx] == 0:
                score = float("inf")
            else:
                bonus = self.c_ucb * math.sqrt(math.log(total_visits) / n_vals[a_idx])
                score = q_vals[a_idx] + bonus
            ucb_scores.append(score)

        chosen_action_id = int(max(range(num_actions), key=lambda i: ucb_scores[i]))
        return self.ACTIONS[chosen_action_id]

    def update_policy(self, intent: QueryTemporalIntent, memory_count: int, budget: int, action_id: int, reward: float) -> None:
        """Update Q-values with observed reward."""
        state_key = self._discretize_state(intent, memory_count, budget)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.2, 0.5, 0.1, 0.1]
            self.n_table[state_key] = [1, 1, 1, 1]

        # Update visit count
        self.n_table[state_key][action_id] += 1
        
        # Incremental sample update
        old_q = self.q_table[state_key][action_id]
        self.q_table[state_key][action_id] = round(old_q + self.alpha * (reward - old_q), 4)

    def save_policy(self) -> None:
        """Persist learned Q-table to disk."""
        data = {
            "q_table": self.q_table,
            "n_table": self.n_table
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_policy(self) -> None:
        """Load persisted Q-table if present."""
        if self.save_path.exists():
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.q_table = data.get("q_table", {})
                    self.n_table = data.get("n_table", {})
            except Exception:
                pass
