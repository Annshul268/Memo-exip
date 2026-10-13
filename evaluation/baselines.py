"""
Baseline implementations for comparative experimental evaluation:
- B0: No persistent memory
- B1: Recent-history window buffer
- B2: Fixed semantic vector retrieval (ChromaDB top-k, no temporal validity)
- B3: Temporal retrieval with fixed deterministic rules
- P: Proposed Adaptive Temporal Memory Manager
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from memory.schema import MemoryRecord, MemoryStatus
from memory.sqlite_store import SQLiteTemporalStore
from memory.vector_store import ChromaVectorStore
from memory.temporal_engine import TemporalMemoryEngine
from memory.hybrid_retriever import HybridTemporalRetriever
from memory.context_budget import ContextBudgetManager, estimate_tokens
from memory.adaptive_controller import AdaptiveMemoryController
from llm.mock_provider import MockLLMProvider


class BaseMemoryBaseline(ABC):
    """Abstract interface for benchmarked memory systems."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def ingest(self, fact_dict: Dict[str, Any]) -> float:
        """Ingest fact and return elapsed latency in milliseconds."""
        pass

    @abstractmethod
    def query(self, query_text: str, budget_tokens: int) -> Dict[str, Any]:
        """Process query and return response, used tokens, and retrieval latency."""
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class B0_NoMemory(BaseMemoryBaseline):
    """B0: Agent receives only the current query with zero persistent memory."""

    def __init__(self):
        self.llm = MockLLMProvider()

    @property
    def name(self) -> str:
        return "B0_NoMemory"

    def ingest(self, fact_dict: Dict[str, Any]) -> float:
        return 0.0  # Does not store facts

    def query(self, query_text: str, budget_tokens: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        prompt = f"### User Query:\n{query_text}\n"
        resp = self.llm.generate(prompt=prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "response": resp,
            "retrieved_memories": [],
            "used_tokens": 0,
            "latency_ms": elapsed
        }

    def reset(self) -> None:
        pass


class B1_RecentWindow(BaseMemoryBaseline):
    """B1: Stores recent statements in a fixed sliding FIFO buffer (N=3)."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.buffer: List[str] = []
        self.llm = MockLLMProvider()

    @property
    def name(self) -> str:
        return "B1_RecentWindow"

    def ingest(self, fact_dict: Dict[str, Any]) -> float:
        t0 = time.perf_counter()
        line = f"{fact_dict['subject']} -> {fact_dict['attribute']}: {fact_dict['value']}"
        self.buffer.append(line)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return (time.perf_counter() - t0) * 1000

    def query(self, query_text: str, budget_tokens: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        context_str = "\n".join([f"- {m}" for m in self.buffer])
        prompt = f"### Retrieved Memories:\n{context_str}\n\n### User Query:\n{query_text}\n"
        used_tokens = estimate_tokens(context_str)
        resp = self.llm.generate(prompt=prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "response": resp,
            "retrieved_memories": self.buffer.copy(),
            "used_tokens": used_tokens,
            "latency_ms": elapsed
        }

    def reset(self) -> None:
        self.buffer.clear()


class B2_FixedSemanticRetrieval(BaseMemoryBaseline):
    """B2: Conventional ChromaDB vector retrieval without temporal validity filtering."""

    def __init__(self, persist_dir: str = "data/eval_b2_chroma"):
        self.vector = ChromaVectorStore(persist_dir=persist_dir, collection_name="b2_fixed_semantic")
        self.llm = MockLLMProvider()

    @property
    def name(self) -> str:
        return "B2_FixedSemantic"

    def ingest(self, fact_dict: Dict[str, Any]) -> float:
        t0 = time.perf_counter()
        rec_id = f"b2_{len(self.vector.collection.get().get('ids', []))}_{fact_dict['attribute']}"
        now = fact_dict.get("valid_from", datetime.now(timezone.utc))
        rec = MemoryRecord(
            id=rec_id,
            subject=fact_dict["subject"],
            attribute=fact_dict["attribute"],
            value=fact_dict["value"],
            valid_from=now,
            status=MemoryStatus.CURRENT  # Standard vector store treats everything as flat/current
        )
        self.vector.index_memory(rec)
        return (time.perf_counter() - t0) * 1000

    def query(self, query_text: str, budget_tokens: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = self.vector.search(query_text, top_k=5)
        # In B2, all retrieved items are dumped into context without temporal filtering
        retrieved_texts = []
        if results:
            data = self.vector.collection.get(ids=[r[0] for r in results])
            retrieved_texts = data.get("documents", [])

        context_str = "\n".join([f"- {doc}" for doc in retrieved_texts])
        used_tokens = estimate_tokens(context_str)
        prompt = f"### Retrieved Memories:\n{context_str}\n\n### User Query:\n{query_text}\n"
        resp = self.llm.generate(prompt=prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "response": resp,
            "retrieved_memories": retrieved_texts,
            "used_tokens": used_tokens,
            "latency_ms": elapsed
        }

    def reset(self) -> None:
        self.vector.clear()


class B3_TemporalFixedRules(BaseMemoryBaseline):
    """B3: Temporal hybrid store with fixed deterministic rules (no adaptive bandit)."""

    def __init__(self, db_path: str = "data/eval_b3.db", chroma_path: str = "data/eval_b3_chroma"):
        self.sqlite = SQLiteTemporalStore(db_path=db_path)
        self.vector = ChromaVectorStore(persist_dir=chroma_path, collection_name="b3_temporal")
        self.engine = TemporalMemoryEngine(sqlite_store=self.sqlite, vector_store=self.vector)
        self.retriever = HybridTemporalRetriever(sqlite_store=self.sqlite, vector_store=self.vector)
        self.llm = MockLLMProvider()

    @property
    def name(self) -> str:
        return "B3_TemporalFixedRules"

    def ingest(self, fact_dict: Dict[str, Any]) -> float:
        t0 = time.perf_counter()
        self.engine.add_or_update_fact(
            subject=fact_dict["subject"],
            attribute=fact_dict["attribute"],
            value=fact_dict["value"],
            valid_from=fact_dict.get("valid_from"),
            is_explicit_correction=fact_dict.get("is_explicit_correction", True)
        )
        return (time.perf_counter() - t0) * 1000

    def query(self, query_text: str, budget_tokens: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(query_text, top_k=5)
        # Select all retrieved items without knapsack budget packing
        docs = [c.record.to_natural_language() for c in candidates]
        context_str = "\n".join([f"- {d}" for d in docs])
        used_tokens = estimate_tokens(context_str)
        prompt = f"### Retrieved Memories:\n{context_str}\n\n### User Query:\n{query_text}\n"
        resp = self.llm.generate(prompt=prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "response": resp,
            "retrieved_memories": docs,
            "used_tokens": used_tokens,
            "latency_ms": elapsed
        }

    def reset(self) -> None:
        self.sqlite.clear()
        self.vector.clear()


class P_ProposedSystem(BaseMemoryBaseline):
    """P: Proposed Adaptive Temporal Memory Manager (Hybrid + Budget Knapsack + Redundancy + Bandit)."""

    def __init__(self, db_path: str = "data/eval_p.db", chroma_path: str = "data/eval_p_chroma"):
        self.sqlite = SQLiteTemporalStore(db_path=db_path)
        self.vector = ChromaVectorStore(persist_dir=chroma_path, collection_name="p_proposed")
        self.engine = TemporalMemoryEngine(sqlite_store=self.sqlite, vector_store=self.vector)
        self.retriever = HybridTemporalRetriever(sqlite_store=self.sqlite, vector_store=self.vector)
        self.budget_manager = ContextBudgetManager()
        self.controller = AdaptiveMemoryController(save_path="data/eval_p_qtable.json")
        self.llm = MockLLMProvider()

    @property
    def name(self) -> str:
        return "P_ProposedAdaptiveTemporal"

    def ingest(self, fact_dict: Dict[str, Any]) -> float:
        t0 = time.perf_counter()
        self.engine.add_or_update_fact(
            subject=fact_dict["subject"],
            attribute=fact_dict["attribute"],
            value=fact_dict["value"],
            valid_from=fact_dict.get("valid_from"),
            importance=fact_dict.get("importance", 0.5),
            is_explicit_correction=fact_dict.get("is_explicit_correction", True)
        )
        return (time.perf_counter() - t0) * 1000

    def query(self, query_text: str, budget_tokens: int) -> Dict[str, Any]:
        t0 = time.perf_counter()
        intent, _ = self.retriever.classifier.classify_intent(query_text)
        mem_count = len(self.sqlite.list_records())
        
        # Adaptive controller parameter selection
        action = self.controller.select_action(intent, mem_count, budget_tokens)
        top_k = action.get("top_k", 5)

        candidates = self.retriever.retrieve(query_text, top_k=top_k)
        selection = self.budget_manager.select_memories(candidates, budget_tokens=budget_tokens)

        docs = [c.record.to_natural_language() for c in selection.selected_memories]
        context_str = "\n".join([f"- {d}" for d in docs])
        prompt = f"### Retrieved Memories:\n{context_str}\n\n### User Query:\n{query_text}\n"
        resp = self.llm.generate(prompt=prompt)
        elapsed = (time.perf_counter() - t0) * 1000

        # Reward feedback to controller
        is_successful = 1.0 if any(c.temporal_relevance > 0.8 for c in selection.selected_memories) else 0.0
        self.controller.update_policy(intent, mem_count, budget_tokens, action["id"], reward=is_successful)

        return {
            "response": resp,
            "retrieved_memories": docs,
            "used_tokens": selection.used_tokens,
            "budget_respected": selection.budget_respected,
            "latency_ms": elapsed
        }

    def reset(self) -> None:
        self.sqlite.clear()
        self.vector.clear()
