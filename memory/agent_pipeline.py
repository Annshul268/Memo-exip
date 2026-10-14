"""
End-to-End Agent Memory Pipeline.
Connects Query Intent -> Hybrid Retrieval -> Context Selection -> LLM Response.
"""

import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from memory.schema import MemoryRecord, ContextSelectionResult, QueryTemporalIntent
from memory.sqlite_store import SQLiteTemporalStore
from memory.vector_store import ChromaVectorStore
from memory.temporal_engine import TemporalMemoryEngine
from memory.hybrid_retriever import HybridTemporalRetriever, TemporalIntentClassifier
from memory.context_budget import ContextBudgetManager, estimate_tokens
from llm.base import BaseLLMProvider
from llm.factory import get_llm_provider


class TemporalMemoryAgentPipeline:
    """Orchestrates query answering and fact ingestion with temporal context."""

    def __init__(
        self,
        sqlite_path: str = "data/temporal_memory.db",
        chroma_path: str = "data/chroma_db",
        llm_provider: Optional[BaseLLMProvider] = None,
        default_budget_tokens: int = 1024
    ):
        self.sqlite = SQLiteTemporalStore(db_path=sqlite_path)
        self.vector = ChromaVectorStore(persist_dir=chroma_path)
        self.engine = TemporalMemoryEngine(sqlite_store=self.sqlite, vector_store=self.vector)
        self.retriever = HybridTemporalRetriever(sqlite_store=self.sqlite, vector_store=self.vector)
        self.budget_manager = ContextBudgetManager()
        self.llm = llm_provider or get_llm_provider()
        self.default_budget_tokens = default_budget_tokens

    def ingest_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        source: str = "user_chat",
        is_explicit_correction: bool = True
    ) -> Dict[str, Any]:
        """Ingest a new fact or update an existing fact version-aware."""
        record, action = self.engine.add_or_update_fact(
            subject=subject,
            attribute=attribute,
            value=value,
            valid_from=valid_from,
            valid_until=valid_until,
            confidence=confidence,
            importance=importance,
            source=source,
            is_explicit_correction=is_explicit_correction
        )
        return {
            "record": record,
            "action": action,
            "status": record.status.value,
            "supersedes_id": record.supersedes_id
        }

    def ingest_natural_statement(self, text: str) -> Dict[str, Any]:
        """
        Heuristic rule-based fact extractor for natural language statements:
        e.g., 'My deadline is November 20' -> subject='user', attribute='deadline', value='November 20'
        e.g., 'The project manager is Alice' -> subject='project', attribute='manager', value='Alice'
        """
        clean_text = text.strip()
        
        # Patterns for common assertions
        # 1. "my <attribute> is <value>"
        m1 = re.match(r"(?:my|the)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:is|has been changed to|is now|was changed to)\s+(.+)", clean_text, re.IGNORECASE)
        if m1:
            attr = m1.group(1).strip().lower()
            val = m1.group(2).strip().rstrip(".")
            return self.ingest_fact(subject="user", attribute=attr, value=val)

        # 2. "<subject>'s <attribute> is <value>"
        m2 = re.match(r"([a-zA-Z0-9_\-]+)'s\s+([a-zA-Z0-9_\-\s]+?)\s+(?:is|is now)\s+(.+)", clean_text, re.IGNORECASE)
        if m2:
            subj = m2.group(1).strip().lower()
            attr = m2.group(2).strip().lower()
            val = m2.group(3).strip().rstrip(".")
            return self.ingest_fact(subject=subj, attribute=attr, value=val)

        # Fallback: treat entire statement as subject='user', attribute='general_note', value=text
        return self.ingest_fact(subject="user", attribute="note", value=clean_text)

    def query(
        self,
        user_query: str,
        budget_tokens: Optional[int] = None,
        target_timestamp: Optional[datetime] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        End-to-end query processing:
        1. Classifies intent.
        2. Retrieves candidate memories.
        3. Packs memories into context budget.
        4. Synthesizes response via LLM.
        """
        budget = budget_tokens or self.default_budget_tokens
        intent, _ = TemporalIntentClassifier.classify_intent(user_query)

        # 1. Hybrid retrieval
        candidates = self.retriever.retrieve(
            query=user_query,
            top_k=top_k,
            target_timestamp=target_timestamp
        )

        # 2. Context budget selection
        selection: ContextSelectionResult = self.budget_manager.select_memories(
            candidates=candidates,
            budget_tokens=budget,
            prompt_overhead_tokens=150
        )

        # 3. Build grounded prompt
        memory_context_str = "\n".join([
            f"- {c.record.to_natural_language()}"
            for c in selection.selected_memories
        ]) if selection.selected_memories else "(No memories selected)"

        system_prompt = (
            "You are an AI assistant equipped with an adaptive temporal memory system. "
            "Your task is to accurately answer user queries grounded exclusively on the retrieved memories. "
            "Pay strict attention to timestamps, status (current vs superseded), and temporal query intent."
        )

        prompt = (
            f"### Retrieved Memories:\n{memory_context_str}\n\n"
            f"### User Query:\n{user_query}\n\n"
            f"### Instructions:\n"
            f"Answer the query accurately based on the memories above. "
            f"If answering about the current state, use the latest valid current fact. "
            f"If asked about a historical or previous state, explicitly reference the superseded historical fact. "
            f"If contradictory facts are marked disputed, mention the ambiguity."
        )

        # 4. Generate response
        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0
        )

        return {
            "query": user_query,
            "temporal_intent": intent.value,
            "response": llm_response,
            "provider": self.llm.provider_name,
            "is_mock": self.llm.is_mock,
            "selection_result": selection,
            "candidate_count": len(candidates),
            "selected_count": len(selection.selected_memories),
            "used_tokens": selection.used_tokens,
            "budget_tokens": budget,
            "budget_respected": selection.budget_respected
        }

    def reset(self) -> None:
        """Purge all stores for clean evaluation or testing."""
        self.sqlite.clear()
        self.vector.clear()
