"""
Hybrid retrieval pipeline.
Combines ChromaDB vector semantic similarity with SQLite authoritative
temporal filtering and query intent classification.
"""

import re
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

from memory.schema import MemoryRecord, MemoryStatus, QueryTemporalIntent, MemoryCandidate
from memory.sqlite_store import SQLiteTemporalStore
from memory.vector_store import ChromaVectorStore


class TemporalIntentClassifier:
    """Classifies temporal intent from query text using deterministic rules."""

    CURRENT_TRIGGERS = [
        r"\bcurrent\b", r"\bcurrently\b", r"\bnow\b", r"\btoday\b",
        r"\blatest\b", r"\bpresent\b", r"\bnewest\b", r"\bright now\b"
    ]
    
    HISTORICAL_TRIGGERS = [
        r"\bpreviously\b", r"\bbefore\b", r"\bearlier\b", r"\bpast\b",
        r"\bused to be\b", r"\binitial\b", r"\bfirst\b", r"\boriginally\b",
        r"\bwas\b", r"\bold\b", r"\bformer\b", r"\bformerly\b"
    ]
    
    TIMELINE_TRIGGERS = [
        r"\bhistory\b", r"\btimeline\b", r"\bprogression\b", r"\bevolution\b",
        r"\bhow did .* change\b", r"\bchanges\b", r"\ball versions\b"
    ]
    
    # Common date regex (e.g. 2026-11-10, Nov 10, November 10, 10/11/2026)
    DATE_TRIGGERS = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}\b",
        r"\bon (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    ]

    @classmethod
    def classify_intent(cls, query: str) -> Tuple[QueryTemporalIntent, Optional[datetime]]:
        q_lower = query.lower()
        
        # Check timeline intent
        for pat in cls.TIMELINE_TRIGGERS:
            if re.search(pat, q_lower):
                return QueryTemporalIntent.TIMELINE, None

        # Check explicit date intent
        for pat in cls.DATE_TRIGGERS:
            if re.search(pat, q_lower):
                return QueryTemporalIntent.AT_TIME, None

        # Check historical triggers
        has_historical = any(re.search(pat, q_lower) for pat in cls.HISTORICAL_TRIGGERS)
        has_current = any(re.search(pat, q_lower) for pat in cls.CURRENT_TRIGGERS)

        if has_historical and not has_current:
            return QueryTemporalIntent.HISTORICAL, None
        if has_current and not has_historical:
            return QueryTemporalIntent.CURRENT, None
            
        # Default: if ambiguous, treat as CURRENT if phrased as present tense ("is", "what's")
        if re.search(r"\b(is|what's|where is)\b", q_lower):
            return QueryTemporalIntent.CURRENT, None
            
        return QueryTemporalIntent.GENERAL, None


class HybridTemporalRetriever:
    """Coordinates hybrid retrieval: semantic search + temporal alignment."""

    def __init__(self, sqlite_store: SQLiteTemporalStore, vector_store: ChromaVectorStore):
        self.sqlite = sqlite_store
        self.vector = vector_store
        self.classifier = TemporalIntentClassifier()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        target_timestamp: Optional[datetime] = None,
        forced_intent: Optional[QueryTemporalIntent] = None
    ) -> List[MemoryCandidate]:
        """
        Hybrid retrieval workflow:
        1. Classify query temporal intent.
        2. Query ChromaDB for candidate records by semantic similarity.
        3. Retrieve authoritative records from SQLite.
        4. Compute temporal alignment score based on intent & validity.
        5. Return candidates with semantic and temporal metrics.
        """
        intent, extracted_time = self.classifier.classify_intent(query)
        effective_intent = forced_intent or intent
        ref_time = target_timestamp or extracted_time or datetime.now(timezone.utc)

        # Retrieve a broader pool from vector index (top_k * 2) to filter temporally
        vector_candidates = self.vector.search(query, top_k=max(top_k * 2, 8))
        
        candidates: List[MemoryCandidate] = []
        seen_ids = set()

        for rec_id, sim_score in vector_candidates:
            if rec_id in seen_ids:
                continue
            seen_ids.add(rec_id)
            
            authoritative_record = self.sqlite.get_record(rec_id)
            if not authoritative_record:
                continue  # Stale vector entry, skip
                
            temp_score = self._compute_temporal_score(authoritative_record, effective_intent, ref_time)
            
            candidates.append(MemoryCandidate(
                record=authoritative_record,
                semantic_similarity=sim_score,
                temporal_relevance=temp_score,
                utility_score=0.0,  # Will be assigned by context selector
                estimated_tokens=0
            ))

        # If TIMELINE intent, ensure all history records for matched entities are included
        if effective_intent == QueryTemporalIntent.TIMELINE and candidates:
            # Collect unique (subject, attribute) pairs
            entities = {(c.record.subject, c.record.attribute) for c in candidates[:3]}
            for subj, attr in entities:
                history_records = self.sqlite.get_history(subj, attr)
                for h_rec in history_records:
                    if h_rec.id not in seen_ids:
                        seen_ids.add(h_rec.id)
                        candidates.append(MemoryCandidate(
                            record=h_rec,
                            semantic_similarity=0.8,
                            temporal_relevance=1.0,
                            utility_score=0.0,
                            estimated_tokens=0
                        ))

        return candidates

    def _compute_temporal_score(
        self,
        record: MemoryRecord,
        intent: QueryTemporalIntent,
        ref_time: datetime
    ) -> float:
        """
        Evaluates temporal alignment between record state and query intent:
        - CURRENT: Favors status == CURRENT, heavily penalizes SUPERSEDED.
        - HISTORICAL: Favors status == SUPERSEDED or older valid_from.
        - AT_TIME: Checks if ref_time in [valid_from, valid_until].
        - TIMELINE: All versions are equally relevant chronologically.
        """
        if intent == QueryTemporalIntent.CURRENT:
            if record.status == MemoryStatus.CURRENT:
                return 1.0
            elif record.status == MemoryStatus.DISPUTED:
                return 0.4
            elif record.status == MemoryStatus.SUPERSEDED:
                return 0.05  # Severe penalty against obsolete facts
            else:
                return 0.1

        elif intent == QueryTemporalIntent.HISTORICAL:
            if record.status == MemoryStatus.SUPERSEDED:
                return 1.0
            elif record.status == MemoryStatus.CURRENT:
                return 0.3   # Still somewhat relevant for baseline context
            else:
                return 0.5

        elif intent == QueryTemporalIntent.AT_TIME:
            # Check interval: valid_from <= ref_time < valid_until
            is_after_start = record.valid_from <= ref_time
            is_before_end = (record.valid_until is None) or (record.valid_until > ref_time)
            if is_after_start and is_before_end:
                return 1.0
            return 0.05

        elif intent == QueryTemporalIntent.TIMELINE:
            return 1.0

        # General intent
        if record.status == MemoryStatus.CURRENT:
            return 0.9
        return 0.4
