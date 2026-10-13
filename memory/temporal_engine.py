"""
Core Temporal Memory Engine.
Manages deterministic, version-aware fact updates, conflict detection,
historical preservation, and dual-store synchronization.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from memory.schema import MemoryRecord, MemoryStatus
from memory.sqlite_store import SQLiteTemporalStore
from memory.vector_store import ChromaVectorStore


class TemporalMemoryEngine:
    """Coordinates versioned fact lifecycle, contradiction detection, and dual storage."""

    def __init__(self, sqlite_store: SQLiteTemporalStore, vector_store: ChromaVectorStore):
        self.sqlite = sqlite_store
        self.vector = vector_store

    def add_or_update_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        source: str = "user_statement",
        is_explicit_correction: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[MemoryRecord, str]:
        """
        Deterministic, version-aware ingestion logic:
        1. Checks for existing active ('current') version of (subject, attribute).
        2. Case A: No existing record -> Insert new current record.
        3. Case B: Existing record with IDENTICAL value -> No-op / increment access count.
        4. Case C: Existing record with DIFFERENT value & explicit update ->
                   Close old record's validity at valid_from; link supersedes_id; mark new as current.
        5. Case D: Conflicting value with uncertain validity / not explicit correction ->
                   Do NOT blindly overwrite. Flag as 'disputed' or record alternative while preserving current truth.
        Returns: (saved_record, action_description)
        """
        now = datetime.now(timezone.utc)
        v_from = valid_from or now
        meta = metadata or {}
        
        norm_subj = subject.strip().lower()
        norm_attr = attribute.strip().lower()
        norm_val = value.strip()
        
        current_record = self.sqlite.get_current(norm_subj, norm_attr)
        
        # Case A: Brand new fact
        if not current_record:
            rec_id = f"mem_{uuid.uuid4().hex[:12]}"
            new_record = MemoryRecord(
                id=rec_id,
                subject=norm_subj,
                attribute=norm_attr,
                value=norm_val,
                created_at=now,
                updated_at=now,
                valid_from=v_from,
                valid_until=valid_until,
                confidence=confidence,
                importance=importance,
                source=source,
                status=MemoryStatus.CURRENT,
                supersedes_id=None,
                metadata=meta
            )
            self.sqlite.add_record(new_record)
            self.vector.index_memory(new_record)
            return new_record, "created_initial_fact"
            
        # Case B: Identical value (redundant fact reinforcement)
        if current_record.value.lower() == norm_val.lower():
            # Reinforce confidence / update touch timestamp
            current_record.access_count += 1
            current_record.updated_at = now
            current_record.confidence = min(1.0, current_record.confidence + 0.05)
            # Update SQLite and vector index
            with self.sqlite._get_connection() as conn:
                conn.execute("""
                UPDATE memories
                SET access_count = ?, updated_at = ?, confidence = ?
                WHERE id = ?;
                """, (current_record.access_count, current_record.updated_at.isoformat(), current_record.confidence, current_record.id))
            self.vector.index_memory(current_record)
            return current_record, "redundant_reinforced_existing_fact"

        # Case C & D: Conflicting / New value for existing attribute
        # Check temporal order: Is new fact timestamp >= current fact's valid_from?
        is_temporally_newer = v_from >= current_record.valid_from
        
        if is_explicit_correction and is_temporally_newer:
            # Deterministic superseding: Old fact is archived, new fact is current
            rec_id = f"mem_{uuid.uuid4().hex[:12]}"
            new_record = MemoryRecord(
                id=rec_id,
                subject=norm_subj,
                attribute=norm_attr,
                value=norm_val,
                created_at=now,
                updated_at=now,
                valid_from=v_from,
                valid_until=valid_until,
                confidence=confidence,
                importance=importance,
                source=source,
                status=MemoryStatus.CURRENT,
                supersedes_id=current_record.id,
                metadata=meta
            )
            # Supersede transactionally in SQLite
            self.sqlite.supersede_fact(current_record.id, new_record, superseding_time=v_from)
            # Update vector index for both records
            archived_record = self.sqlite.get_record(current_record.id)
            if archived_record:
                self.vector.index_memory(archived_record)
            self.vector.index_memory(new_record)
            return new_record, f"superseded_previous_version_{current_record.id}"
            
        else:
            # Ambiguity or retro-active conflicting statement without clear resolution
            # To protect data integrity, create a new record flagged as DISPUTED without silently overwriting
            rec_id = f"mem_{uuid.uuid4().hex[:12]}"
            disputed_record = MemoryRecord(
                id=rec_id,
                subject=norm_subj,
                attribute=norm_attr,
                value=norm_val,
                created_at=now,
                updated_at=now,
                valid_from=v_from,
                valid_until=valid_until,
                confidence=confidence * 0.7,  # penalize confidence due to conflict
                importance=importance,
                source=source,
                status=MemoryStatus.DISPUTED,
                supersedes_id=None,
                metadata={**meta, "conflict_with": current_record.id}
            )
            self.sqlite.add_record(disputed_record)
            self.vector.index_memory(disputed_record)
            return disputed_record, f"ambiguous_conflict_detected_with_{current_record.id}"

    def get_current_fact(self, subject: str, attribute: str) -> Optional[MemoryRecord]:
        """Fetch the confirmed active current fact."""
        rec = self.sqlite.get_current(subject, attribute)
        if rec:
            self.sqlite.record_access(rec.id)
        return rec

    def get_historical_fact(self, subject: str, attribute: str, timestamp: datetime) -> Optional[MemoryRecord]:
        """Point-in-time lookup for the fact valid at historical timestamp."""
        rec = self.sqlite.get_valid_at(subject, attribute, timestamp)
        if rec:
            self.sqlite.record_access(rec.id)
        return rec

    def get_fact_lineage(self, subject: str, attribute: str) -> List[MemoryRecord]:
        """Retrieve the complete version history chain for an attribute."""
        return self.sqlite.get_history(subject, attribute)

    def reconcile_storage(self) -> Dict[str, int]:
        """Reconcile ChromaDB vector index from authoritative SQLite store."""
        records = self.sqlite.list_records()
        return self.vector.reconcile_with_sqlite(records)
