"""
SQLite implementation of the structured temporal source-of-truth store.
Maintains relational integrity, temporal intervals, and version lineage.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from memory.schema import MemoryRecord, MemoryStatus


class SQLiteTemporalStore:
    """Authoritative relational store for temporal memories and version lineage."""

    def __init__(self, db_path: str = "data/temporal_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Create schema and appropriate temporal indexes."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_until TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                importance REAL NOT NULL DEFAULT 0.5,
                source TEXT NOT NULL DEFAULT 'user_statement',
                status TEXT NOT NULL DEFAULT 'current',
                supersedes_id TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (supersedes_id) REFERENCES memories(id) ON DELETE SET NULL
            );
            """)

            # Indexes for high-throughput temporal and entity queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON memories (subject, attribute);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memories (status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_valid_range ON memories (valid_from, valid_until);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_supersedes ON memories (supersedes_id);")

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert a SQLite row into a validated MemoryRecord."""
        def parse_dt(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            try:
                dt = datetime.fromisoformat(val)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                return None

        metadata_dict = {}
        if row["metadata"]:
            try:
                metadata_dict = json.loads(row["metadata"])
            except Exception:
                metadata_dict = {}

        return MemoryRecord(
            id=row["id"],
            subject=row["subject"],
            attribute=row["attribute"],
            value=row["value"],
            created_at=parse_dt(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=parse_dt(row["updated_at"]) or datetime.now(timezone.utc),
            valid_from=parse_dt(row["valid_from"]) or datetime.now(timezone.utc),
            valid_until=parse_dt(row["valid_until"]),
            confidence=row["confidence"],
            importance=row["importance"],
            source=row["source"],
            status=MemoryStatus(row["status"]),
            supersedes_id=row["supersedes_id"],
            access_count=row["access_count"],
            last_accessed_at=parse_dt(row["last_accessed_at"]),
            metadata=metadata_dict
        )

    def add_record(self, record: MemoryRecord) -> None:
        """Atomically insert a new memory record."""
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO memories (
                id, subject, attribute, value,
                created_at, updated_at, valid_from, valid_until,
                confidence, importance, source, status,
                supersedes_id, access_count, last_accessed_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.id,
                record.subject.strip().lower(),
                record.attribute.strip().lower(),
                record.value.strip(),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.valid_from.isoformat(),
                record.valid_until.isoformat() if record.valid_until else None,
                record.confidence,
                record.importance,
                record.source,
                record.status.value,
                record.supersedes_id,
                record.access_count,
                record.last_accessed_at.isoformat() if record.last_accessed_at else None,
                json.dumps(record.metadata)
            ))

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        """Fetch record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM memories WHERE id = ?;", (record_id,))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def get_current(self, subject: str, attribute: str) -> Optional[MemoryRecord]:
        """Get the active 'current' version of a specific fact."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
            SELECT * FROM memories
            WHERE subject = ? AND attribute = ? AND status = 'current'
            ORDER BY valid_from DESC, created_at DESC
            LIMIT 1;
            """, (subject.strip().lower(), attribute.strip().lower()))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def get_history(self, subject: str, attribute: str) -> List[MemoryRecord]:
        """Retrieve the complete chronological version history for an attribute."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
            SELECT * FROM memories
            WHERE subject = ? AND attribute = ?
            ORDER BY valid_from ASC, created_at ASC;
            """, (subject.strip().lower(), attribute.strip().lower()))
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_valid_at(self, subject: str, attribute: str, timestamp: datetime) -> Optional[MemoryRecord]:
        """
        Point-in-time query: find the fact valid at a historical timestamp T.
        Condition: valid_from <= T AND (valid_until IS NULL OR valid_until > T).
        """
        ts_iso = timestamp.isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute("""
            SELECT * FROM memories
            WHERE subject = ? AND attribute = ?
              AND valid_from <= ?
              AND (valid_until IS NULL OR valid_until > ?)
            ORDER BY valid_from DESC, created_at DESC
            LIMIT 1;
            """, (subject.strip().lower(), attribute.strip().lower(), ts_iso, ts_iso))
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def supersede_fact(self, old_record_id: str, new_record: MemoryRecord, superseding_time: datetime) -> None:
        """
        Transactional update:
        1. Closes the validity of the older fact (valid_until = superseding_time, status = superseded).
        2. Links the new fact's supersedes_id to the old record.
        3. Inserts the new fact with status = current.
        Rolled back atomically if any step fails.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Update old record
            cursor.execute("""
            UPDATE memories
            SET status = 'superseded',
                valid_until = ?,
                updated_at = ?
            WHERE id = ?;
            """, (
                superseding_time.isoformat(),
                datetime.now(timezone.utc).isoformat(),
                old_record_id
            ))
            
            # 2. Insert new record
            cursor.execute("""
            INSERT INTO memories (
                id, subject, attribute, value,
                created_at, updated_at, valid_from, valid_until,
                confidence, importance, source, status,
                supersedes_id, access_count, last_accessed_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                new_record.id,
                new_record.subject.strip().lower(),
                new_record.attribute.strip().lower(),
                new_record.value.strip(),
                new_record.created_at.isoformat(),
                new_record.updated_at.isoformat(),
                new_record.valid_from.isoformat(),
                new_record.valid_until.isoformat() if new_record.valid_until else None,
                new_record.confidence,
                new_record.importance,
                new_record.source,
                new_record.status.value,
                old_record_id,
                new_record.access_count,
                None,
                json.dumps(new_record.metadata)
            ))

    def mark_disputed(self, record_ids: List[str], reason: str) -> None:
        """Mark records as disputed when an unresolvable contradiction occurs."""
        with self._get_connection() as conn:
            for rid in record_ids:
                conn.execute("""
                UPDATE memories
                SET status = 'disputed', updated_at = ?
                WHERE id = ?;
                """, (datetime.now(timezone.utc).isoformat(), rid))

    def record_access(self, record_id: str) -> None:
        """Increment access counter and update last_accessed_at timestamp."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE memories
            SET access_count = access_count + 1, last_accessed_at = ?
            WHERE id = ?;
            """, (now_iso, record_id))

    def list_records(self, status: Optional[MemoryStatus] = None, subject: Optional[str] = None) -> List[MemoryRecord]:
        """List records with optional status and subject filtering."""
        query = "SELECT * FROM memories WHERE 1=1"
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if subject:
            query += " AND subject = ?"
            params.append(subject.strip().lower())
        query += " ORDER BY valid_from DESC, created_at DESC;"
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_record(r) for r in cursor.fetchall()]

    def delete_record(self, record_id: str) -> bool:
        """Explicitly remove a record (for testing or hard forget operations)."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?;", (record_id,))
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear the entire database (used in test fixtures)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memories;")
