"""
Data models and schemas for temporal memory management.
Defines structured records, temporal validity intervals, version lineage, and query intents.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class MemoryStatus(str, Enum):
    CURRENT = "current"          # Active, currently valid fact
    SUPERSEDED = "superseded"    # Historical version replaced by a newer fact
    DISPUTED = "disputed"        # Conflicting information detected without definitive resolution
    EXPIRED = "expired"          # Fact whose real-world validity period has lapsed


class QueryTemporalIntent(str, Enum):
    CURRENT = "current"          # "What is X now?", "What is my current address?"
    HISTORICAL = "historical"    # "What was X before?", "What did I do previously?"
    AT_TIME = "at_time"          # "What was the deadline on November 10?"
    TIMELINE = "timeline"        # "How did X change over time?", "What is the history of X?"
    GENERAL = "general"          # Open/unspecified temporal intent


class MemoryRecord(BaseModel):
    id: str = Field(..., description="Unique deterministic or UUID identifier for the memory")
    subject: str = Field(..., description="Entity or topic subject (e.g. 'user', 'project_deadline')")
    attribute: str = Field(..., description="Attribute or relation name (e.g. 'office_location', 'submission_date')")
    value: str = Field(..., description="Factual value or textual proposition")
    
    # Transaction / System Time (when the memory was ingested/updated in storage)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the record was initially committed to the store"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this specific record was last touched"
    )
    
    # Real-World Validity Time (when the fact is true in the world)
    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Real-world timestamp when the fact became valid"
    )
    valid_until: Optional[datetime] = Field(
        default=None,
        description="Real-world timestamp when the fact ceases to be valid (None = indefinitely into the future)"
    )
    
    # Reliability and prioritization metrics
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score in the statement's accuracy")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Inherent importance weight (0 to 1)")
    source: str = Field(default="user_statement", description="Provenance or source reference (e.g. 'user_chat', 'system_event')")
    
    # Versioning & Lifecycle
    status: MemoryStatus = Field(default=MemoryStatus.CURRENT, description="Lifecycle status")
    supersedes_id: Optional[str] = Field(default=None, description="ID of earlier memory record this fact supersedes")
    
    # Usage analytics
    access_count: int = Field(default=0, ge=0, description="Number of times retrieved")
    last_accessed_at: Optional[datetime] = Field(default=None, description="Last retrieval timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary auxiliary metadata")

    def to_natural_language(self) -> str:
        """Render fact into natural language for vector embedding and agent context."""
        time_str = f" [Valid from {self.valid_from.isoformat()}"
        if self.valid_until:
            time_str += f" to {self.valid_until.isoformat()}]"
        else:
            time_str += " onward]"
        status_str = f" (Status: {self.status.value})"
        return f"{self.subject} -> {self.attribute}: {self.value}{time_str}{status_str}"


class MemoryCandidate(BaseModel):
    """Candidate memory evaluated by the context-budget selector."""
    record: MemoryRecord
    semantic_similarity: float = Field(default=0.0, description="Cosine similarity from vector retriever")
    temporal_relevance: float = Field(default=0.0, description="Temporal alignment score with query intent")
    utility_score: float = Field(default=0.0, description="Composite utility score")
    estimated_tokens: int = Field(default=0, description="Estimated token footprint")
    rejection_reason: Optional[str] = Field(default=None, description="Reason if candidate was rejected")


class ContextSelectionResult(BaseModel):
    """Result of context-budget-aware memory selection."""
    selected_memories: List[MemoryCandidate]
    rejected_memories: List[MemoryCandidate]
    total_budget_tokens: int
    used_tokens: int
    remaining_tokens: int
    budget_respected: bool
