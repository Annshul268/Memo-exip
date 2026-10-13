"""
Pydantic API request and response models for FastAPI endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from memory.schema import MemoryRecord, MemoryStatus, ContextSelectionResult


class AddMemoryRequest(BaseModel):
    subject: str = Field(..., json_schema_extra={"example": "user"}, description="Subject entity name")
    attribute: str = Field(..., json_schema_extra={"example": "deadline"}, description="Attribute name")
    value: str = Field(..., json_schema_extra={"example": "November 20"}, description="Fact value")
    valid_from: Optional[datetime] = Field(default=None, description="ISO timestamp when fact became valid")
    valid_until: Optional[datetime] = Field(default=None, description="ISO timestamp when fact lapses")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = Field(default="api_client")
    is_explicit_correction: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchMemoryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is my current deadline?"})
    top_k: int = Field(default=5, ge=1, le=50)
    target_timestamp: Optional[datetime] = Field(default=None)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is my current deadline?"})
    budget_tokens: int = Field(default=1024, ge=128, le=8192)
    top_k: int = Field(default=5, ge=1, le=50)
    target_timestamp: Optional[datetime] = Field(default=None)


class AgentQueryResponse(BaseModel):
    query: str
    temporal_intent: str
    response: str
    provider: str
    is_mock: bool
    used_tokens: int
    budget_tokens: int
    budget_respected: bool
    candidate_count: int
    selected_count: int
    selected_memories: List[Dict[str, Any]]
    rejected_memories: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    is_mock_llm: bool
    sqlite_connected: bool
    chroma_connected: bool
    memory_count: int
