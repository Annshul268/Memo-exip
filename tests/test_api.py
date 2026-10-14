"""
Automated tests for FastAPI endpoints using TestClient.
"""

import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
import app.routes as app_routes
from memory.agent_pipeline import TemporalMemoryAgentPipeline
from llm.mock_provider import MockLLMProvider


@pytest.fixture
def client_with_clean_stores():
    temp_dir = Path(tempfile.mkdtemp())
    test_pipeline = TemporalMemoryAgentPipeline(
        sqlite_path=str(temp_dir / "api_test.db"),
        chroma_path=str(temp_dir / "api_chroma"),
        llm_provider=MockLLMProvider()
    )
    
    # Override global pipeline
    app_routes._pipeline = test_pipeline
    
    client = TestClient(fastapi_app)
    yield client
    
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_health_endpoint(client_with_clean_stores):
    client = client_with_clean_stores
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["is_mock_llm"] is True


def test_add_and_query_memory_api(client_with_clean_stores):
    client = client_with_clean_stores
    
    # 1. Ingest initial fact
    payload = {
        "subject": "user",
        "attribute": "supervisor",
        "value": "Dr. Smith",
        "confidence": 1.0,
        "importance": 0.8
    }
    resp1 = client.post("/api/memories", json=payload)
    assert resp1.status_code == 201
    assert resp1.json()["action"] == "created_initial_fact"
    
    # 2. Query current supervisor
    curr_resp = client.get("/api/memories/current?subject=user&attribute=supervisor")
    assert curr_resp.status_code == 200
    assert curr_resp.json()["value"] == "Dr. Smith"
    
    # 3. Update supervisor
    update_payload = {
        "subject": "user",
        "attribute": "supervisor",
        "value": "Dr. Adams",
        "is_explicit_correction": True
    }
    resp2 = client.post("/api/memories", json=update_payload)
    assert resp2.status_code == 201
    assert "superseded" in resp2.json()["action"]
    
    # 4. History check
    hist_resp = client.get("/api/memories/history?subject=user&attribute=supervisor")
    assert hist_resp.status_code == 200
    assert len(hist_resp.json()) == 2
    
    # 5. Agent query endpoint
    agent_resp = client.post("/api/agent/query", json={"query": "Who is my current supervisor now?"})
    assert agent_resp.status_code == 200
    agent_data = agent_resp.json()
    assert "Dr. Adams" in agent_data["response"]
    assert agent_data["budget_respected"] is True

    # 6. Reset endpoint requires confirm=true
    err_reset = client.post("/api/memories/reset?confirm=false")
    assert err_reset.status_code == 400

    ok_reset = client.post("/api/memories/reset?confirm=true")
    assert ok_reset.status_code == 200

    empty_check = client.get("/api/memories")
    assert empty_check.status_code == 200
    assert len(empty_check.json()) == 0
