

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# Test 1: Verify health endpoint
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


# Test 2: Reject requests without an API key
def test_agent_without_api_key():
    response = client.post(
        "/agent",
        json={"question": "How much did I spend?"}
    )

    assert response.status_code in (401, 503)


# Test 3: Reject an invalid API key
def test_agent_with_invalid_api_key(monkeypatch):
    monkeypatch.setenv(
        "FINSIGHT_API_KEY",
        "test-secret-key"
    )
    monkeypatch.setenv(
        "FINSIGHT_DEMO_USER_ID",
        "demo"
    )

    response = client.post(
        "/agent",
        headers={"X-API-Key": "wrong-key"},
        json={"question": "How much did I spend?"}
    )

    assert response.status_code == 401


# Test 4: Reject a missing question after authentication
def test_agent_missing_question(monkeypatch):
    monkeypatch.setenv(
        "FINSIGHT_API_KEY",
        "test-secret-key"
    )
    monkeypatch.setenv(
        "FINSIGHT_DEMO_USER_ID",
        "demo"
    )

    response = client.post(
        "/agent",
        headers={"X-API-Key": "test-secret-key"},
        json={}
    )

    assert response.status_code == 422
