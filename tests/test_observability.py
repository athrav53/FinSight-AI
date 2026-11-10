"""
FinSight AI - LLM observability tests

No Ollama, PostgreSQL or network needed.

Run:
    python -m pytest tests/test_observability.py -v
"""

import json
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
import requests
from fastapi.testclient import TestClient

from rag import observability


def read_log():
    lines = observability.LOG_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def write_events(events):
    with observability.LOG_FILE.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event) + "\n")


def event(feature="rag", status="success", latency=1000, p=100, c=20, age_hours=0):
    ts = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return {"timestamp": ts.isoformat(), "feature": feature, "status": status,
            "latency_ms": latency, "prompt_tokens": p, "completion_tokens": c}


# =====================================================
# RECORDING
# =====================================================

def test_log_keeps_only_allowed_fields():
    observability.log_llm_metrics({
        "feature": "rag", "status": "success", "latency_ms": 12.5,
        "prompt": "secret question", "answer": "secret answer", "api_key": "x",
    })

    logged = read_log()[0]

    assert logged["feature"] == "rag"
    assert "prompt" not in logged and "answer" not in logged and "api_key" not in logged


def test_request_id_is_attached_from_context():
    token = observability.current_request_id.set("req-123")
    try:
        observability.log_llm_metrics({"feature": "agent", "status": "success"})
    finally:
        observability.current_request_id.reset(token)

    assert read_log()[0]["request_id"] == "req-123"


def test_logging_failure_never_raises(monkeypatch):
    monkeypatch.setattr(observability, "LOG_FILE", observability.LOG_FILE.parent)  # a directory
    observability.log_llm_metrics({"status": "success"})   # must not raise


# =====================================================
# SUMMARY
# =====================================================

def test_percentile():
    assert observability.percentile(list(range(1, 101)), 95) == 95
    assert observability.percentile([], 95) is None


def test_summary_failure_rate_latency_and_features():
    write_events([
        event(latency=1000), event(latency=3000),
        event(status="failure", latency=None, p=None, c=None),
        event(feature="agent", latency=2000),
        event(latency=9999, age_hours=48),          # outside the 24 h window
    ])

    summary = observability.summarize_llm_metrics(24)

    assert summary["total_calls"] == 4
    assert summary["failure_rate"] == 0.25
    assert summary["avg_latency_ms"] == 2000
    assert summary["p95_latency_ms"] == 3000
    assert summary["prompt_tokens"] == 300
    assert summary["by_feature"]["rag"]["total_calls"] == 3
    assert summary["by_feature"]["agent"]["failure_rate"] == 0.0


def test_cost_uses_prices_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_1M", "0.5")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_1M", "1.5")
    write_events([event(p=1_000_000, c=1_000_000)])

    assert observability.summarize_llm_metrics(1)["estimated_cost_usd"] == 2.0


def test_empty_or_missing_log_is_safe():
    summary = observability.summarize_llm_metrics(24)
    assert summary["total_calls"] == 0 and summary["avg_latency_ms"] is None


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        observability.read_events(0)


# =====================================================
# AGENT CALLS ARE RECORDED
# =====================================================

def test_agent_success_is_logged():
    from agent import financial_agent

    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": "{}", "prompt_eval_count": 40,
                                  "eval_count": 5, "total_duration": 2_000_000}

    with mock.patch.object(financial_agent.SESSION, "post", return_value=response):
        financial_agent.call_ollama("Q", json_mode=True)

    logged = read_log()[0]
    assert logged["feature"] == "agent"
    assert logged["status"] == "success"
    assert logged["total_tokens"] == 45


def test_agent_failure_is_logged_and_still_raised():
    from agent import financial_agent

    with mock.patch.object(financial_agent.SESSION, "post",
                           side_effect=requests.ConnectionError("down")):
        with pytest.raises(requests.ConnectionError):
            financial_agent.call_ollama("Q")

    logged = read_log()[0]
    assert logged["status"] == "failure"
    assert logged["error_type"] == "ConnectionError"


# =====================================================
# API
# =====================================================

def test_metrics_endpoint_requires_api_key(monkeypatch):
    from api.main import app

    monkeypatch.setenv("FINSIGHT_API_KEY", "test-key")
    client = TestClient(app)

    assert client.get("/metrics").status_code == 401

    write_events([event(), event(feature="agent")])
    response = client.get("/metrics?hours=24", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    assert response.json()["total_calls"] == 2
    assert client.get("/metrics?hours=0", headers={"X-API-Key": "test-key"}).status_code == 422


def test_api_request_id_reaches_llm_log(monkeypatch):
    from api.main import app
    from rag.observability import log_llm_metrics

    monkeypatch.setenv("FINSIGHT_API_KEY", "test-key")

    def fake_agent(question, user_id):
        log_llm_metrics({"feature": "agent", "status": "success", "latency_ms": 5})
        return {"status": "success", "answer": "ok"}

    with mock.patch("api.main.run_agent", side_effect=fake_agent):
        response = TestClient(app).post(
            "/agent", json={"question": "total?"}, headers={"X-API-Key": "test-key"}
        )

    assert response.status_code == 200
    assert read_log()[0]["request_id"] == response.headers["X-Request-ID"]