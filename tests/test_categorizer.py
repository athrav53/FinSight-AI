"""
FinSight AI - Transaction categorisation tests

No Ollama, PostgreSQL or network needed (the LLM is mocked).

Run:
    python -m pytest tests/test_categorizer.py -v
"""

import json
from unittest import mock

import pytest
import requests
from fastapi.testclient import TestClient

from ai import categorizer as cat
from evaluation import categorization_evaluator as ev


def llm_reply(category, confidence=0.9, reason="test"):
    return json.dumps({"category": category, "confidence": confidence, "reason": reason})


# =====================================================
# RULES
# =====================================================

@pytest.mark.parametrize("description, tx_type, expected", [
    ("UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl", "debit", "Food"),
    ("AMAZON PRIME VIDEO", "debit", "Entertainment"),        # longest match beats "amazon"
    ("AMAZON REFUND ORDER 1", "credit", "Refund"),
    ("NEFT DR HOUSE RENT NOV", "debit", "Rent"),
    ("NEFT CR TENANT FLAT 302 RENT", "credit", "Rental Income"),  # type decides
    ("jio prepaid recharge", "DR", "Recharge"),               # case and type spelling
])
def test_rules(description, tx_type, expected):
    result = cat.rule_categorize(description, tx_type)
    assert result.category == expected
    assert result.source == "rules"


def test_rules_use_whole_words():
    # "tax" must not match "taxi"
    assert cat.rule_categorize("BLUE TAXI SERVICES", "debit") is None


def test_no_rule_returns_none():
    assert cat.rule_categorize("UPI/DR/ HALDIRAM SNACKS", "debit") is None


def test_empty_description_rejected():
    with pytest.raises(ValueError):
        cat.categorize("   ", "debit")


# =====================================================
# STRUCTURED OUTPUT (PYDANTIC)
# =====================================================

def test_prediction_rejects_unknown_category():
    with pytest.raises(Exception):
        cat.CategoryPrediction.model_validate_json(llm_reply("Crypto"))


def test_schema_lists_only_allowed_categories():
    schema = cat.CategoryPrediction.model_json_schema()
    assert set(schema["properties"]["category"]["enum"]) == set(cat.ALL_CATEGORIES)


# =====================================================
# LLM PATH (MOCKED)
# =====================================================

def test_llm_valid_answer():
    with mock.patch.object(cat, "call_llm", return_value=llm_reply("Food", 0.8)):
        result = cat.llm_categorize("UPI/DR/ HALDIRAM SNACKS", "debit")

    assert result.category == "Food"
    assert result.source == "llm"
    assert result.confidence == 0.8


def test_llm_invalid_json_falls_back():
    with mock.patch.object(cat, "call_llm", return_value="not json"):
        result = cat.llm_categorize("UPI/DR/ HALDIRAM SNACKS", "debit")

    assert (result.category, result.source, result.error) == ("Others", "fallback", "invalid_output")


def test_llm_income_category_for_debit_is_rejected():
    with mock.patch.object(cat, "call_llm", return_value=llm_reply("Salary")):
        result = cat.llm_categorize("UPI/DR/ SOMETHING", "debit")

    assert result.error == "category_type_mismatch"
    assert result.category == "Others"


def test_llm_down_falls_back():
    with mock.patch.object(cat, "call_llm", side_effect=requests.ConnectionError("down")):
        result = cat.llm_categorize("UPI/DR/ HALDIRAM SNACKS", "debit")

    assert result.error == "model_unavailable"


def test_hybrid_skips_llm_when_rule_matches():
    with mock.patch.object(cat, "call_llm") as llm:
        result = cat.categorize("SWIGGY ORDER", "debit", mode="hybrid")

    llm.assert_not_called()
    assert result.source == "rules"


def test_hybrid_uses_llm_when_no_rule_matches():
    with mock.patch.object(cat, "call_llm", return_value=llm_reply("Food")) as llm:
        result = cat.categorize("UPI/DR/ HALDIRAM SNACKS", "debit", mode="hybrid")

    llm.assert_called_once()
    assert result.source == "llm"


def test_llm_call_is_logged_for_observability():
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": llm_reply("Food"),
                                  "prompt_eval_count": 50, "eval_count": 10}

    with mock.patch.object(cat.SESSION, "post", return_value=response), \
         mock.patch.object(cat, "log_llm_metrics") as log:
        cat.call_llm("prompt")

    assert log.call_args.args[0]["feature"] == "categorizer"
    assert log.call_args.args[0]["total_tokens"] == 60


def test_description_is_truncated_in_prompt():
    long_text = "A" * 1000
    with mock.patch.object(cat, "call_llm", return_value=llm_reply("Others")) as llm:
        cat.llm_categorize(long_text, "debit")

    assert "A" * 201 not in llm.call_args.args[0]


# =====================================================
# EVALUATION
# =====================================================

def test_dataset_labels_are_valid():
    cases = ev.load_cases()
    assert len(cases) >= 50
    assert all(c["expected_category"] in cat.ALL_CATEGORIES for c in cases)
    assert all(c["transaction_type"] in {"debit", "credit"} for c in cases)


def test_evaluator_metrics():
    cases = [
        {"id": "a", "description": "SWIGGY", "transaction_type": "debit", "expected_category": "Food"},
        {"id": "b", "description": "UNKNOWN SHOP", "transaction_type": "debit", "expected_category": "Shopping"},
    ]

    report = ev.evaluate_mode(cases, "rules")

    assert report["accuracy"] == 0.5
    assert report["rule_coverage"] == 0.5
    assert report["fallbacks"] == 1
    assert report["errors"][0]["id"] == "b"


def test_rules_never_wrong_when_they_match():
    """On the labelled set, every rule match must be correct."""

    report = ev.evaluate_mode(ev.load_cases(), "rules")
    wrong_rule_matches = [e for e in report["errors"] if e["source"] == "rules"]

    assert wrong_rule_matches == []


# =====================================================
# API
# =====================================================

def test_categorize_endpoint(monkeypatch):
    from api.main import app

    monkeypatch.setenv("FINSIGHT_API_KEY", "test-key")
    client = TestClient(app)

    body = {"description": "SWIGGY ORDER 12", "transaction_type": "debit", "mode": "rules"}

    assert client.post("/categorize", json=body).status_code == 401

    response = client.post("/categorize", json=body, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["category"] == "Food"

    bad = {**body, "mode": "magic"}
    assert client.post("/categorize", json=bad, headers={"X-API-Key": "test-key"}).status_code == 422