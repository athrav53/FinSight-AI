
"""
FinSight AI - Financial Agent Unit Tests

Tests tool validation and agent behaviour.
No live PostgreSQL or Ollama connection required.
"""

from unittest.mock import patch

import pytest
import requests

from agent import tools
from agent.financial_agent import (
    validate_tool_call,
    run_agent,
    validate_answer,
    template_answer,
)


# =====================================================
# TEST 1 - VALID TOOL SELECTION
# =====================================================

def test_valid_total_expenses_tool():

    result = validate_tool_call({
        "tool": "get_total_expenses",
        "arguments": {}
    })

    assert result["tool"] == "get_total_expenses"
    assert result["arguments"] == {}


# =====================================================
# TEST 2 - VALID MONTHLY SPENDING TOOL
# =====================================================

def test_valid_monthly_spending_tool():

    result = validate_tool_call({
        "tool": "get_monthly_spending",
        "arguments": {}
    })

    assert result["tool"] == "get_monthly_spending"


# =====================================================
# TEST 3 - VALID TRANSACTION SEARCH
# =====================================================

def test_valid_transaction_search():

    result = validate_tool_call({
        "tool": "search_transactions",
        "arguments": {
            "search_term": "Amazon",
            "limit": 10
        }
    })

    assert result["arguments"]["search_term"] == "Amazon"


# =====================================================
# TEST 4 - REJECT UNKNOWN TOOL
# =====================================================

def test_reject_unknown_tool():

    with pytest.raises(ValueError):

        validate_tool_call({
            "tool": "delete_transactions",
            "arguments": {}
        })


# =====================================================
# TEST 5 - REJECT INVALID LIMIT
# =====================================================

def test_reject_invalid_limit():

    with pytest.raises(ValueError):

        validate_tool_call({
            "tool": "get_top_expenses",
            "arguments": {
                "limit": 100
            }
        })


# =====================================================
# TEST 6 - REJECT INVALID SEARCH
# =====================================================

def test_reject_empty_search():

    with pytest.raises(ValueError):

        validate_tool_call({
            "tool": "search_transactions",
            "arguments": {
                "search_term": ""
            }
        })


# =====================================================
# TEST 7 - REJECT EXTRA ARGUMENTS
# =====================================================

def test_reject_extra_arguments():

    with pytest.raises(ValueError):

        validate_tool_call({
            "tool": "get_total_expenses",
            "arguments": {
                "sql": "DROP TABLE bank_transactions"
            }
        })


# =====================================================
# TEST 8 - SUCCESSFUL AGENT EXECUTION
# =====================================================

@patch("agent.financial_agent.generate_final_answer")
@patch("agent.financial_agent.execute_tool")
@patch("agent.financial_agent.select_tool")
def test_successful_agent_execution(
    mock_select,
    mock_execute,
    mock_answer
):

    mock_select.return_value = {
        "tool": "get_total_expenses",
        "arguments": {}
    }

    mock_execute.return_value = {
        "status": "success",
        "data": {
            "total_expenses": 463956.45,
            "currency": "INR"
        }
    }

    mock_answer.return_value = "₹4,63,956.45"

    result = run_agent(
        "How much did I spend?"
    )

    assert result["status"] == "success"

    assert result["tool"] == "get_total_expenses"

    assert result["answer"] == "₹4,63,956.45"


# =====================================================
# TEST 9 - HANDLE TOOL FAILURE
# =====================================================

@patch("agent.financial_agent.execute_tool")
@patch("agent.financial_agent.select_tool")
def test_tool_failure(
    mock_select,
    mock_execute
):

    mock_select.return_value = {
        "tool": "get_total_expenses",
        "arguments": {}
    }

    mock_execute.return_value = {
        "status": "error",
        "message": "Database unavailable"
    }

    result = run_agent(
        "How much did I spend?"
    )

    assert result["status"] == "tool_error"

    assert "Database unavailable" in result["answer"]


# =====================================================
# TEST 10 - REJECT EMPTY QUESTION
# =====================================================

def test_empty_question():

    result = run_agent("")

    assert result["status"] == "error"


# =====================================================
# ANSWER VALIDATION, USER SCOPE AND FRIENDLY ERRORS
# =====================================================

TOTAL_RESULT = {
    "status": "success",
    "data": {
        "tool": "get_total_expenses",
        "transaction_count": 200,
        "total_expenses": 463956.45,
        "currency": "INR"
    }
}


# TEST 11 - INVENTED AMOUNT IS REPLACED BY THE DATABASE ANSWER

@patch("agent.financial_agent.generate_final_answer")
@patch("agent.financial_agent.execute_tool")
@patch("agent.financial_agent.select_tool")
def test_invented_amount_is_never_shown(mock_select, mock_execute, mock_answer):

    mock_select.return_value = {"tool": "get_total_expenses", "arguments": {}}
    mock_execute.return_value = TOTAL_RESULT
    mock_answer.return_value = "You spent ₹9,99,999 in total."

    result = run_agent("How much did I spend?")

    assert result["status"] == "success"
    assert "9,99,999" not in result["answer"]
    assert "₹4,63,956.45" in result["answer"]
    assert result["answer_source"] == "template_after_validation_failure"
    assert result["invented_amounts"] == ["999999.00"]


# TEST 12 - CORRECT MODEL ANSWER IS KEPT

@patch("agent.financial_agent.generate_final_answer")
@patch("agent.financial_agent.execute_tool")
@patch("agent.financial_agent.select_tool")
def test_correct_model_answer_is_kept(mock_select, mock_execute, mock_answer):

    mock_select.return_value = {"tool": "get_total_expenses", "arguments": {}}
    mock_execute.return_value = TOTAL_RESULT
    mock_answer.return_value = "You spent ₹4,63,956.45 across 200 transactions."

    result = run_agent("How much did I spend?")

    assert result["answer"] == "You spent ₹4,63,956.45 across 200 transactions."
    assert result["answer_source"] == "model"


# TEST 13 - ANSWER STILL WORKS IF THE MODEL FAILS AFTER THE QUERY

@patch("agent.financial_agent.generate_final_answer")
@patch("agent.financial_agent.execute_tool")
@patch("agent.financial_agent.select_tool")
def test_model_down_after_query_uses_database_answer(mock_select, mock_execute, mock_answer):

    mock_select.return_value = {"tool": "get_total_expenses", "arguments": {}}
    mock_execute.return_value = TOTAL_RESULT
    mock_answer.side_effect = requests.ConnectionError("connection refused")

    result = run_agent("How much did I spend?")

    assert result["status"] == "success"
    assert "₹4,63,956.45" in result["answer"]
    assert result["answer_source"] == "template_model_unavailable"


# TEST 14 - FRIENDLY MESSAGE WHEN OLLAMA IS OFF

@patch("agent.financial_agent.call_ollama")
def test_ollama_down_gives_friendly_message(mock_ollama):

    mock_ollama.side_effect = requests.ConnectionError(
        "HTTPConnectionPool(host='127.0.0.1', port=11434): Max retries"
    )

    result = run_agent("How much did I spend?")

    assert result["status"] == "error"
    assert "HTTPConnectionPool" not in result["answer"]
    assert "Ollama" in result["answer"]


# TEST 15 - FRIENDLY MESSAGE WHEN THE MODEL RETURNS BAD JSON

@patch("agent.financial_agent.call_ollama")
def test_bad_json_gives_friendly_message(mock_ollama):

    mock_ollama.return_value = {"response": "not json", "latency_seconds": 0}

    result = run_agent("How much did I spend?")

    assert result["status"] == "error"
    assert "Expecting value" not in result["answer"]
    assert result["error_type"] == "JSONDecodeError"


# TEST 16 - USER SCOPE COMES FROM THE APP, NOT THE MODEL

@patch("agent.financial_agent.generate_final_answer")
@patch("agent.financial_agent.execute_tool")
@patch("agent.financial_agent.select_tool")
def test_user_scope_is_passed_to_tools(mock_select, mock_execute, mock_answer):

    mock_select.return_value = {"tool": "get_total_expenses", "arguments": {}}
    mock_execute.return_value = TOTAL_RESULT
    mock_answer.return_value = "₹4,63,956.45"

    result = run_agent("How much did I spend?", user_id="USR00001")

    assert mock_execute.call_args.kwargs["user_id"] == "USR00001"
    assert result["user_scope"] == "USR00001"


# TEST 17 - THE MODEL CANNOT CHOOSE ANOTHER USER

def test_model_cannot_set_user_id():

    result = tools.execute_tool(
        "get_total_expenses",
        {"user_id": "USR00002"},
        user_id="USR00001"
    )

    assert result["status"] == "error"


# TEST 18 - TOOLS FILTER BY USER IN SQL (PARAMETERISED)

def test_tool_query_is_filtered_by_user():

    captured = {}

    def fake_run_query(query, params):
        captured["params"] = params
        captured["query"] = repr(query)
        return [{"transaction_count": 3, "total_expenses": 100}]

    with patch.object(tools, "run_query", side_effect=fake_run_query):
        tools.get_total_expenses(user_id="USR00001")

    assert captured["params"] == ["USR00001"]
    assert "user_id = %s" in captured["query"]

    with patch.object(tools, "run_query", side_effect=fake_run_query):
        tools.get_total_expenses()

    assert captured["params"] == []
    assert "user_id" not in captured["query"]


# TEST 19 - INVALID USER IDS ARE REJECTED

def test_invalid_user_id_is_rejected():

    result = tools.execute_tool(
        "get_total_expenses",
        {},
        user_id="x' OR '1'='1"
    )

    assert result["status"] == "error"


# TEST 20 - DATABASE ERRORS DO NOT LEAK DETAILS

def test_database_error_message_is_generic():

    with patch.object(tools, "run_query", side_effect=RuntimeError("password authentication failed for user postgres")):
        result = tools.execute_tool("get_total_expenses", {})

    assert result["status"] == "error"
    assert "password" not in result["message"]


# TEST 21 - AMOUNT CHECK RULES

def test_validate_answer_rules():

    data = TOTAL_RESULT["data"]

    assert validate_answer("₹4,63,956.45 across 200 transactions", data)[0]
    assert validate_answer("Total is Rs. 463956", data)[0]          # rounded rupees
    assert validate_answer("On 2026-09-30 you made 200 payments", data)[0]
    assert not validate_answer("You spent ₹9,99,999", data)[0]


# TEST 22 - TEMPLATE ANSWERS CONTAIN ONLY DATABASE AMOUNTS

def test_template_answers_pass_validation():

    samples = {
        "get_total_expenses": TOTAL_RESULT["data"],
        "get_top_merchants": {"merchants": [
            {"merchant": "Amazon", "transaction_count": 4, "total_spent": 15234.5}]},
        "get_monthly_spending": {"months": [
            {"month": "2026-09", "transaction_count": 40, "total_spent": 123456.78}]},
        "search_transactions": {"search_term": "Amazon", "transactions": [
            {"transaction_date": "2026-09-01", "merchant": "Amazon",
             "transaction_type": "debit", "amount": 999.0}]},
    }

    for tool_name, data in samples.items():
        answer = template_answer(tool_name, data)
        assert validate_answer(answer, data)[0], (tool_name, answer)