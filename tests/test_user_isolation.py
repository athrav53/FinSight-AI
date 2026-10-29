import pytest
from agent import tools


@pytest.mark.parametrize("tool_name,arguments", [
    ("get_total_expenses", {}),
    ("get_top_expenses", {"limit": 3}),
    ("get_category_spending", {}),
    ("get_monthly_spending", {}),
    ("get_top_merchants", {"limit": 3}),
    ("search_transactions", {"search_term": "Amazon", "limit": 3}),
])
def test_user_isolation(monkeypatch, tool_name, arguments):
    captured = []

    def fake_run_query(query, params):
        captured.append((query, params))

        if tool_name == "get_total_expenses":
            return [{
                "transaction_count": 0,
                "total_expenses": 0
            }]

        return []

    monkeypatch.setattr(tools, "run_query", fake_run_query)

    result = tools.execute_tool(
        tool_name,
        arguments,
        user_id="test_user_01"
    )

    assert result["status"] == "success"
    assert len(captured) == 1

    query, params = captured[0]

    assert "test_user_01" in params
    assert "user_id" in str(query)


def test_ai_cannot_change_user():
    result = tools.execute_tool(
        "get_total_expenses",
        {"user_id": "other_user"},
        user_id="test_user_01"
    )

    assert result["status"] == "error"


@pytest.mark.parametrize("bad_id", [
    "",
    "user 1",
    "x' OR 1=1 --",
    "a" * 31
])
def test_invalid_user_id(bad_id):
    with pytest.raises(ValueError):
        tools.validate_user_id(bad_id)