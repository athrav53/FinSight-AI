
import pytest

from agent.tools import (
    execute_tool,
    validate_user_id,
    user_filter,
    AVAILABLE_TOOLS,
)


def test_missing_user_id_rejected():
    result = execute_tool(
        "get_total_expenses",
        {},
        user_id=None
    )

    assert result["status"] == "error"
    assert result["message"] == "User ID is required."


def test_invalid_user_id_rejected():
    with pytest.raises(ValueError, match="Invalid user ID"):
        validate_user_id("invalid user id!")


def test_valid_user_id_accepted():
    validate_user_id("demo")
    validate_user_id("USR00001")


def test_user_filter_requires_scope():
    with pytest.raises(ValueError, match="User ID is required"):
        user_filter(None)


def test_user_filter_uses_parameter():
    fragment, params = user_filter("demo")

    assert params == ["demo"]
    assert "user_id = %s" in fragment.as_string()


def test_unknown_tool_rejected():
    result = execute_tool(
        "delete_all_transactions",
        {},
        user_id="demo"
    )

    assert result["status"] == "error"


def test_llm_cannot_choose_user():
    result = execute_tool(
        "get_total_expenses",
        {"user_id": "USR00001"},
        user_id="demo"
    )

    assert result["status"] == "error"


def test_only_six_approved_tools():
    assert len(AVAILABLE_TOOLS) == 6
