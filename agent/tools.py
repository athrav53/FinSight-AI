"""
FinSight AI - Database Agent Tools

These tools allow the AI agent to retrieve
financial information from PostgreSQL.

Safety rules:
- All queries are read-only SELECT statements.
- All values are passed as SQL parameters (no string building).
- Only allowlisted tools can run.
- The user whose data is read (user_id) is set by the
  application, never chosen by the language model.
"""

import logging
import re
from decimal import Decimal
from datetime import date, datetime

from psycopg import sql
from psycopg.rows import dict_row

from db.connection import get_connection


logger = logging.getLogger(__name__)

USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,30}$")


# =====================================================
# HELPER - CONVERT DATABASE VALUES
# =====================================================

def serialize_value(value):
    """
    Convert PostgreSQL values into JSON-compatible types.
    """

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return value


def serialize_rows(rows):
    """
    Convert database records into normal Python dictionaries.
    """

    return [
        {
            key: serialize_value(value)
            for key, value in row.items()
        }
        for row in rows
    ]


# =====================================================
# HELPER - VALIDATION AND USER SCOPE
# =====================================================

def validate_limit(limit):
    """Limits must be whole numbers from 1 to 20."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("Limit must be an integer.")

    if not 1 <= limit <= 20:
        raise ValueError("Limit must be between 1 and 20.")


def validate_user_id(user_id):
    """None means all users; otherwise a short safe identifier."""

    if user_id is None:
        return

    if not isinstance(user_id, str) or not USER_ID_PATTERN.match(user_id):
        raise ValueError("Invalid user ID.")


def user_filter(user_id):
    """
    SQL fragment + parameters restricting rows to one user.

    Returns an empty fragment when user_id is None (all users).
    """

    validate_user_id(user_id)

    if user_id is None:
        return sql.SQL(""), []

    return sql.SQL(" AND user_id = %s"), [user_id]


def run_query(query, params):
    """Run one read-only query and return all rows as dictionaries."""

    with get_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()


# =====================================================
# TOOL 1 - TOTAL EXPENSES
# =====================================================

def get_total_expenses(user_id=None):
    """
    Calculate total debit transactions.
    """

    scope, params = user_filter(user_id)

    query = sql.SQL("""
        SELECT
            COUNT(*) AS transaction_count,
            COALESCE(SUM(amount), 0) AS total_expenses
        FROM bank_transactions
        WHERE LOWER(transaction_type) = 'debit'{scope};
    """).format(scope=scope)

    result = run_query(query, params)[0]

    return {
        "tool": "get_total_expenses",
        "transaction_count": result["transaction_count"],
        "total_expenses": float(result["total_expenses"]),
        "currency": "INR"
    }


# =====================================================
# TOOL 2 - TOP EXPENSES
# =====================================================

def get_top_expenses(limit=5, user_id=None):
    """
    Return the largest debit transactions.
    """

    validate_limit(limit)

    scope, params = user_filter(user_id)

    query = sql.SQL("""
        SELECT
            transaction_date,
            merchant,
            category,
            description,
            amount
        FROM bank_transactions
        WHERE LOWER(transaction_type) = 'debit'{scope}
        ORDER BY amount DESC, transaction_date DESC
        LIMIT %s;
    """).format(scope=scope)

    rows = run_query(query, params + [limit])

    return {
        "tool": "get_top_expenses",
        "count": len(rows),
        "expenses": serialize_rows(rows),
        "currency": "INR"
    }


# =====================================================
# TOOL 3 - CATEGORY SPENDING
# =====================================================

def get_category_spending(user_id=None):
    """
    Calculate spending by category.
    """

    scope, params = user_filter(user_id)

    query = sql.SQL("""
        SELECT
            COALESCE(NULLIF(TRIM(category), ''), 'Uncategorized')
                AS category,
            COUNT(*) AS transaction_count,
            SUM(amount) AS total_spent
        FROM bank_transactions
        WHERE LOWER(transaction_type) = 'debit'{scope}
        GROUP BY
            COALESCE(NULLIF(TRIM(category), ''), 'Uncategorized')
        ORDER BY total_spent DESC;
    """).format(scope=scope)

    rows = run_query(query, params)

    return {
        "tool": "get_category_spending",
        "categories": serialize_rows(rows),
        "currency": "INR"
    }


# =====================================================
# TOOL 4 - MONTHLY SPENDING
# =====================================================

def get_monthly_spending(user_id=None):
    """
    Calculate total debit spending for each month.
    """

    scope, params = user_filter(user_id)

    query = sql.SQL("""
        SELECT
            TO_CHAR(
                DATE_TRUNC('month', transaction_date),
                'YYYY-MM'
            ) AS month,
            COUNT(*) AS transaction_count,
            COALESCE(SUM(amount), 0) AS total_spent
        FROM bank_transactions
        WHERE LOWER(transaction_type) = 'debit'{scope}
        GROUP BY DATE_TRUNC('month', transaction_date)
        ORDER BY DATE_TRUNC('month', transaction_date);
    """).format(scope=scope)

    rows = run_query(query, params)

    return {
        "tool": "get_monthly_spending",
        "months": serialize_rows(rows),
        "currency": "INR"
    }


# =====================================================
# TOOL 5 - TOP MERCHANTS
# =====================================================

def get_top_merchants(limit=5, user_id=None):
    """
    Return merchants with the highest total spending.
    """

    validate_limit(limit)

    scope, params = user_filter(user_id)

    query = sql.SQL("""
        SELECT
            COALESCE(
                NULLIF(TRIM(merchant), ''),
                'Unknown Merchant'
            ) AS merchant,
            COUNT(*) AS transaction_count,
            SUM(amount) AS total_spent
        FROM bank_transactions
        WHERE LOWER(transaction_type) = 'debit'{scope}
        GROUP BY
            COALESCE(
                NULLIF(TRIM(merchant), ''),
                'Unknown Merchant'
            )
        ORDER BY total_spent DESC, merchant ASC
        LIMIT %s;
    """).format(scope=scope)

    rows = run_query(query, params + [limit])

    return {
        "tool": "get_top_merchants",
        "count": len(rows),
        "merchants": serialize_rows(rows),
        "currency": "INR"
    }


# =====================================================
# TOOL 6 - SEARCH TRANSACTIONS
# =====================================================

def search_transactions(search_term, limit=10, user_id=None):
    """
    Search transactions by merchant, category,
    or description.
    """

    if not isinstance(search_term, str):
        raise ValueError("Search term must be text.")

    search_term = search_term.strip()

    if not search_term:
        raise ValueError("Search term cannot be empty.")

    if len(search_term) > 100:
        raise ValueError("Search term is too long.")

    validate_limit(limit)

    scope, scope_params = user_filter(user_id)

    # Escape SQL LIKE wildcard characters so that
    # user input is treated as literal search text.
    escaped_term = (
        search_term
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )

    pattern = f"%{escaped_term}%"

    query = sql.SQL("""
        SELECT
            transaction_date,
            transaction_type,
            merchant,
            category,
            description,
            amount
        FROM bank_transactions
        WHERE
            (
                merchant ILIKE %s
                OR category ILIKE %s
                OR description ILIKE %s
            ){scope}
        ORDER BY transaction_date DESC, id DESC
        LIMIT %s;
    """).format(scope=scope)

    rows = run_query(
        query,
        [pattern, pattern, pattern] + scope_params + [limit]
    )

    return {
        "tool": "search_transactions",
        "search_term": search_term,
        "count": len(rows),
        "transactions": serialize_rows(rows),
        "currency": "INR"
    }


# =====================================================
# TOOL REGISTRY
# =====================================================

AVAILABLE_TOOLS = {
    "get_total_expenses": get_total_expenses,
    "get_top_expenses": get_top_expenses,
    "get_category_spending": get_category_spending,
    "get_monthly_spending": get_monthly_spending,
    "get_top_merchants": get_top_merchants,
    "search_transactions": search_transactions,
}


# =====================================================
# TOOL EXECUTION
# =====================================================

def execute_tool(tool_name, arguments=None, user_id=None):
    """
    Execute only approved tools.

    Prevents the agent from executing arbitrary
    Python functions or SQL queries.

    user_id comes from the application (the selected user),
    never from the language model's arguments.
    """

    if tool_name not in AVAILABLE_TOOLS:
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}"
        }

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return {
            "status": "error",
            "message": "Tool arguments must be a dictionary."
        }

    if "user_id" in arguments:
        # The model must never choose whose data to read.
        return {
            "status": "error",
            "message": "The user cannot be chosen in tool arguments."
        }

    try:
        tool_function = AVAILABLE_TOOLS[tool_name]

        result = tool_function(**arguments, user_id=user_id)

        return {
            "status": "success",
            "data": result
        }

    except (ValueError, TypeError) as error:
        # Validation problems are safe to show to the user.
        return {
            "status": "error",
            "message": str(error)
        }

    except Exception:
        # Database errors can contain connection or SQL details:
        # log them, show a generic message.
        logger.exception("Tool %s failed", tool_name)

        return {
            "status": "error",
            "message": (
                "The database query could not be completed. "
                "Check that PostgreSQL is running."
            )
        }


# =====================================================
# DIRECT TEST
# =====================================================

if __name__ == "__main__":

    import json

    print("\n===== FINSIGHT AI AGENT TOOLS =====")

    test_cases = [
        ("get_total_expenses", {}),
        ("get_top_expenses", {"limit": 5}),
        ("get_category_spending", {}),
        ("get_monthly_spending", {}),
        ("get_top_merchants", {"limit": 5}),
        ("search_transactions", {
            "search_term": "Amazon",
            "limit": 10
        }),
    ]

    for tool_name, arguments in test_cases:

        print(f"\n===== {tool_name.upper()} =====")

        result = execute_tool(
            tool_name,
            arguments
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )