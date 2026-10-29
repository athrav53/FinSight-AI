
"""
FinSight AI - Financial Insights Engine

Provides:
- Income and expense summary
- Top spending categories
- Large transaction detection
- Rule-based financial observations

Uses pandas and synthetic or validated transaction data.

This module does not access or modify PostgreSQL.
"""

import pandas as pd


def prepare_data(df):
    """Validate and standardize transaction data."""

    required_columns = [
        "transaction_date",
        "transaction_type",
        "amount",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = df.copy()

    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"],
        errors="coerce"
    )

    data["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce"
    )

    data["transaction_type"] = (
        data["transaction_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = data.dropna(
        subset=["transaction_date", "amount"]
    )

    data = data[
        (data["amount"] > 0)
        & data["transaction_type"].isin(
            ["debit", "credit"]
        )
    ]

    if "category" not in data.columns:
        data["category"] = "Uncategorized"

    data["category"] = (
        data["category"]
        .fillna("Uncategorized")
        .astype(str)
        .str.strip()
        .replace("", "Uncategorized")
    )

    return data


def generate_financial_insights(df):
    """
    Generate deterministic financial insights.

    Returns a dictionary suitable for a dashboard,
    report, or later LLM integration.
    """

    data = prepare_data(df)

    if data.empty:
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_cash_flow": 0.0,
            "top_categories": [],
            "large_transactions": [],
            "observations": [
                "No valid transactions are available."
            ],
        }

    expenses = data[
        data["transaction_type"] == "debit"
    ]

    income = data[
        data["transaction_type"] == "credit"
    ]

    total_expenses = float(
        expenses["amount"].sum()
    )

    total_income = float(
        income["amount"].sum()
    )

    net_cash_flow = total_income - total_expenses

    # -------------------------------------
    # TOP SPENDING CATEGORIES
    # -------------------------------------

    category_totals = (
        expenses.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    top_categories = [
        {
            "category": str(category),
            "amount": round(float(amount), 2),
        }
        for category, amount in category_totals.items()
    ]

    # -------------------------------------
    # LARGE TRANSACTION DETECTION
    # -------------------------------------

    large_transactions = []

    if len(expenses) >= 3:

        threshold = (
            expenses["amount"].mean()
            + 2 * expenses["amount"].std(ddof=0)
        )

        flagged = expenses[
            expenses["amount"] > threshold
        ].sort_values(
            "amount",
            ascending=False
        )

        for _, row in flagged.head(10).iterrows():

            large_transactions.append({
                "date": row[
                    "transaction_date"
                ].date().isoformat(),
                "amount": round(
                    float(row["amount"]),
                    2
                ),
                "category": row["category"],
            })

    # -------------------------------------
    # GENERATE OBSERVATIONS
    # -------------------------------------

    observations = []

    observations.append(
        f"Total recorded expenses are "
        f"₹{total_expenses:,.2f}."
    )

    if total_income > 0:

        observations.append(
            f"Total recorded income is "
            f"₹{total_income:,.2f}."
        )

        if net_cash_flow < 0:
            observations.append(
                "Recorded expenses exceed income "
                "for the selected transactions."
            )

        else:
            observations.append(
                "Recorded income covers expenses "
                "for the selected transactions."
            )

    else:

        observations.append(
            "No income transactions were found. "
            "Cash-flow conclusions may be incomplete."
        )

    if top_categories:

        top_category = top_categories[0]

        observations.append(
            f"The highest spending category is "
            f"{top_category['category']} "
            f"(₹{top_category['amount']:,.2f})."
        )

    if large_transactions:

        observations.append(
            f"{len(large_transactions)} unusually "
            f"large expense transaction(s) were "
            f"flagged using a statistical rule. "
            f"These are not necessarily fraudulent."
        )

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_cash_flow": round(net_cash_flow, 2),
        "top_categories": top_categories,
        "large_transactions": large_transactions,
        "observations": observations,
    }


if __name__ == "__main__":

    sample = pd.DataFrame([
        {
            "transaction_date": "2026-10-01",
            "transaction_type": "Debit",
            "category": "Food",
            "amount": 250,
        },
        {
            "transaction_date": "2026-10-02",
            "transaction_type": "Debit",
            "category": "Shopping",
            "amount": 800,
        },
        {
            "transaction_date": "2026-10-03",
            "transaction_type": "Credit",
            "category": "Salary",
            "amount": 45000,
        },
    ])

    result = generate_financial_insights(sample)

    print("\n===== FINSIGHT AI INSIGHTS TEST =====")

    print("Income:", result["total_income"])
    print("Expenses:", result["total_expenses"])
    print("Net Cash Flow:", result["net_cash_flow"])

    print("\nObservations:")

    for observation in result["observations"]:
        print("-", observation)

    assert result["total_income"] == 45000
    assert result["total_expenses"] == 1050
    assert result["net_cash_flow"] == 43950

    print("\nFINANCIAL INSIGHTS TEST PASSED!")
