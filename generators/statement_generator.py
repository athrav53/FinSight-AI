"""
statement_generator.py

Generates complete synthetic bank statements.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from generators.user_generator import generate_users
from generators.account_generator import generate_accounts
from generators.budget_generator import generate_budgets
from generators.transaction_generator import generate_transactions


def generate_statements(number_of_users=10):
    """
    Generate complete bank statements.
    """

    users = generate_users(number_of_users)

    accounts = generate_accounts(users)

    budgets = generate_budgets(users)

    all_transactions = []

    for user, account, budget in zip(users, accounts, budgets):

        transactions = generate_transactions(
            user,
            account,
            budget
        )

        all_transactions.extend(transactions)

    df = pd.DataFrame(all_transactions)

    return df


if __name__ == "__main__":

    # Generate statements
    df = generate_statements(5)

    print("\nFirst 10 Transactions:\n")
    print(df.head(10))

    print(f"\nTotal Transactions: {len(df)}")

    # Create data folder if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Generate unique filename
    filename = (
        "data/sample_bank_statement_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )

    # Save CSV
    df.to_csv(
        filename,
        index=False
    )

    print("\n✅ CSV saved successfully!")
    print(f"📁 File Location: {filename}")