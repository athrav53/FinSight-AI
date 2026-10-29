"""
transaction_generator.py

Generate realistic transactions from a user's monthly budget.
"""

import random
from datetime import datetime, timedelta

from modules.config_loader import load_config

MERCHANTS = load_config("merchants.json")
PAYMENT_MODES = load_config("payment_modes.json")


def random_transaction_date():
    """Generate a random transaction date within the last 30 days."""

    today = datetime.today()

    days = random.randint(0, 29)

    return (today - timedelta(days=days)).strftime("%Y-%m-%d")


def choose_payment_mode():

    mode_type = random.choice(list(PAYMENT_MODES.keys()))

    return random.choice(PAYMENT_MODES[mode_type])


def generate_transactions(user, account, budget):

    transactions = []

    balance = account["opening_balance"]

    for category, monthly_budget in budget["budget"].items():

        if category == "Savings":
            continue

        number_of_transactions = random.randint(3, 12)

        remaining = monthly_budget

        for i in range(number_of_transactions):

            if i == number_of_transactions - 1:

                amount = round(remaining, 2)

            else:

                amount = round(
                    random.uniform(
                        50,
                        remaining / (number_of_transactions - i)
                    ),
                    2
                )

                remaining -= amount

            merchant = random.choice(
                MERCHANTS.get(
                    category,
                    ["Unknown"]
                )
            )

            payment_mode = choose_payment_mode()

            balance -= amount

            transactions.append({

                "user_id": user["user_id"],

                "account_number":
                    account["account_number"],

                "transaction_date":
                    random_transaction_date(),

                "transaction_type":
                    "Debit",

                "category":
                    category,

                "merchant":
                    merchant,

                "description":
                    f"{payment_mode}/{merchant}",

                "amount":
                    amount,

                "balance":
                    round(balance, 2)

            })

    return transactions