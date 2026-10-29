"""
budget_generator.py

Creates a monthly spending budget for every user.
"""

import random

# Budget allocation percentages
BUDGET_RULES = {
    "student": {
        "Food": 0.30,
        "Transport": 0.10,
        "Shopping": 0.15,
        "Entertainment": 0.15,
        "Recharge": 0.05,
        "Savings": 0.25
    },

    "salaried": {
        "Food": 0.12,
        "Rent": 0.25,
        "Shopping": 0.10,
        "Utilities": 0.08,
        "Investment": 0.15,
        "Travel": 0.05,
        "Savings": 0.25
    },

    "freelancer": {
        "Food": 0.15,
        "Software": 0.15,
        "Travel": 0.10,
        "Shopping": 0.10,
        "Office Expense": 0.10,
        "Savings": 0.40
    },

    "family": {
        "Groceries": 0.20,
        "Medical": 0.08,
        "Education": 0.12,
        "Shopping": 0.10,
        "Utilities": 0.10,
        "EMI": 0.20,
        "Savings": 0.20
    }
}


def generate_budget(user):

    income = user["monthly_income"]

    persona = user["persona"]

    rules = BUDGET_RULES[persona]

    budget = {}

    for category, percentage in rules.items():

        amount = round(
            income * percentage,
            2
        )

        budget[category] = amount

    return {

        "user_id": user["user_id"],

        "persona": persona,

        "monthly_income": income,

        "budget": budget

    }


def generate_budgets(users):

    return [

        generate_budget(user)

        for user in users

    ]


if __name__ == "__main__":

    from generators.user_generator import generate_users

    users = generate_users(3)

    budgets = generate_budgets(users)

    for budget in budgets:

        print(budget)