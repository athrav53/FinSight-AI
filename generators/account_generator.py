"""
account_generator.py

Generates realistic bank accounts for users.
"""

import random

BANKS = [
    {
        "name": "HDFC Bank",
        "ifsc_prefix": "HDFC000"
    },
    {
        "name": "State Bank of India",
        "ifsc_prefix": "SBIN000"
    },
    {
        "name": "ICICI Bank",
        "ifsc_prefix": "ICIC000"
    }
]


def generate_account_number():
    """
    Generate a random 12-digit account number.
    """
    return "".join(
        random.choices("0123456789", k=12)
    )


def generate_ifsc(bank):
    """
    Generate a valid-looking IFSC code.
    """
    return (
        bank["ifsc_prefix"]
        + str(random.randint(1000, 9999))
    )


def generate_opening_balance(persona, income):
    """
    Generate realistic opening balance based on persona.
    """

    if persona == "student":
        return random.randint(1000, 15000)

    elif persona == "salaried":
        return random.randint(
            income,
            income * 3
        )

    elif persona == "freelancer":
        return random.randint(
            5000,
            income * 2
        )

    elif persona == "family":
        return random.randint(
            income,
            income * 4
        )

    return random.randint(1000, 50000)


def generate_account(user):
    """
    Generate one bank account for a user.
    """

    bank = random.choice(BANKS)

    return {

        "user_id": user["user_id"],

        "bank_name": bank["name"],

        "account_number": generate_account_number(),

        "ifsc": generate_ifsc(bank),

        "opening_balance": generate_opening_balance(
            user["persona"],
            user["monthly_income"]
        )

    }


def generate_accounts(users):
    """
    Generate accounts for all users.
    """

    return [
        generate_account(user)
        for user in users
    ]


if __name__ == "__main__":

    from generators.user_generator import generate_users

    users = generate_users(5)

    accounts = generate_accounts(users)

    for account in accounts:
        print(account)