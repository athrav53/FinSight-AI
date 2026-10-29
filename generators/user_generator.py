"""
user_generator.py

Generates realistic synthetic users for FinSight AI.
"""

import random
from faker import Faker

from modules.config_loader import load_config

fake = Faker("en_IN")

PERSONAS = load_config("personas.json")

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

CITIES = {
    "student": [
        "Pune",
        "Bengaluru",
        "Hyderabad",
        "Mumbai",
        "Delhi"
    ],
    "salaried": [
        "Bengaluru",
        "Hyderabad",
        "Pune",
        "Mumbai",
        "Chennai",
        "Gurugram"
    ],
    "freelancer": [
        "Pune",
        "Indore",
        "Jaipur",
        "Ahmedabad",
        "Kochi"
    ],
    "family": [
        "Pune",
        "Nashik",
        "Surat",
        "Ahmedabad",
        "Indore"
    ]
}


def generate_account_number():

    return "".join(
        random.choices(
            "0123456789",
            k=12
        )
    )


def generate_ifsc(bank):

    return (
        bank["ifsc_prefix"]
        +
        str(random.randint(1000, 9999))
    )


def generate_opening_balance(persona, income):

    if persona == "student":
        return random.randint(1000, 15000)

    if persona == "salaried":
        return random.randint(
            income,
            income * 3
        )

    if persona == "freelancer":
        return random.randint(
            5000,
            income * 2
        )

    return random.randint(
        income,
        income * 4
    )


def generate_user(user_number):

    persona = random.choice(
        list(PERSONAS.keys())
    )

    config = PERSONAS[persona]

    age = random.randint(
        config["age_range"][0],
        config["age_range"][1]
    )

    income = random.randint(
        config["income_range"][0],
        config["income_range"][1]
    )

    bank = random.choice(BANKS)

    city = random.choice(
        CITIES[persona]
    )

    return {

        "user_id":
            f"USR{user_number:05d}",

        "name":
            fake.name(),

        "age":
            age,

        "city":
            city,

        "email":
            fake.email(),

        "phone":
            fake.phone_number(),

        "persona":
            persona,

        "monthly_income":
            income,

        "bank_name":
            bank["name"],

        "account_number":
            generate_account_number(),

        "ifsc":
            generate_ifsc(bank),

        "opening_balance":
            generate_opening_balance(
                persona,
                income
            )

    }


def generate_users(count):

    return [
        generate_user(i)
        for i in range(1, count + 1)
    ]


if __name__ == "__main__":

    users = generate_users(10)

    for user in users:
        print(user)