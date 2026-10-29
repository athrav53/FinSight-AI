
"""
FinSight AI - Full Loader Integration Test

Uses finsight_ai_test ONLY.

Tests:
1. Initial transaction count
2. Actual import of synthetic HDFC statement
3. Duplicate-file protection
4. Duplicate-transaction protection
5. Final transaction count

Never connects the loader to the production database.
"""

import os
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import load_dotenv

import db.load_transactions as loader


TEST_DATABASE = "finsight_ai_test"
TEST_FILE = Path("data/test_hdfc.csv")


def test_connection():
    """Create a connection to the isolated test database."""

    env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path)

    password = (
        os.getenv("DB_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
        or os.getenv("PGPASSWORD")
    )

    if not password:
        raise RuntimeError("Database password not found.")

    # Read host, port, and user from the existing
    # working connection.
    from db.connection import get_connection

    with get_connection() as main_conn:
        params = dict(main_conn.info.get_parameters())

    params["dbname"] = TEST_DATABASE
    params["password"] = password

    connection = psycopg.connect(**params)

    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database()")
        actual_database = cursor.fetchone()[0]

    if actual_database != TEST_DATABASE:
        connection.close()
        raise RuntimeError(
            f"Unsafe database connection: {actual_database}"
        )

    return connection


def get_count():
    with test_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM bank_transactions"
            )
            return cursor.fetchone()[0]


def run_tests():

    print("\n===== FINSIGHT AI LOADER INTEGRATION TEST =====")

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    # Redirect all loader database connections
    # to finsight_ai_test.
    with patch.object(
        loader,
        "get_connection",
        side_effect=test_connection,
    ):

        initial_count = get_count()
        print("Initial test database rows:", initial_count)

        # TEST 1: Actual import
        first = loader.load_transactions(
            TEST_FILE,
            dry_run=False,
        )

        print("First import status:", first["status"])
        print("Rows inserted:", first["inserted"])

        assert first["status"] == "imported"
        assert first["inserted"] == 3

        after_first = get_count()

        assert after_first == initial_count + 3

        # TEST 2: Reimport the same file
        second = loader.load_transactions(
            TEST_FILE,
            dry_run=False,
        )

        print("Second import status:", second["status"])

        assert second["status"] == "duplicate_file"
        assert second["inserted"] == 0

        assert get_count() == after_first

        # TEST 3: Duplicate transaction check
        from etl.loader import load_bank_statement
        from etl.mapper import map_columns
        from etl.cleaner import clean_bank_statement

        df = load_bank_statement(TEST_FILE)
        df = map_columns(df)
        df = clean_bank_statement(df)

        with test_connection() as connection:
            with connection.cursor() as cursor:
                duplicates = loader.check_transaction_overlap(
                    cursor,
                    df,
                )

        assert len(duplicates) == 3

        print(
            "Overlapping transactions detected:",
            len(duplicates),
        )

        print(
            "Final test database rows:",
            get_count(),
        )

    print("\nALL LOADER INTEGRATION TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
