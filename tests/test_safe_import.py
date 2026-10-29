
"""
FinSight AI - PostgreSQL Safe Import Integration Test

Tests:
1. Synthetic statement processing
2. Transaction validation
3. Duplicate detection before insertion
4. Temporary PostgreSQL insertion
5. Duplicate detection after insertion
6. Transaction rollback
7. Database remains unchanged

IMPORTANT:
- Uses a synthetic test CSV.
- Does not permanently insert transactions.
- Rolls back temporary database changes.
"""

from pathlib import Path

import pytest
from psycopg import sql

from db.connection import get_connection
from db.load_transactions import (
    TRANSACTION_COLUMNS,
    prepare_records,
    check_transaction_overlap,
)
from etl.loader import load_bank_statement
from etl.mapper import map_columns
from etl.cleaner import clean_bank_statement
from etl.validator import validate_transactions


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Synthetic test statement
TEST_FILE = PROJECT_ROOT / "data" / "test_hdfc.csv"


def get_transaction_count(cursor):
    """Return the number of transactions in the database."""

    cursor.execute(
        "SELECT COUNT(*) FROM bank_transactions"
    )

    return cursor.fetchone()[0]


def test_safe_import():
    """
    Test importing synthetic transactions without
    permanently modifying the PostgreSQL database.
    """

    print("\n===== FINSIGHT AI SAFE IMPORT TEST =====")

    # STEP 1: Check test file exists
    assert TEST_FILE.exists(), (
        f"Missing test file: {TEST_FILE}"
    )

    # STEP 2: Load and process synthetic statement
    df = load_bank_statement(TEST_FILE)

    df = map_columns(df)

    df = clean_bank_statement(df)

    report = validate_transactions(df)

    assert report["valid"], (
        f"Statement validation failed: {report['errors']}"
    )

    valid_rows = report["valid_rows"]

    assert len(valid_rows) == 3, (
        f"Expected 3 valid rows, got {len(valid_rows)}"
    )

    records = prepare_records(valid_rows)

    print("Statement validation: PASSED")
    print("Transactions ready:", len(records))

    # STEP 3: Prepare safe SQL insert query
    insert_query = sql.SQL("""
        INSERT INTO bank_transactions ({})
        VALUES ({})
    """).format(
        sql.SQL(", ").join(
            sql.Identifier(column)
            for column in TRANSACTION_COLUMNS
        ),
        sql.SQL(", ").join(
            sql.Placeholder()
            for _ in TRANSACTION_COLUMNS
        ),
    )

    # STEP 4: Open PostgreSQL connection
    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            # Coordinate with the import lock
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (20261022,),
            )

            original_count = get_transaction_count(cursor)

            print(
                "Original database transaction count:",
                original_count,
            )

            # STEP 5: Check duplicates before insertion
            duplicates_before = check_transaction_overlap(
                cursor,
                valid_rows,
            )

            print(
                "Possible duplicates before insertion:",
                len(duplicates_before),
            )

            if duplicates_before:
                pytest.skip(
                    "Synthetic sample transactions already "
                    "exist in the database. Test skipped."
                )

            # STEP 6: Insert transactions temporarily
            cursor.executemany(
                insert_query,
                records,
            )

            temporary_count = get_transaction_count(cursor)

            assert temporary_count == original_count + 3, (
                "Temporary insertion count mismatch."
            )

            print(
                "Temporary insertion: PASSED",
                f"({temporary_count} rows visible)",
            )

            # STEP 7: Check duplicates after insertion
            duplicates_after = check_transaction_overlap(
                cursor,
                valid_rows,
            )

            assert len(duplicates_after) == 3, (
                "Duplicate detection failed after insertion."
            )

            print(
                "Duplicate detection after insertion: PASSED"
            )

            # STEP 8: Roll back temporary insertion
            connection.rollback()

            with connection.cursor() as verify_cursor:
                final_count = get_transaction_count(
                    verify_cursor
                )

            assert final_count == original_count, (
                "Database row count changed after rollback."
            )

            print("Rollback verification: PASSED")

            print(
                "Final database transaction count:",
                final_count,
            )

    finally:
        # Ensure no uncommitted changes remain
        connection.rollback()
        connection.close()

    print("\nALL SAFE IMPORT TESTS PASSED!")


if __name__ == "__main__":
    test_safe_import()