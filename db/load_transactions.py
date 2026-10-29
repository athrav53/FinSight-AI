

"""
FinSight AI - Safe PostgreSQL Transaction Loader

Features:
- Bank statement ETL and validation
- SHA-256 file duplicate protection
- Transaction overlap detection
- Atomic PostgreSQL imports
- Advisory transaction locking
- Dry-run support

IMPORTANT:
- Dry run never writes to PostgreSQL.
- Possible duplicates block the entire import.
- Existing transactions are never updated or deleted.
"""

import hashlib
from pathlib import Path

import pandas as pd
from psycopg import sql

from db.connection import get_connection
from etl.loader import load_bank_statement
from etl.mapper import map_columns
from etl.cleaner import clean_bank_statement
from etl.validator import validate_transactions


TRANSACTION_COLUMNS = [
    "user_id",
    "account_number",
    "transaction_date",
    "transaction_type",
    "category",
    "merchant",
    "description",
    "amount",
    "balance",
]

# Dedicated lock for FinSight AI imports.
IMPORT_LOCK_ID = 20261022


def calculate_file_hash(file_path):
    """Calculate SHA-256 fingerprint of a statement file."""

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_text(value):
    """Normalize text for conservative duplicate comparison."""

    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().lower().split())


def transaction_key(row):
    """
    Build a comparison key.

    This is a possible-duplicate heuristic, not a
    guaranteed unique bank transaction identifier.
    """

    transaction_date = pd.to_datetime(
        row["transaction_date"]
    ).date().isoformat()

    return (
        transaction_date,
        normalize_text(row["transaction_type"]),
        round(float(row["amount"]), 2),
        normalize_text(row.get("description")),
        normalize_text(row.get("account_number")),
    )


def prepare_records(df):
    """Convert validated transactions to PostgreSQL records."""

    df = df.copy()

    for column in TRANSACTION_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[TRANSACTION_COLUMNS].astype(object)
    df = df.where(pd.notna(df), None)

    return list(df.itertuples(index=False, name=None))


def check_transaction_overlap(cursor, incoming_df):
    """
    Find possible duplicates in PostgreSQL and within
    the incoming statement.

    All comparisons happen inside the import transaction.
    """

    if incoming_df.empty:
        return []

    dates = pd.to_datetime(
        incoming_df["transaction_date"]
    ).dt.date

    start_date = dates.min()
    end_date = dates.max()

    cursor.execute(
        """
        SELECT
            transaction_date,
            transaction_type,
            amount,
            description,
            account_number
        FROM bank_transactions
        WHERE transaction_date BETWEEN %s AND %s
        """,
        (start_date, end_date),
    )

    existing_keys = set()

    for (
        transaction_date,
        transaction_type,
        amount,
        description,
        account_number,
    ) in cursor.fetchall():

        existing_keys.add(
            (
                transaction_date.isoformat(),
                normalize_text(transaction_type),
                round(float(amount), 2),
                normalize_text(description),
                normalize_text(account_number),
            )
        )

    incoming_keys = set()
    possible_duplicates = []

    for index, row in incoming_df.iterrows():

        key = transaction_key(row)

        if key in existing_keys:
            possible_duplicates.append(
                {
                    "row_index": str(index),
                    "reason": "Possible match in PostgreSQL",
                    "transaction_date": key[0],
                    "transaction_type": key[1],
                    "amount": key[2],
                    "description": row.get("description"),
                }
            )

        elif key in incoming_keys:
            possible_duplicates.append(
                {
                    "row_index": str(index),
                    "reason": "Repeated in uploaded statement",
                    "transaction_date": key[0],
                    "transaction_type": key[1],
                    "amount": key[2],
                    "description": row.get("description"),
                }
            )

        incoming_keys.add(key)

    return possible_duplicates


def load_transactions(file_path, dry_run=True):
    """
    Validate and optionally import transactions.

    dry_run=True:
        Validate only. No database changes.

    dry_run=False:
        Check for overlapping transactions inside a
        locked PostgreSQL transaction before insertion.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Statement file not found: {file_path}"
        )

    file_hash = calculate_file_hash(file_path)

    # ---------------------------------------------
    # STEP 1: EXTRACT
    # ---------------------------------------------

    df = load_bank_statement(file_path)

    # ---------------------------------------------
    # STEP 2: TRANSFORM
    # ---------------------------------------------

    df = map_columns(df)
    df = clean_bank_statement(df)

    # ---------------------------------------------
    # STEP 3: VALIDATE
    # ---------------------------------------------

    report = validate_transactions(df)

    if not report["valid"]:
        return {
            "status": "validation_failed",
            "inserted": 0,
            "file_hash": file_hash,
            "report": report,
        }

    valid_rows = report["valid_rows"].copy()

    records = prepare_records(valid_rows)

    if dry_run:
        return {
            "status": "dry_run",
            "inserted": 0,
            "ready": len(records),
            "file_hash": file_hash,
            "report": report,
        }

    # ---------------------------------------------
    # STEP 4: PREPARE SQL
    # ---------------------------------------------

    create_tracking_table = """
        CREATE TABLE IF NOT EXISTS statement_imports (
            id BIGSERIAL PRIMARY KEY,
            file_hash CHAR(64) NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            transaction_count INTEGER NOT NULL,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """

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

    # ---------------------------------------------
    # STEP 5: ATOMIC IMPORT
    # ---------------------------------------------

    with get_connection() as connection:
        with connection.cursor() as cursor:

            # Prevent simultaneous imports through
            # this loader from checking/inserting
            # overlapping rows concurrently.
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (IMPORT_LOCK_ID,),
            )

            cursor.execute(create_tracking_table)

            # Check if the exact file was imported.
            cursor.execute(
                """
                SELECT id
                FROM statement_imports
                WHERE file_hash = %s
                """,
                (file_hash,),
            )

            if cursor.fetchone() is not None:
                return {
                    "status": "duplicate_file",
                    "inserted": 0,
                    "file_hash": file_hash,
                    "report": report,
                }

            # Check overlapping transactions while
            # holding the import lock.
            duplicates = check_transaction_overlap(
                cursor,
                valid_rows,
            )

            if duplicates:
                return {
                    "status": "possible_duplicates",
                    "inserted": 0,
                    "duplicate_count": len(duplicates),
                    "possible_duplicates": duplicates,
                    "file_hash": file_hash,
                    "report": report,
                }

            # Claim the file hash.
            cursor.execute(
                """
                INSERT INTO statement_imports (
                    file_hash,
                    filename,
                    transaction_count
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (file_hash) DO NOTHING
                RETURNING id
                """,
                (
                    file_hash,
                    file_path.name,
                    len(records),
                ),
            )

            claim = cursor.fetchone()

            if claim is None:
                return {
                    "status": "duplicate_file",
                    "inserted": 0,
                    "file_hash": file_hash,
                    "report": report,
                }

            cursor.executemany(
                insert_query,
                records,
            )

    return {
        "status": "imported",
        "inserted": len(records),
        "file_hash": file_hash,
        "report": report,
    }


if __name__ == "__main__":

    print("\n===== FINSIGHT AI IMPORT TEST =====")

    test_file = Path("data/test_hdfc.csv")

    result = load_transactions(
        test_file,
        dry_run=True,
    )

    print("Status:", result["status"])
    print("Ready:", result.get("ready", 0))
    print("Inserted:", result["inserted"])

    assert result["status"] == "dry_run"
    assert result["ready"] == 3
    assert result["inserted"] == 0

    print("\nSafe import dry-run test passed!")
