
"""
FinSight AI - PostgreSQL Import Preview

Compares cleaned, validated transactions against existing
database records before allowing a statement import.

READ-ONLY: Does not insert, update, or delete records.
"""

from decimal import Decimal

import pandas as pd

from db.connection import get_connection


MATCH_COLUMNS = [
    "transaction_date",
    "transaction_type",
    "amount",
    "description",
    "account_number",
]


def normalize_value(value):
    """Normalize values for comparison."""

    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return value


def transaction_key(row):
    """Create a comparison key for a transaction."""

    date_value = pd.to_datetime(
        row["transaction_date"],
        errors="coerce"
    )

    date_value = (
        date_value.date().isoformat()
        if not pd.isna(date_value)
        else None
    )

    amount = Decimal(str(row["amount"])).quantize(
        Decimal("0.01")
    )

    return (
        date_value,
        str(row["transaction_type"]).strip().title(),
        amount,
        normalize_value(row.get("description")),
        normalize_value(row.get("account_number")),
    )


def preview_import(df):
    """
    Compare incoming transactions with PostgreSQL.

    Returns:
        total
        new_count
        possible_duplicate_count
        new_rows
        possible_duplicates

    No database writes are performed.
    """

    if df is None or df.empty:
        raise ValueError("No transactions to preview.")

    df = df.copy().reset_index(drop=True)

    required = [
        "transaction_date",
        "transaction_type",
        "amount",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for col in MATCH_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Read existing transactions from PostgreSQL.
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    transaction_date,
                    transaction_type,
                    amount,
                    description,
                    account_number
                FROM bank_transactions
            """)

            existing_rows = cursor.fetchall()

    existing_keys = set()

    for row in existing_rows:
        record = dict(zip(MATCH_COLUMNS, row))
        existing_keys.add(transaction_key(record))

    new_indices = []
    duplicate_indices = []

    seen_in_upload = set()

    for index, row in df.iterrows():
        key = transaction_key(row)

        if key in existing_keys or key in seen_in_upload:
            duplicate_indices.append(index)
        else:
            new_indices.append(index)

        seen_in_upload.add(key)

    new_rows = df.iloc[new_indices].copy()
    possible_duplicates = df.iloc[duplicate_indices].copy()

    return {
        "total": len(df),
        "new_count": len(new_rows),
        "possible_duplicate_count": len(possible_duplicates),
        "new_rows": new_rows,
        "possible_duplicates": possible_duplicates,
    }


if __name__ == "__main__":

    from pathlib import Path
    from etl.loader import load_bank_statement
    from etl.mapper import map_columns
    from etl.cleaner import clean_bank_statement
    from etl.validator import validate_transactions

    print("\n===== FINSIGHT AI IMPORT PREVIEW =====")

    file_path = Path("data/test_hdfc.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {file_path}"
        )

    df = load_bank_statement(file_path)
    df = map_columns(df)
    df = clean_bank_statement(df)

    validation = validate_transactions(df)

    if not validation["valid"]:
        raise ValueError(
            f"Validation failed: {validation['errors']}"
        )

    result = preview_import(validation["valid_rows"])

    print("Total transactions:", result["total"])
    print("New transactions:", result["new_count"])
    print(
        "Possible duplicates:",
        result["possible_duplicate_count"]
    )

    print("\nNew transaction preview:")
    print(result["new_rows"].to_string(index=False))

    print("\nImport preview completed successfully!")
