

from pathlib import Path

from db.connection import get_connection
from db.load_transactions import (
    IMPORT_LOCK_ID,
    TRANSACTION_COLUMNS,
    calculate_file_hash,
    check_transaction_overlap,
    prepare_records,
)
from etl.validator import validate_transactions
from .pdf_adapter import pdf_to_dataframe


def import_pdf(pdf_path, user_id="demo", dry_run=True):
    """Safely validate and import demo PDF transactions."""

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    df = pdf_to_dataframe(pdf_path, user_id=user_id)
    report = validate_transactions(df)

    if not report["valid"]:
        return {
            "status": "validation_failed",
            "inserted": 0,
            "errors": report["errors"],
        }

    valid_rows = report["valid_rows"]
    file_hash = calculate_file_hash(pdf_path)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Serialize imports using the existing loader lock.
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (IMPORT_LOCK_ID,),
            )

            cursor.execute(
                """
                SELECT to_regclass('public.statement_imports')
                """
            )
            tracking_exists = cursor.fetchone()[0] is not None

            if tracking_exists:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM statement_imports
                        WHERE file_hash = %s
                    )
                    """,
                    (file_hash,),
                )
                if cursor.fetchone()[0]:
                    return {
                        "status": "duplicate_file",
                        "inserted": 0,
                    }

            duplicates = check_transaction_overlap(
                cursor, valid_rows
            )
            if duplicates:
                return {
                    "status": "possible_duplicates",
                    "inserted": 0,
                    "duplicate_count": len(duplicates),
                }

            if dry_run:
                return {
                    "status": "dry_run",
                    "inserted": 0,
                    "ready": len(valid_rows),
                }

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS statement_imports (
                    id BIGSERIAL PRIMARY KEY,
                    file_hash CHAR(64) NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    transaction_count INTEGER NOT NULL,
                    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                INSERT INTO statement_imports
                    (file_hash, filename, transaction_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (file_hash) DO NOTHING
                RETURNING id
                """,
                (file_hash, pdf_path.name, len(valid_rows)),
            )

            if cursor.fetchone() is None:
                return {
                    "status": "duplicate_file",
                    "inserted": 0,
                }

            records = prepare_records(valid_rows)

            from psycopg import sql

            query = sql.SQL(
                "INSERT INTO bank_transactions ({}) VALUES ({})"
            ).format(
                sql.SQL(", ").join(
                    map(sql.Identifier, TRANSACTION_COLUMNS)
                ),
                sql.SQL(", ").join(
                    sql.Placeholder()
                    for _ in TRANSACTION_COLUMNS
                ),
            )

            cursor.executemany(query, records)

    return {
        "status": "imported",
        "inserted": len(valid_rows),
    }
