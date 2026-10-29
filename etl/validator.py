
"""
FinSight AI - ETL Transaction Validator

Validates cleaned bank statements before database insertion.

Features:
- Required column checks
- Row-level validation
- Duplicate warnings
- Negative balance warnings
- Merchant quality checks
- Valid and rejected row reports

Does NOT insert data into PostgreSQL.
"""

from pathlib import Path
import pandas as pd

from etl.loader import load_bank_statement
from etl.mapper import map_columns
from etl.cleaner import clean_bank_statement


REQUIRED_COLUMNS = [
    "transaction_date",
    "description",
    "amount",
    "transaction_type"
]

VALID_TRANSACTION_TYPES = ["Debit", "Credit"]


def validate_transactions(df):
    """
    Validate a bank statement DataFrame.

    Keeps backward-compatible keys:
        valid
        errors
        warnings

    Adds:
        total_rows
        valid_count
        rejected_count
        valid_rows
        rejected_rows
        row_report
    """

    errors = []
    warnings = []

    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "valid": False,
            "errors": ["Input must be a pandas DataFrame."],
            "warnings": [],
            "total_rows": 0,
            "valid_count": 0,
            "rejected_count": 0,
            "valid_rows": pd.DataFrame(),
            "rejected_rows": pd.DataFrame(),
            "row_report": pd.DataFrame()
        }

    df = df.copy().reset_index(drop=True)

    if df.empty:
        errors.append("Bank statement contains no transactions.")

    # 1. Check required columns.
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns: {missing_columns}"
        )

        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "total_rows": len(df),
            "valid_count": 0,
            "rejected_count": len(df),
            "valid_rows": df.iloc[0:0].copy(),
            "rejected_rows": df.copy(),
            "row_report": pd.DataFrame()
        }

    # 2. Prepare row-level report.
    row_issues = [[] for _ in range(len(df))]
    row_warnings = [[] for _ in range(len(df))]

    def add_error(mask, message):
        for index in df.index[mask]:
            row_issues[index].append(message)

    def add_warning(mask, message):
        for index in df.index[mask]:
            row_warnings[index].append(message)

    # 3. Validate dates.
    dates = pd.to_datetime(
        df["transaction_date"],
        errors="coerce",
        dayfirst=False
    )

    invalid_dates = dates.isna()

    if invalid_dates.any():
        errors.append(
            f"{invalid_dates.sum()} invalid transaction dates"
        )
        add_error(invalid_dates, "Invalid transaction date")

    # 4. Validate amounts.
    amounts = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    invalid_amounts = amounts.isna() | (amounts <= 0)

    if invalid_amounts.any():
        errors.append(
            f"{invalid_amounts.sum()} invalid transaction amounts"
        )
        add_error(invalid_amounts, "Amount must be greater than zero")

    # 5. Validate transaction types.
    types = (
        df["transaction_type"]
        .astype("string")
        .str.strip()
    )

    invalid_types = ~types.isin(VALID_TRANSACTION_TYPES)

    if invalid_types.any():
        errors.append(
            f"{invalid_types.sum()} invalid transaction types"
        )
        add_error(invalid_types, "Invalid transaction type")

    # 6. Validate descriptions.
    descriptions = (
        df["description"]
        .astype("string")
        .str.strip()
    )

    missing_descriptions = (
        descriptions.isna() | descriptions.eq("")
    ).fillna(True)

    if missing_descriptions.any():
        errors.append(
            f"{missing_descriptions.sum()} missing descriptions"
        )
        add_error(missing_descriptions, "Missing description")

    # 7. Check possible duplicates.
    duplicate_columns = [
        col for col in [
            "user_id",
            "account_number",
            "transaction_date",
            "description",
            "amount",
            "transaction_type",
            "balance"
        ]
        if col in df.columns
    ]

    if duplicate_columns:
        duplicates = df.duplicated(
            subset=duplicate_columns,
            keep=False
        )

        if duplicates.any():
            warnings.append(
                f"{duplicates.sum()} rows belong to possible "
                "duplicate transaction groups"
            )
            add_warning(duplicates, "Possible duplicate transaction")

    # 8. Check negative balances.
    if "balance" in df.columns:
        balances = pd.to_numeric(
            df["balance"],
            errors="coerce"
        )

        negative_balances = balances < 0

        if negative_balances.any():
            warnings.append(
                f"{negative_balances.sum()} negative balances found"
            )
            add_warning(negative_balances, "Negative balance")

    # 9. Check unknown merchants.
    if "merchant" in df.columns:
        merchants = (
            df["merchant"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        unknown_merchants = (
            merchants.isna()
            | merchants.isin(["", "unknown"])
        ).fillna(True)

        if unknown_merchants.any():
            warnings.append(
                f"{unknown_merchants.sum()} transactions "
                "need merchant identification"
            )
            add_warning(
                unknown_merchants,
                "Merchant identification needed"
            )

    # 10. Build row-level validation report.
    row_report = pd.DataFrame({
        "source_row": df.index + 2,
        "status": [
            "Rejected" if issues else "Valid"
            for issues in row_issues
        ],
        "errors": [
            "; ".join(issues)
            for issues in row_issues
        ],
        "warnings": [
            "; ".join(items)
            for items in row_warnings
        ]
    })

    valid_mask = row_report["status"].eq("Valid")

    valid_rows = df.loc[valid_mask].copy()
    rejected_rows = df.loc[~valid_mask].copy()

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_rows": len(df),
        "valid_count": len(valid_rows),
        "rejected_count": len(rejected_rows),
        "valid_rows": valid_rows,
        "rejected_rows": rejected_rows,
        "row_report": row_report
    }


if __name__ == "__main__":

    print("\n===== FINSIGHT AI VALIDATION TEST =====")

    test_files = [
        Path("data/test_hdfc.csv"),
        Path("data/test_sbi.csv"),
        Path("data/test_icici.csv")
    ]

    tested = 0

    for file_path in test_files:

        if not file_path.exists():
            print(f"\nMissing test file: {file_path}")
            continue

        print(f"\nTesting: {file_path.name}")

        df = load_bank_statement(file_path)
        df = map_columns(df)
        df = clean_bank_statement(df)

        result = validate_transactions(df)

        print("Total rows:", result["total_rows"])
        print("Valid rows:", result["valid_count"])
        print("Rejected rows:", result["rejected_count"])
        print("Validation passed:", result["valid"])

        print("\nRow report:")
        print(result["row_report"].to_string(index=False))

        if result["errors"]:
            print("\nErrors:", result["errors"])

        if result["warnings"]:
            print("\nWarnings:", result["warnings"])

        assert result["valid"] is True
        assert result["valid_count"] == 3
        assert result["rejected_count"] == 0

        print(f"PASSED: {file_path.name}")
        tested += 1

    assert tested == 3, (
        "All three sample bank statement files are required."
    )

    # Test rejection of an invalid amount.
    invalid_df = pd.DataFrame({
        "transaction_date": ["2026-10-01", "2026-10-02"],
        "description": ["UPI Payment", "Salary"],
        "transaction_type": ["Debit", "Credit"],
        "amount": [450, -100]
    })

    invalid_result = validate_transactions(invalid_df)

    assert invalid_result["valid"] is False
    assert invalid_result["valid_count"] == 1
    assert invalid_result["rejected_count"] == 1

    print("\nPASSED: Invalid amount rejection test")
    print("\nAll validator tests passed!")
