
"""
FinSight AI - Universal Bank Statement Cleaner

Cleans mapped bank statements from HDFC, SBI, ICICI,
and FinSight Generic.

No database writes are performed here.
"""

import re
from datetime import datetime, date

import pandas as pd


def clean_amount(value):
    """Convert bank amounts into numeric values."""

    if pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if text.lower() in {
        "", "nan", "none", "null", "-", "na", "n/a"
    }:
        return float("nan")

    negative = text.startswith("(") and text.endswith(")")

    if negative:
        text = text[1:-1]

    text = re.sub(
        r"(?i)\b(?:inr|rs)\b\.?",
        "",
        text
    )

    text = text.replace("₹", "").replace(",", "").strip()

    try:
        amount = float(text)
        return -amount if negative else amount
    except (ValueError, TypeError):
        return float("nan")


def parse_transaction_date(value):
    """
    Parse common Indian bank statement date formats.

    Ambiguous dates are interpreted as DD/MM/YYYY.
    """

    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value)

    text = str(value).strip()

    if text.lower() in {"", "nan", "none", "null", "nat"}:
        return pd.NaT

    supported_formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    for fmt in supported_formats:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue

    return pd.NaT


def clean_bank_statement(df):
    """Clean and validate essential transaction fields."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Bank statement is empty or invalid.")

    df = df.copy()

    # 1. Normalize columns
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    # 2. Clean text
    for col in [
        "user_id",
        "account_number",
        "description",
        "merchant",
        "category"
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(
                {"": pd.NA, "nan": pd.NA, "None": pd.NA}
            )

    # 3. Clean dates
    if "transaction_date" not in df.columns:
        raise ValueError("Missing transaction_date column.")

    original_dates = df["transaction_date"].copy()

    df["transaction_date"] = original_dates.apply(
        parse_transaction_date
    )

    invalid_dates = df["transaction_date"].isna()

    if invalid_dates.any():
        examples = original_dates.loc[
            invalid_dates
        ].head(10).tolist()

        print("\nINVALID DATE EXAMPLES:")
        for value in examples:
            print(repr(value))

    # 4. Clean numeric columns
    for col in ["amount", "debit", "credit", "balance"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_amount)

    # 5. Convert legacy debit/credit columns
    if "amount" not in df.columns:
        if not {"debit", "credit"}.issubset(df.columns):
            raise ValueError(
                "Missing amount or debit/credit columns."
            )

        debit = df["debit"].fillna(0)
        credit = df["credit"].fillna(0)

        invalid_direction = (
            ((debit > 0) & (credit > 0))
            | ((debit <= 0) & (credit <= 0))
            | (debit < 0)
            | (credit < 0)
        )

        if invalid_direction.any():
            raise ValueError(
                f"{invalid_direction.sum()} invalid "
                "debit/credit transactions."
            )

        df["transaction_type"] = "Debit"
        df.loc[credit > 0, "transaction_type"] = "Credit"
        df["amount"] = debit.where(debit > 0, credit)

        df = df.drop(
            columns=["debit", "credit"],
            errors="ignore"
        )

    # 6. Standardize transaction types
    if "transaction_type" not in df.columns:
        raise ValueError("Missing transaction_type column.")

    df["transaction_type"] = (
        df["transaction_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    invalid_types = (
        ~df["transaction_type"].isin(["Debit", "Credit"])
    ).fillna(True)

    # 7. Validate
    invalid_amounts = (
        df["amount"].isna()
        | (df["amount"] <= 0)
    )

    date_count = int(invalid_dates.sum())
    amount_count = int(invalid_amounts.sum())
    type_count = int(invalid_types.sum())

    if date_count or amount_count or type_count:
        raise ValueError(
            "Invalid bank statement data: "
            f"{date_count} invalid dates, "
            f"{amount_count} invalid amounts, "
            f"{type_count} invalid transaction types."
        )

    # 8. Standardize dates
    df["transaction_date"] = (
        pd.to_datetime(df["transaction_date"])
        .dt.strftime("%Y-%m-%d")
    )

    # 9. Flag possible duplicates
    duplicate_columns = [
        col for col in [
            "user_id",
            "account_number",
            "transaction_date",
            "transaction_type",
            "amount",
            "balance",
            "description"
        ]
        if col in df.columns
    ]

    if duplicate_columns:
        duplicates = df.duplicated(
            subset=duplicate_columns,
            keep=False
        )

        if duplicates.any():
            print(
                f"Warning: {int(duplicates.sum())} "
                "possible duplicate rows."
            )

    print("\nTransaction cleaning completed successfully!")
    print("Total transactions:", len(df))
    print(
        "Debit transactions:",
        int((df["transaction_type"] == "Debit").sum())
    )
    print(
        "Credit transactions:",
        int((df["transaction_type"] == "Credit").sum())
    )

    return df


if __name__ == "__main__":
    from pathlib import Path
    from etl.loader import load_bank_statement
    from etl.mapper import map_columns

    print("\n===== FINSIGHT AI CLEANER TEST =====")

    test_files = [
        Path("data/test_hdfc.csv"),
        Path("data/test_sbi.csv"),
        Path("data/test_icici.csv")
    ]

    tested = 0

    for file_path in test_files:
        if not file_path.exists():
            print("Missing test file:", file_path)
            continue

        print("\nTesting:", file_path.name)

        raw_df = load_bank_statement(file_path)
        mapped_df = map_columns(raw_df)
        cleaned_df = clean_bank_statement(mapped_df)

        print(
            cleaned_df[
                [
                    "transaction_date",
                    "transaction_type",
                    "amount",
                    "balance"
                ]
            ].to_string(index=False)
        )

        assert len(cleaned_df) == 3
        assert cleaned_df["transaction_type"].tolist() == [
            "Debit", "Debit", "Credit"
        ]
        assert cleaned_df["amount"].tolist() == [
            450.0, 1200.0, 35000.0
        ]

        print("PASSED:", file_path.name)
        tested += 1

    assert tested == 3, "All three test files must be present."

    print("\nAll cleaner tests passed!")
