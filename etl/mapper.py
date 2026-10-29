
"""
FinSight AI - ETL Column Mapper

Standardizes column names from different bank statements.
"""

import pandas as pd


COLUMN_MAPPING = {
    "date": "transaction_date",
    "txn date": "transaction_date",
    "transaction date": "transaction_date",
    "value date": "transaction_date",

    "narration": "description",
    "remarks": "description",
    "transaction remarks": "description",
    "particulars": "description",

    "withdrawal amt.": "debit",
    "withdrawal amount": "debit",
    "debit": "debit",
    "debit amount": "debit",

    "deposit amt.": "credit",
    "deposit amount": "credit",
    "credit": "credit",
    "credit amount": "credit",

    "closing balance": "balance",
    "running balance": "balance",
    "balance": "balance",

    "account no": "account_number",
    "account number": "account_number"
}


def map_columns(df):
    """
    Convert different bank column names into standard names.
    """

    df = df.copy()

    # Normalize names before mapping
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    df = df.rename(columns=COLUMN_MAPPING)

    # Detect duplicate standardized columns
    if df.columns.duplicated().any():
        raise ValueError(
            "Multiple source columns map to the same name."
        )

    print("Column mapping completed successfully!")

    return df


if __name__ == "__main__":

    # Sample HDFC-style statement
    sample_data = {
        "Date": ["10/10/2026", "11/10/2026"],
        "Narration": ["UPI/Swiggy", "NEFT/Salary"],
        "Withdrawal Amt.": [450, 0],
        "Deposit Amt.": [0, 45000],
        "Closing Balance": [15500, 60500]
    }

    df = pd.DataFrame(sample_data)

    print("\nBefore mapping:")
    print(df.columns.tolist())

    mapped_df = map_columns(df)

    print("\nAfter mapping:")
    print(mapped_df.columns.tolist())

    print("\nMapped data:")
    print(mapped_df)
