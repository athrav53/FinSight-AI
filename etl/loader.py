
"""
FinSight AI - ETL Pipeline
Module: CSV File Loader
"""

from pathlib import Path
import pandas as pd


def load_bank_statement(file_path):
    """
    Load a bank statement from a CSV file.
    """

    path = Path(file_path)

    # Check whether file exists
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    # Read CSV
    df = pd.read_csv(
        path,
        dtype={"account_number": str}
    )

    # Check whether file is empty
    if df.empty:
        raise ValueError(
            "Bank statement is empty!"
        )

    print("Bank statement loaded successfully!")
    print("Total rows:", len(df))
    print("Total columns:", len(df.columns))

    return df


if __name__ == "__main__":

    # Automatically find the newest statement
    data_folder = Path("data")

    files = list(
        data_folder.glob("sample_bank_statement*.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No bank statement found in data folder."
        )

    latest_file = max(
        files,
        key=lambda file: file.stat().st_mtime
    )

    df = load_bank_statement(latest_file)

    print("\nFile:", latest_file.name)

    print("\nFirst 5 transactions:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())
