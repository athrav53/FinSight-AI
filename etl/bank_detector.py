
"""
FinSight AI - Bank Statement Detector

Supported formats:
1. HDFC
2. SBI
3. ICICI
4. FinSight Generic
"""

import re
import pandas as pd


def normalize_column(column):
    """Normalize column names for comparison."""
    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", " ", column)
    return " ".join(column.split())


BANK_PATTERNS = {
    "HDFC": {
        "date",
        "narration",
        "chq ref no",
        "value dt",
        "withdrawal amt",
        "deposit amt",
        "closing balance"
    },

    "SBI": {
        "txn date",
        "value date",
        "description",
        "ref no cheque no",
        "debit",
        "credit",
        "balance"
    },

    "ICICI": {
        "transaction date",
        "value date",
        "transaction remarks",
        "withdrawal amount inr",
        "deposit amount inr",
        "balance inr"
    },

    "FinSight Generic": {
        "user id",
        "account number",
        "transaction date",
        "transaction type",
        "category",
        "merchant",
        "description",
        "amount",
        "balance"
    }
}


def detect_bank(df):
    """
    Detect the statement format from column names.

    Returns:
        HDFC, SBI, ICICI, FinSight Generic, or Unknown
    """

    if df is None or len(df.columns) == 0:
        return "Unknown"

    columns = {
        normalize_column(column)
        for column in df.columns
    }

    scores = {}

    for bank, patterns in BANK_PATTERNS.items():
        scores[bank] = len(columns.intersection(patterns))

    best_bank = max(scores, key=scores.get)
    best_score = scores[best_bank]

    # Require at least 3 matching columns
    if best_score < 3:
        return "Unknown"

    # Require at least 45% of expected columns
    coverage = best_score / len(BANK_PATTERNS[best_bank])

    if coverage < 0.45:
        return "Unknown"

    # Avoid ambiguous detection
    matching_banks = [
        bank
        for bank, score in scores.items()
        if score == best_score
    ]

    if len(matching_banks) > 1:
        return "Unknown"

    return best_bank


if __name__ == "__main__":
    import pandas as pd

    print("\n===== FINSIGHT AI BANK DETECTOR =====")

    hdfc = pd.DataFrame(columns=[
        "Date", "Narration", "Chq./Ref.No.",
        "Value Dt", "Withdrawal Amt.",
        "Deposit Amt.", "Closing Balance"
    ])

    sbi = pd.DataFrame(columns=[
        "Txn Date", "Value Date", "Description",
        "Ref No./Cheque No.", "Debit", "Credit", "Balance"
    ])

    icici = pd.DataFrame(columns=[
        "Transaction Date", "Value Date",
        "Transaction Remarks",
        "Withdrawal Amount (INR)",
        "Deposit Amount (INR)",
        "Balance (INR)"
    ])

    print("HDFC:", detect_bank(hdfc))
    print("SBI:", detect_bank(sbi))
    print("ICICI:", detect_bank(icici))