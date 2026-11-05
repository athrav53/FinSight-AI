
import pytest

from finsight.ingestion.column_mapper import detect_columns


@pytest.mark.parametrize(
    "headers",
    [
        ["Date", "Description", "Debit", "Credit", "Balance"],
        ["Txn Date", "Narration", "Withdrawal", "Deposit", "Running Balance"],
        ["Transaction Date", "Particulars", "Dr", "Cr", "Closing Balance"],
        ["Value Date", "Remarks", "Withdrawal Amt", "Deposit Amt", "Balance"],
    ],
)
def test_bank_header_variations(headers):
    assert detect_columns(headers) == {
        "date": 0,
        "description": 1,
        "debit": 2,
        "credit": 3,
        "balance": 4,
    }


def test_reordered_columns():
    headers = [
        "Balance", "Credit", "Narration",
        "Txn Date", "Debit"
    ]

    assert detect_columns(headers) == {
        "date": 3,
        "description": 2,
        "debit": 4,
        "credit": 1,
        "balance": 0,
    }


def test_missing_column():
    headers = ["Date", "Description", "Debit", "Balance"]

    with pytest.raises(ValueError, match="Missing required columns"):
        detect_columns(headers)
