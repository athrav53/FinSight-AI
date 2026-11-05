

@'
"""Parse bank statement PDFs into validated transactions."""

import re

from .pdf_extractor import extract_pdf
from .schema import Statement, Transaction, TxnType, parse_money
from .reconcile import reconcile
from .column_mapper import detect_columns


def parse_demo_statement(pdf_path):
    """Parse synthetic DEMO BANK statements using flexible columns."""

    extracted = extract_pdf(pdf_path)

    text = "\n".join(
        page["text"] for page in extracted["pages"]
    )

    def field(label):
        match = re.search(
            rf"^{re.escape(label)}:\s*(.+)$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )

        if not match:
            raise ValueError(f"Missing field: {label}")

        return match.group(1).strip()

    period = re.search(
        r"Statement Period:\s*"
        r"(\d{2}/\d{2}/\d{4})\s*-\s*"
        r"(\d{2}/\d{2}/\d{4})",
        text,
    )

    if not period:
        raise ValueError("Statement period not found")

    transactions = []

    for table in extracted["tables"]:
        rows = table["rows"]

        if not rows:
            continue

        # Detect column positions dynamically
        try:
            columns = detect_columns(rows[0])
        except ValueError:
            continue

        for row in rows[1:]:
            if not row or all(not cell for cell in row):
                continue

            if len(row) <= max(columns.values()):
                raise ValueError(
                    f"Incomplete transaction row: {row}"
                )

            date = row[columns["date"]]
            description = row[columns["description"]]
            debit = row[columns["debit"]]
            credit = row[columns["credit"]]
            balance = row[columns["balance"]]

            debit_amount = parse_money(debit)
            credit_amount = parse_money(credit)

            # Exactly one of debit or credit must contain an amount
            if (debit_amount is None) == (credit_amount is None):
                raise ValueError(
                    f"Invalid debit/credit values: {row}"
                )

            amount = (
                debit_amount
                if debit_amount is not None
                else credit_amount
            )

            txn_type = (
                TxnType.DEBIT
                if debit_amount is not None
                else TxnType.CREDIT
            )

            transactions.append(
                Transaction(
                    row_index=len(transactions) + 1,
                    txn_date=date,
                    description=description,
                    amount=amount,
                    txn_type=txn_type,
                    balance=balance,
                )
            )

    if not transactions:
        raise ValueError(
            "No valid transaction tables found in PDF"
        )

    statement = Statement(
        bank="DEMO BANK",
        account_masked=field("Account"),
        period_start=period.group(1),
        period_end=period.group(2),
        opening_balance=field("Opening Balance"),
        closing_balance=field("Closing Balance"),
        transactions=transactions,
    )

    return statement


if __name__ == "__main__":

    pdf_path = "data/sample_statements/sample.pdf"

    statement = parse_demo_statement(pdf_path)

    report = reconcile(statement)

    print("Bank:", statement.bank)
    print("Transactions:", len(statement.transactions))
    print(report.summary())
'@ | Set-Content -Path ".\finsight\ingestion\statement_parser.py" -Encoding UTF8
