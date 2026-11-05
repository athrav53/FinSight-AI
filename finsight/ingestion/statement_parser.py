"""Parse synthetic DEMO BANK PDF statements."""

import re

from .pdf_extractor import extract_pdf
from .schema import Statement, Transaction, TxnType, parse_money
from .reconcile import reconcile


def parse_demo_statement(pdf_path):
    extracted = extract_pdf(pdf_path)
    text = "\n".join(p["text"] for p in extracted["pages"])

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
        r"Statement Period:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})",
        text,
    )
    if not period:
        raise ValueError("Statement period not found")

    transactions = []

    for table in extracted["tables"]:
        rows = table["rows"]
        if not rows:
            continue

        headers = [str(x or "").strip().lower() for x in rows[0]]
        if headers != ["date", "description", "debit", "credit", "balance"]:
            continue

        for row in rows[1:]:
            if not row or all(not cell for cell in row):
                continue

            date, description, debit, credit, balance = row
            debit_amount = parse_money(debit)
            credit_amount = parse_money(credit)

            if (debit_amount is None) == (credit_amount is None):
                raise ValueError(f"Invalid debit/credit: {row}")

            transactions.append(
                Transaction(
                    row_index=len(transactions) + 1,
                    txn_date=date,
                    description=description,
                    amount=debit_amount if debit_amount is not None else credit_amount,
                    txn_type=TxnType.DEBIT if debit_amount is not None else TxnType.CREDIT,
                    balance=balance,
                )
            )

    if not transactions:
        raise ValueError("No transactions found")

    return Statement(
        bank="DEMO BANK",
        account_masked=field("Account"),
        period_start=period.group(1),
        period_end=period.group(2),
        opening_balance=field("Opening Balance"),
        closing_balance=field("Closing Balance"),
        transactions=transactions,
    )


if __name__ == "__main__":
    statement = parse_demo_statement("data/sample_statements/sample.pdf")
    report = reconcile(statement)
    print("Bank:", statement.bank)
    print("Transactions:", len(statement.transactions))
    print(report.summary())
