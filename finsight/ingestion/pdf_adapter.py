
import pandas as pd

from .statement_parser import parse_demo_statement


def pdf_to_dataframe(pdf_path, user_id=1):
    """Convert a parsed demo PDF into the existing ETL format."""

    statement = parse_demo_statement(pdf_path)

    rows = []

    for txn in statement.transactions:
        rows.append({
            "user_id": user_id,
            "account_number": statement.account_masked,
            "transaction_date": txn.txn_date.isoformat(),
            "transaction_type": txn.txn_type.value.capitalize(),
            "category": None,
            "merchant": None,
            "description": txn.description,
            "amount": float(txn.amount),
            "balance": float(txn.balance),
        })

    return pd.DataFrame(rows)
