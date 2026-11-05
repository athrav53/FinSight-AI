from decimal import Decimal
from pathlib import Path

from finsight.ingestion.statement_parser import parse_demo_statement
from finsight.ingestion.reconcile import reconcile


def test_demo_pdf_end_to_end():
    pdf_path = Path("data/sample_statements/sample.pdf")

    statement = parse_demo_statement(pdf_path)
    report = reconcile(statement)

    assert statement.bank == "DEMO BANK"
    assert len(statement.transactions) == 7
    assert report.is_valid
    assert report.computed_closing == Decimal("14800.00")
    assert report.total_diff == Decimal("0.00")
