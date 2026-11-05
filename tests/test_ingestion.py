
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from finsight.ingestion.reconcile import reconcile
from finsight.ingestion.schema import Statement, Transaction, parse_money


def make_stmt(rows, opening="10,000.00", closing="12,700.00"):
    return Statement(
        bank="HDFC",
        account_masked="XXXXXX1234",
        period_start="01/10/2026",
        period_end="31/10/2026",
        opening_balance=opening,
        closing_balance=closing,
        transactions=rows,
    )


def good_rows():
    return [
        dict(
            row_index=0,
            txn_date="02/10/2026",
            description="UPI-SWIGGY",
            amount="500.00",
            txn_type="debit",
            balance="9,500.00",
        ),
        dict(
            row_index=1,
            txn_date="05-Oct-2026",
            description="SALARY CREDIT",
            amount="5,000",
            txn_type="credit",
            balance="14,500.00",
        ),
        dict(
            row_index=2,
            txn_date="10/10/26",
            description="RENT",
            amount="1,800.00",
            txn_type="debit",
            balance="12,700.00",
        ),
    ]


# ---------- parsing ----------

@pytest.mark.parametrize("raw,expected", [
    ("1,23,456.78", Decimal("123456.78")),
    ("₹500", Decimal("500")),
    ("Rs. 99.50", Decimal("99.50")),
    ("2,000.00 Dr", Decimal("-2000.00")),
    ("2,000.00 Cr", Decimal("2000.00")),
    ("(150.00)", Decimal("-150.00")),
    (1200.5, Decimal("1200.5")),
])
def test_parse_money(raw, expected):
    assert parse_money(raw) == expected


def test_dates_are_day_first():
    t = Transaction(
        row_index=0,
        txn_date="04/11/2026",
        description="x",
        amount="1",
        txn_type="debit",
    )
    assert t.txn_date == date(2026, 11, 4)


def test_rejects_bad_rows():
    with pytest.raises(ValidationError):
        Transaction(
            row_index=0,
            txn_date="04/11/2026",
            description="x",
            amount="-5",
            txn_type="debit",
        )

    with pytest.raises(ValidationError):
        Transaction(
            row_index=0,
            txn_date="not a date",
            description="x",
            amount="5",
            txn_type="debit",
        )


def test_rejects_unmasked_account():
    with pytest.raises(ValidationError):
        Statement(
            bank="SBI",
            account_masked="123456789012",
            period_start="01/10/2026",
            period_end="31/10/2026",
            opening_balance="0",
            closing_balance="0",
            transactions=[],
        )


# ---------- reconciliation ----------

def test_clean_statement_reconciles():
    r = reconcile(make_stmt(good_rows()))
    assert r.is_valid, r.summary()
    assert r.order == "oldest_first"
    assert r.total_credits == Decimal("5000")
    assert r.total_debits == Decimal("2300")


def test_newest_first_statement_is_detected():
    r = reconcile(make_stmt(list(reversed(good_rows()))))
    assert r.is_valid, r.summary()
    assert r.order == "newest_first"


def test_flipped_debit_credit_is_pinpointed():
    rows = good_rows()
    rows[2]["txn_type"] = "credit"
    r = reconcile(make_stmt(rows))
    assert not r.is_valid
    assert r.issues[0].row_index == 2
    assert "flipped" in r.issues[0].hint


def test_missing_row_is_detected():
    rows = good_rows()
    del rows[1]
    r = reconcile(make_stmt(rows))
    assert not r.is_valid
    assert r.total_diff != 0
    assert any(i.kind == "balance_mismatch" for i in r.issues)


def test_compensating_errors_caught_by_running_balance():
    rows = good_rows()
    rows[0]["amount"] = "600.00"
    rows[2]["amount"] = "1,700.00"
    r = reconcile(make_stmt(rows))

    assert r.computed_closing == r.stated_closing
    assert not r.is_valid
    assert {i.row_index for i in r.issues} == {0, 2}


def test_out_of_period_row_flagged():
    rows = good_rows()
    rows[0]["txn_date"] = "28/09/2026"
    r = reconcile(make_stmt(rows))
    assert any(
        i.kind == "out_of_period" and i.row_index == 0
        for i in r.issues
    )


def test_statement_without_running_balances_uses_total_only():
    rows = [{**r, "balance": None} for r in good_rows()]
    r = reconcile(make_stmt(rows))
    assert r.is_valid
    assert not r.running_balance_checked
