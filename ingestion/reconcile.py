
"""Bank statement balance reconciliation."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Optional

from .schema import Statement, Transaction


TOLERANCE = Decimal("0.01")


@dataclass
class RowIssue:
    row_index: int
    kind: Literal["balance_mismatch", "out_of_period"]
    expected: Optional[Decimal] = None
    actual: Optional[Decimal] = None
    hint: str = ""


@dataclass
class ReconciliationReport:
    is_valid: bool
    order: Literal["oldest_first", "newest_first", "unknown"]
    computed_closing: Decimal
    stated_closing: Decimal
    total_credits: Decimal
    total_debits: Decimal
    running_balance_checked: bool
    issues: list[RowIssue] = field(default_factory=list)

    @property
    def total_diff(self) -> Decimal:
        return self.stated_closing - self.computed_closing

    def summary(self) -> str:
        status = "OK" if self.is_valid else "FAILED"

        lines = [
            f"Reconciliation {status} ({self.order})",
            f"Credits: {self.total_credits}",
            f"Debits: {self.total_debits}",
            f"Computed closing: {self.computed_closing}",
            f"Stated closing: {self.stated_closing}",
            f"Difference: {self.total_diff}",
        ]

        for issue in self.issues:
            lines.append(
                f"Row {issue.row_index}: {issue.kind} "
                f"expected={issue.expected} actual={issue.actual} "
                f"{issue.hint}"
            )

        return "\n".join(lines)


def _first_row_fits(start: Decimal, txn: Transaction) -> bool:
    return (
        txn.balance is not None
        and abs(start + txn.signed_amount - txn.balance) <= TOLERANCE
    )


def _detect_order(stmt: Statement):
    txns = stmt.transactions

    if not txns:
        return txns, "unknown"

    if _first_row_fits(stmt.opening_balance, txns[0]):
        return txns, "oldest_first"

    if _first_row_fits(stmt.opening_balance, txns[-1]):
        return list(reversed(txns)), "newest_first"

    if txns[0].txn_date > txns[-1].txn_date:
        return list(reversed(txns)), "newest_first"

    if txns[0].txn_date < txns[-1].txn_date:
        return txns, "oldest_first"

    return txns, "unknown"


def _hint(diff: Decimal, txn: Transaction) -> str:
    if abs(abs(diff) - 2 * txn.amount) <= TOLERANCE:
        return "(debit/credit likely flipped)"

    if abs(diff) > TOLERANCE:
        return "(possible missing row or incorrect amount)"

    return ""


def reconcile(stmt: Statement) -> ReconciliationReport:
    ordered, order = _detect_order(stmt)

    issues = []

    credits = sum(
        (t.amount for t in ordered if t.signed_amount > 0),
        Decimal("0"),
    )

    debits = sum(
        (t.amount for t in ordered if t.signed_amount < 0),
        Decimal("0"),
    )

    computed_closing = stmt.opening_balance + credits - debits

    has_balances = any(t.balance is not None for t in ordered)
    running = stmt.opening_balance

    for txn in ordered:
        expected = running + txn.signed_amount

        if txn.balance is not None:
            if abs(expected - txn.balance) > TOLERANCE:
                issues.append(
                    RowIssue(
                        row_index=txn.row_index,
                        kind="balance_mismatch",
                        expected=expected,
                        actual=txn.balance,
                        hint=_hint(txn.balance - expected, txn),
                    )
                )

            running = txn.balance
        else:
            running = expected

    for txn in ordered:
        if not (
            stmt.period_start <= txn.txn_date <= stmt.period_end
        ):
            issues.append(
                RowIssue(
                    row_index=txn.row_index,
                    kind="out_of_period",
                    hint=f"{txn.txn_date} is outside statement period",
                )
            )

    total_ok = (
        abs(computed_closing - stmt.closing_balance) <= TOLERANCE
    )

    return ReconciliationReport(
        is_valid=total_ok and not issues,
        order=order,
        computed_closing=computed_closing,
        stated_closing=stmt.closing_balance,
        total_credits=credits,
        total_debits=debits,
        running_balance_checked=has_balances,
        issues=issues,
    )
