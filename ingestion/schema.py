
"""Pydantic models for Indian bank statement validation."""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


DATE_FORMATS = (
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%d.%m.%Y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%d %b %y",
    "%d-%B-%Y",
    "%Y-%m-%d",
)

_STRIP = re.compile(r"(₹|INR|Rs\.?|,|\s)", re.IGNORECASE)


def parse_money(value) -> Optional[Decimal]:
    """Convert Indian currency formats into Decimal."""

    if value is None or value == "":
        return None

    if isinstance(value, Decimal):
        return value

    if isinstance(value, (int, float)):
        return Decimal(str(value))

    s = str(value).strip()
    negative = False

    suffix = s[-2:].lower()

    if suffix in ("dr", "cr"):
        negative = suffix == "dr"
        s = s[:-2]

    s = _STRIP.sub("", s)

    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]

    try:
        amount = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid money value: {value!r}") from exc

    return -amount if negative else amount


def parse_date(value) -> date:
    """Parse Indian day-first date formats."""

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    value = str(value).strip()

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    raise ValueError(f"Invalid date: {value!r}")


class TxnType(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class Transaction(BaseModel):
    row_index: int = Field(ge=0)
    txn_date: date
    description: str = Field(min_length=1)
    amount: Decimal = Field(gt=0)
    txn_type: TxnType
    balance: Optional[Decimal] = None
    ref_no: Optional[str] = None

    @field_validator("txn_date", mode="before")
    @classmethod
    def validate_date(cls, value):
        return parse_date(value)

    @field_validator("amount", "balance", mode="before")
    @classmethod
    def validate_money(cls, value):
        return parse_money(value)

    @field_validator("description")
    @classmethod
    def clean_description(cls, value: str) -> str:
        return " ".join(value.split())

    @property
    def signed_amount(self) -> Decimal:
        if self.txn_type == TxnType.CREDIT:
            return self.amount
        return -self.amount


class Statement(BaseModel):
    bank: str
    account_masked: Optional[str] = None
    period_start: date
    period_end: date
    opening_balance: Decimal
    closing_balance: Decimal
    transactions: list[Transaction]

    @field_validator("period_start", "period_end", mode="before")
    @classmethod
    def validate_dates(cls, value):
        return parse_date(value)

    @field_validator("opening_balance", "closing_balance", mode="before")
    @classmethod
    def validate_balances(cls, value):
        return parse_money(value)

    @field_validator("account_masked")
    @classmethod
    def validate_account(cls, value):
        if value and re.search(r"\d{5,}", value):
            raise ValueError("Account number must be masked")
        return value

    @model_validator(mode="after")
    def validate_period(self):
        if self.period_start > self.period_end:
            raise ValueError("Start date cannot exceed end date")
        return self
