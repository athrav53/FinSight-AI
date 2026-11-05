"""Normalize bank statement transaction column names."""

import re

COLUMN_ALIASES = {
    "date": ["date", "txn date", "transaction date", "value date"],
    "description": ["description", "narration", "particulars", "remarks"],
    "debit": ["debit", "withdrawal", "withdrawal amt", "dr"],
    "credit": ["credit", "deposit", "deposit amt", "cr"],
    "balance": ["balance", "closing balance", "running balance"],
}


def normalize_header(value):
    """Normalize whitespace, punctuation and capitalization."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def detect_columns(headers):
    """Map PDF column positions to standard transaction fields."""
    normalized = [normalize_header(h) for h in headers]
    mapping = {}

    for field, aliases in COLUMN_ALIASES.items():
        candidates = {normalize_header(alias) for alias in aliases}

        for index, header in enumerate(normalized):
            if header in candidates:
                mapping[field] = index
                break

    required = {"date", "description", "debit", "credit", "balance"}
    missing = required - set(mapping)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Duplicate column positions detected")

    return mapping
