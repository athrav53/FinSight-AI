
from pathlib import Path
from decimal import Decimal
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table,
    TableStyle, Spacer
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

output = Path("data/sample_statements/sample.pdf")
output.parent.mkdir(parents=True, exist_ok=True)

doc = SimpleDocTemplate(str(output), pagesize=(595, 842))
styles = getSampleStyleSheet()
story = []

story.append(Paragraph(
    "DEMO BANK - ACCOUNT STATEMENT", styles["Title"]
))
story.append(Spacer(1, 14))

for line in [
    "Customer: Demo Customer",
    "Account: XXXX1234",
    "Statement Period: 01/10/2026 - 31/10/2026",
    "Opening Balance: INR 10000.00",
    "Closing Balance: INR 14800.00",
]:
    story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 6))

transactions = [
    ("02/10/2026", "Grocery Store", "1200.00", ""),
    ("05/10/2026", "Salary Credit", "", "8000.00"),
    ("09/10/2026", "Electricity Bill", "1500.00", ""),
    ("14/10/2026", "Online Shopping", "2500.00", ""),
    ("20/10/2026", "Refund Received", "", "500.00"),
    ("25/10/2026", "Restaurant", "800.00", ""),
    ("28/10/2026", "Transfer Received", "", "2300.00"),
]

balance = Decimal("10000.00")
rows = [["Date", "Description", "Debit", "Credit", "Balance"]]

for date, description, debit, credit in transactions:
    balance += Decimal(credit or "0") - Decimal(debit or "0")
    rows.append([
        date, description, debit, credit, f"{balance:.2f}"
    ])

table = Table(rows, colWidths=[82, 170, 82, 82, 82])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ("TOPPADDING", (0, 0), (-1, -1), 9),
    ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
]))

story.append(Spacer(1, 20))
story.append(table)
story.append(Spacer(1, 20))
story.append(Paragraph(
    "Synthetic test data only - not a real bank statement.",
    styles["Italic"]
))

doc.build(story)

print(f"PDF created: {output.resolve()}")
print(f"Calculated closing balance: {balance:.2f}")
