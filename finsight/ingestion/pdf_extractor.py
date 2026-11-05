
"""Extract text and tables from text-based bank statement PDFs."""

from pathlib import Path

import pdfplumber


def extract_pdf(pdf_path: str | Path) -> dict:
    """Extract page text and tables from a PDF statement."""

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    pages = []
    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(
            pdf.pages, start=1
        ):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            pages.append({
                "page_number": page_number,
                "text": text,
                "tables": tables,
            })

            for table in tables:
                all_tables.append({
                    "page_number": page_number,
                    "rows": table,
                })

    return {
        "filename": pdf_path.name,
        "page_count": len(pages),
        "pages": pages,
        "tables": all_tables,
    }
