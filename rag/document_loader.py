"""
FinSight AI - RAG Document Loader

Features:
- Load TXT and Markdown documents
- Split documents into overlapping chunks
- Preserve document source information
- Prepare text for embedding and retrieval

No API calls or database changes.
"""

from pathlib import Path
import tempfile


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(folder_path):
    """
    Read supported text documents from a folder.

    Returns a list containing document text
    and source filenames.
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Document folder not found: {folder}"
        )

    documents = []

    for file_path in sorted(folder.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size=120, overlap=25):
    """
    Split text into overlapping word-based chunks.

    chunk_size: Maximum words per chunk
    overlap: Words shared between adjacent chunks
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size - 1"
        )

    words = text.split()

    if not words:
        return []

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[
            start:start + chunk_size
        ]

        if not chunk_words:
            break

        chunks.append(
            " ".join(chunk_words)
        )

        if start + chunk_size >= len(words):
            break

    return chunks


def prepare_document_chunks(
    documents,
    chunk_size=120,
    overlap=25
):
    """
    Convert loaded documents into chunks.

    Each chunk retains its source filename.
    """

    prepared_chunks = []

    for document in documents:

        chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(chunks):

            prepared_chunks.append({
                "chunk_id": (
                    f"{document['source']}::{index}"
                ),
                "source": document["source"],
                "chunk_index": index,
                "text": chunk
            })

    return prepared_chunks


# =================================================
# OFFLINE TEST
# =================================================

if __name__ == "__main__":

    with tempfile.TemporaryDirectory() as folder:

        sample_file = (
            Path(folder) / "budgeting_guide.txt"
        )

        sample_file.write_text(
            "A monthly budget helps track income "
            "and expenses. "
            "Spending categories include groceries, "
            "transport, housing, and entertainment. "
            "Reviewing spending patterns helps "
            "identify unnecessary expenses.",
            encoding="utf-8"
        )

        documents = load_documents(folder)

        assert len(documents) == 1

        print("Document loading: PASSED")

        chunks = prepare_document_chunks(
            documents,
            chunk_size=10,
            overlap=2
        )

        assert len(chunks) > 1

        assert all(
            chunk["source"] == "budgeting_guide.txt"
            for chunk in chunks
        )

        print("Text chunking: PASSED")
        print("Source tracking: PASSED")

        print("\nGenerated chunks:")

        for chunk in chunks:
            print(
                f"\n{chunk['chunk_id']}: "
                f"{chunk['text']}"
            )

    print(
        "\nRAG DOCUMENT LOADER TEST PASSED!"
    )