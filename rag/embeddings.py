"""
FinSight AI - Local RAG Embeddings

Features:
- Generate embeddings using Ollama
- No paid API required
- Supports multiple document chunks
- Validates embedding responses
- Tests semantic similarity

Embedding model: nomic-embed-text
"""

import os
import math

import requests
from dotenv import load_dotenv
from pathlib import Path

from rag.document_loader import (
    load_documents,
    prepare_document_chunks,
)


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text"
)


# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

def generate_embeddings(texts):
    """
    Generate vector embeddings for multiple texts.

    Args:
        texts: List of strings.

    Returns:
        List of embedding vectors.
    """

    if not isinstance(texts, list):
        raise TypeError(
            "texts must be a list"
        )

    if not texts:
        return []

    if not all(
        isinstance(text, str) and text.strip()
        for text in texts
    ):
        raise ValueError(
            "All texts must be non-empty strings"
        )

    try:

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": texts
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        embeddings = result.get(
            "embeddings"
        )

        if not isinstance(embeddings, list):
            raise ValueError(
                "Invalid embeddings response"
            )

        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedding count mismatch"
            )

        if not embeddings:
            raise ValueError(
                "No embeddings returned"
            )

        vector_size = len(embeddings[0])

        if vector_size == 0:
            raise ValueError(
                "Empty embedding vector"
            )

        for vector in embeddings:

            if len(vector) != vector_size:
                raise ValueError(
                    "Inconsistent vector dimensions"
                )

            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    "Invalid embedding values"
                )

        return embeddings

    except requests.RequestException as error:

        raise RuntimeError(
            "Could not connect to Ollama embedding API. "
            "Make sure Ollama is running and "
            f"'{EMBEDDING_MODEL}' is downloaded."
        ) from error


# =====================================================
# EMBED DOCUMENT CHUNKS
# =====================================================

def embed_document_chunks(chunks):
    """
    Attach embeddings to document chunks.
    """

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    embedded_chunks = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        embedded_chunks.append({
            **chunk,
            "embedding": embedding
        })

    return embedded_chunks


# =====================================================
# COSINE SIMILARITY
# =====================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calculate similarity between two vectors.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vector dimensions do not match"
        )

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b
        )
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    print(
        "\n===== FINSIGHT AI EMBEDDING TEST ====="
    )

    sample_texts = [
        "How can I reduce my monthly expenses?",
        "What are ways to save money every month?",
        "Python is a programming language."
    ]

    vectors = generate_embeddings(
        sample_texts
    )

    assert len(vectors) == 3

    print(
        "Embedding generation: PASSED"
    )

    print(
        "Vector dimensions:",
        len(vectors[0])
    )

    similar_score = cosine_similarity(
        vectors[0],
        vectors[1]
    )

    unrelated_score = cosine_similarity(
        vectors[0],
        vectors[2]
    )

    print(
        "Related sentence similarity:",
        round(similar_score, 4)
    )

    print(
        "Unrelated sentence similarity:",
        round(unrelated_score, 4)
    )

    assert similar_score > unrelated_score

    print(
        "Semantic similarity test: PASSED"
    )

    print(
        "\nRAG EMBEDDINGS TEST PASSED!"
    )