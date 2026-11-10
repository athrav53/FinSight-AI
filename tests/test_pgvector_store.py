
"""
Tests for the FinSight AI PostgreSQL pgvector store.

These tests mock PostgreSQL and Ollama so they do not
require a running database or embedding model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.pgvector_store import PgVectorStore


def make_store():
    """Create a store without connecting to PostgreSQL."""
    with patch.object(PgVectorStore, "_initialize_database"):
        return PgVectorStore()


def test_pgvector_store_dimension():
    store = make_store()
    assert store.dimension == 768


def test_valid_embedding():
    vector = np.ones(768, dtype=np.float32)

    result = PgVectorStore._validate_vector(vector)

    assert result.shape == (768,)
    assert result.dtype == np.float32


def test_invalid_embedding_dimension():
    with pytest.raises(ValueError, match="dimension mismatch"):
        PgVectorStore._validate_vector([0.1] * 10)


def test_zero_embedding_rejected():
    with pytest.raises(ValueError, match="Zero-length"):
        PgVectorStore._validate_vector([0.0] * 768)


def test_nan_embedding_rejected():
    vector = np.ones(768, dtype=np.float32)
    vector[0] = np.nan

    with pytest.raises(ValueError, match="invalid values"):
        PgVectorStore._validate_vector(vector)


def test_empty_documents_rejected():
    store = make_store()

    with pytest.raises(ValueError, match="No document chunks"):
        store.add_documents([])


def test_empty_question_rejected():
    store = make_store()

    with pytest.raises(ValueError, match="non-empty"):
        store.search("   ")


def test_invalid_top_k_rejected():
    store = make_store()

    with pytest.raises(ValueError, match="positive integer"):
        store.search("What is a budget?", top_k=0)


def test_search_returns_expected_structure():
    store = make_store()

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        (
            "test-chunk-1",
            "test_document.txt",
            0,
            "A budget tracks income and expenses.",
            0.95,
        )
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_conn

    with (
        patch(
            "rag.pgvector_store.generate_embeddings",
            return_value=[[0.1] * 768],
        ),
        patch(
            "rag.pgvector_store.get_connection",
            return_value=mock_context,
        ),
        patch("rag.pgvector_store.register_vector"),
    ):
        results = store.search("How do budgets work?", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "test-chunk-1"
    assert results[0]["source"] == "test_document.txt"
    assert results[0]["similarity_score"] == 0.95

    executed_sql = mock_cursor.execute.call_args.args[0]
    assert "%s::vector" in executed_sql


def test_add_documents_inserts_chunks():
    store = make_store()

    embedded_chunks = [
        {
            "chunk_id": "test-chunk-1",
            "source": "test_document.txt",
            "chunk_index": 0,
            "text": "A budget tracks income and expenses.",
            "embedding": [0.1] * 768,
        }
    ]

    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_conn

    with (
        patch(
            "rag.pgvector_store.embed_document_chunks",
            return_value=embedded_chunks,
        ),
        patch(
            "rag.pgvector_store.get_connection",
            return_value=mock_context,
        ),
        patch("rag.pgvector_store.register_vector"),
    ):
        count = store.add_documents(
            [{"text": "A budget tracks income and expenses."}]
        )

    assert count == 1
    assert mock_cursor.execute.call_count == 1

    executed_sql = mock_cursor.execute.call_args.args[0]
    assert "ON CONFLICT (collection_id, chunk_id)" in executed_sql

