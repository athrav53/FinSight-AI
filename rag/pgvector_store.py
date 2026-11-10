

"""
FinSight AI - PostgreSQL pgvector Store

Features:
- Persistent embeddings in PostgreSQL
- Ollama nomic-embed-text (768 dimensions)
- HNSW cosine similarity search
- Collection isolation
- Safe parameterized SQL
- Compatible with existing RAG pipeline
- Existing NumPy VectorStore remains unchanged
"""

import math
import uuid

import numpy as np
from pgvector.psycopg import register_vector

from db.connection import get_connection
from rag.vector_store import VectorStore
from rag.embeddings import (
    embed_document_chunks,
    generate_embeddings,
)


EMBEDDING_DIMENSION = 768


class PgVectorStore(VectorStore):
    """
    PostgreSQL-backed vector store with collection isolation.

    Each collection can contain chunks with identical chunk IDs
    without overwriting chunks from another collection.
    """

    def __init__(self, collection_id=None):
        super().__init__()

        if collection_id is None:
            collection_id = str(uuid.uuid4())

        if (
            not isinstance(collection_id, str)
            or not collection_id.strip()
        ):
            raise ValueError(
                "collection_id must be a non-empty string."
            )

        self.collection_id = collection_id
        self.dimension = EMBEDDING_DIMENSION

        self._initialize_database()

    def _initialize_database(self):
        """
        Ensure pgvector storage exists and verify its schema.

        Does not drop tables, delete records, or rebuild
        the primary key during repeated initialization.
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:

                cursor.execute(
                    "CREATE EXTENSION IF NOT EXISTS vector"
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        collection_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(768) NOT NULL,
                        CONSTRAINT rag_chunks_pkey
                            PRIMARY KEY (collection_id, chunk_id)
                    )
                    """
                )

                # Support existing databases where the
                # collection_id column was added by migration.
                cursor.execute(
                    """
                    ALTER TABLE rag_chunks
                    ADD COLUMN IF NOT EXISTS collection_id TEXT
                    """
                )

                cursor.execute(
                    """
                    UPDATE rag_chunks
                    SET collection_id = 'legacy'
                    WHERE collection_id IS NULL
                    """
                )

                cursor.execute(
                    """
                    ALTER TABLE rag_chunks
                    ALTER COLUMN collection_id SET NOT NULL
                    """
                )

                # Verify the primary key without dropping it.
                cursor.execute(
                    """
                    SELECT pg_get_constraintdef(oid)
                    FROM pg_constraint
                    WHERE conname = 'rag_chunks_pkey'
                      AND conrelid = 'rag_chunks'::regclass
                    """
                )

                result = cursor.fetchone()

                if (
                    result is None
                    or result[0]
                    != "PRIMARY KEY (collection_id, chunk_id)"
                ):
                    raise RuntimeError(
                        "rag_chunks requires the composite "
                        "primary key (collection_id, chunk_id). "
                        "Run the database migration first."
                    )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_rag_chunks_embedding
                    ON rag_chunks
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_rag_chunks_collection
                    ON rag_chunks (collection_id)
                    """
                )

    @staticmethod
    def _validate_vector(vector):
        """
        Validate an Ollama embedding and convert it
        to PostgreSQL-compatible float32.
        """

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "Embedding must be a 1D vector."
            )

        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"received {len(vector)}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Embedding contains invalid values."
            )

        if np.linalg.norm(vector) == 0:
            raise ValueError(
                "Zero-length embedding detected."
            )

        return vector

    def add_documents(self, chunks):
        """
        Embed document chunks and persist them in PostgreSQL.

        Existing chunks in the same collection are updated.
        Chunks in other collections remain untouched.
        """

        if not chunks:
            raise ValueError(
                "No document chunks provided."
            )

        embedded_chunks = embed_document_chunks(chunks)

        validated_chunks = []

        for item in embedded_chunks:
            vector = self._validate_vector(
                item["embedding"]
            )

            validated_chunks.append(
                (
                    self.collection_id,
                    item["chunk_id"],
                    item["source"],
                    item["chunk_index"],
                    item["text"],
                    vector,
                )
            )

        with get_connection() as conn:
            register_vector(conn)

            with conn.cursor() as cursor:

                for item in validated_chunks:
                    cursor.execute(
                        """
                        INSERT INTO rag_chunks (
                            collection_id,
                            chunk_id,
                            source,
                            chunk_index,
                            content,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)

                        ON CONFLICT (collection_id, chunk_id)
                        DO UPDATE SET
                            source = EXCLUDED.source,
                            chunk_index = EXCLUDED.chunk_index,
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding
                        """,
                        item,
                    )

        return len(validated_chunks)

    def search(self, question, top_k=3):
        """
        Search only within the selected collection.

        Uses PostgreSQL pgvector cosine distance.
        Returns results compatible with the existing
        RAG pipeline.
        """

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(
                "Question must be a non-empty string."
            )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        embeddings = generate_embeddings(
            [question]
        )

        query_vector = self._validate_vector(
            embeddings[0]
        )

        with get_connection() as conn:
            register_vector(conn)

            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        source,
                        chunk_index,
                        content,
                        1 - (embedding <=> %s::vector)
                            AS similarity_score
                    FROM rag_chunks
                    WHERE collection_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        self.collection_id,
                        query_vector,
                        top_k,
                    ),
                )

                rows = cursor.fetchall()

        results = []

        for row in rows:
            similarity = float(row[4])

            if not math.isfinite(similarity):
                raise ValueError(
                    "Invalid similarity score returned."
                )

            results.append(
                {
                    "chunk_id": row[0],
                    "source": row[1],
                    "chunk_index": row[2],
                    "text": row[3],
                    "similarity_score": round(
                        similarity,
                        4,
                    ),
                }
            )

        return results


