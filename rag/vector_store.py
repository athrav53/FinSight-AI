"""
FinSight AI - RAG Vector Search Engine

Features:
- Store document chunks and embeddings
- Perform semantic similarity search
- Return top-k relevant passages
- Preserve source information
- No paid API required
- No database modifications
"""

import numpy as np

from rag.embeddings import (
    generate_embeddings,
    embed_document_chunks,
)


class VectorStore:
    """
    In-memory vector store for document retrieval.

    This first version does not persist vectors to disk.
    """

    def __init__(self):
        self.chunks = []
        self.matrix = None
        self.dimension = None

    def add_documents(self, chunks):
        """
        Generate embeddings and build the vector index.
        """

        if not chunks:
            raise ValueError(
                "No document chunks provided."
            )

        embedded_chunks = embed_document_chunks(chunks)

        vectors = np.asarray(
            [
                item["embedding"]
                for item in embedded_chunks
            ],
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must form a 2D matrix."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embeddings contain invalid values."
            )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Zero-length embedding detected."
            )

        # Normalize vectors for cosine similarity.
        normalized_vectors = vectors / norms

        self.matrix = normalized_vectors

        self.dimension = vectors.shape[1]

        self.chunks = [
            {
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "text": item["text"],
            }
            for item in embedded_chunks
        ]

        return len(self.chunks)

    def search(self, question, top_k=3):
        """
        Retrieve the most semantically relevant chunks.

        Returns:
            List of dictionaries containing:
            - chunk_id
            - source
            - text
            - similarity_score
        """

        if self.matrix is None:
            raise RuntimeError(
                "Vector store is empty. "
                "Add documents before searching."
            )

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question must be a non-empty string."
            )

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(
                "top_k must be a positive integer."
            )

        query_vector = np.asarray(
            generate_embeddings([question])[0],
            dtype=np.float32,
        )

        if len(query_vector) != self.dimension:
            raise ValueError(
                "Query embedding dimension mismatch."
            )

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise ValueError(
                "Query embedding has zero magnitude."
            )

        query_vector = query_vector / query_norm

        # Cosine similarity between query and documents.
        scores = self.matrix @ query_vector

        k = min(top_k, len(self.chunks))

        best_indices = np.argsort(scores)[::-1][:k]

        results = []

        for index in best_indices:

            chunk = self.chunks[int(index)]

            results.append({
                **chunk,
                "similarity_score": round(
                    float(scores[index]),
                    4,
                ),
            })

        return results


# =====================================================
# INTEGRATION TEST
# =====================================================

if __name__ == "__main__":

    print("\n===== FINSIGHT AI VECTOR SEARCH TEST =====")

    sample_chunks = [
        {
            "chunk_id": "budgeting.txt::0",
            "source": "budgeting.txt",
            "chunk_index": 0,
            "text": (
                "A monthly budget helps track income "
                "and expenses. Reviewing spending "
                "categories can help identify areas "
                "where costs can be reduced."
            ),
        },
        {
            "chunk_id": "emergency_fund.txt::0",
            "source": "emergency_fund.txt",
            "chunk_index": 0,
            "text": (
                "An emergency fund is money reserved "
                "for unexpected expenses such as "
                "medical bills or urgent repairs."
            ),
        },
        {
            "chunk_id": "python.txt::0",
            "source": "python.txt",
            "chunk_index": 0,
            "text": (
                "Python is a programming language "
                "used for data analysis and automation."
            ),
        },
    ]

    store = VectorStore()

    count = store.add_documents(sample_chunks)

    assert count == 3

    print("Document indexing: PASSED")
    print("Indexed chunks:", count)

    results = store.search(
        "How can I create a monthly budget?",
        top_k=2,
    )

    assert len(results) == 2

    print("Semantic retrieval: PASSED")

    print("\nTop retrieved passages:")

    for result in results:
        print(
            f"\nSource: {result['source']}"
            f"\nScore: {result['similarity_score']}"
            f"\nText: {result['text']}"
        )

    assert results[0]["source"] == "budgeting.txt"

    print("\nRelevant document ranking: PASSED")

    print("\nRAG VECTOR SEARCH TEST PASSED!")