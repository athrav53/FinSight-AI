"""
FinSight AI - Lightweight RAG Re-ranker

Uses TF-IDF cosine similarity to re-rank
candidate chunks returned by vector search.

No paid API required.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Reranker:

    def __init__(self, top_k=3):
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        self.top_k = top_k

    def rerank(self, question, candidates):

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not candidates:
            return []

        texts = [question] + [
            candidate["text"]
            for candidate in candidates
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        tfidf_matrix = vectorizer.fit_transform(texts)

        query_vector = tfidf_matrix[0]
        document_vectors = tfidf_matrix[1:]

        scores = cosine_similarity(
            query_vector,
            document_vectors,
        )[0]

        ranked_results = []

        for candidate, score in zip(candidates, scores):

            ranked_results.append({
                **candidate,
                "rerank_score": float(score),
            })

        ranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked_results[:self.top_k]


if __name__ == "__main__":

    reranker = Reranker(top_k=2)

    candidates = [
        {
            "source": "budgeting.txt",
            "text": "A monthly budget helps track income and expenses.",
            "similarity_score": 0.82,
        },
        {
            "source": "python.txt",
            "text": "Python is a programming language.",
            "similarity_score": 0.75,
        },
        {
            "source": "emergency_fund.txt",
            "text": "An emergency fund covers unexpected expenses.",
            "similarity_score": 0.70,
        },
    ]

    results = reranker.rerank(
        "How can I create a monthly budget?",
        candidates,
    )

    print("\n===== RERANKER TEST =====")

    for result in results:
        print(
            f"Source: {result['source']} "
            f"| Rerank Score: {result['rerank_score']:.4f}"
        )

    assert results[0]["source"] == "budgeting.txt"

    print("\nRERANKER TEST PASSED!")