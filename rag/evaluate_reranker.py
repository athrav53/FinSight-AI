"""
FinSight AI - Reranker Evaluation

Compare:
1. Vector search Top 3
2. Vector search Top 10 -> TF-IDF reranking -> Top 3

Uses the existing 40-question evaluation dataset.
"""

import json
import time
from pathlib import Path

from rag.document_loader import load_documents, chunk_text
from rag.vector_store import VectorStore
from rag.reranker import Reranker


RAG_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE = RAG_DIR / "knowledge_base"

EVALUATION_FILE = (
    RAG_DIR / "evaluation" / "retrieval_questions.json"
)

REPORT_FILE = (
    RAG_DIR / "evaluation" / "reranker_results.json"
)

TOP_K = 3
CANDIDATE_K = 10


def load_test_cases():
    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:
        return json.load(file)


def prepare_chunks():
    documents = load_documents(str(KNOWLEDGE_BASE))
    chunks = []

    for document in documents:
        text_chunks = chunk_text(
            document["text"],
            chunk_size=120,
            overlap=25
        )

        for index, text in enumerate(text_chunks):
            chunks.append({
                "chunk_id": f"{document['source']}_{index}",
                "source": document["source"],
                "chunk_index": index,
                "text": text
            })

    return chunks


def calculate_metrics(results, expected):
    sources = [item["source"] for item in results]

    rank = next(
        (
            index
            for index, source in enumerate(sources, 1)
            if source == expected
        ),
        None
    )

    hit = rank is not None

    unique_sources = set(sources)

    precision = (
        int(expected in unique_sources) / len(unique_sources)
        if unique_sources
        else 0.0
    )

    return {
        "hit": int(hit),
        "reciprocal_rank": 1 / rank if hit else 0.0,
        "document_precision": precision
    }


def evaluate():
    test_cases = load_test_cases()

    if not test_cases:
        raise ValueError("Evaluation dataset is empty.")

    chunks = prepare_chunks()

    if not chunks:
        raise ValueError("No knowledge-base chunks found.")

    print(f"Loaded questions: {len(test_cases)}")
    print(f"Created chunks: {len(chunks)}")

    print("\nBuilding vector store...")

    store = VectorStore()
    store.add_documents(chunks)

    reranker = Reranker(top_k=TOP_K)

    totals = {
        "baseline": {
            "hits": 0,
            "mrr": 0.0,
            "precision": 0.0,
            "latency": 0.0
        },
        "reranked": {
            "hits": 0,
            "mrr": 0.0,
            "precision": 0.0,
            "latency": 0.0
        }
    }

    for number, test in enumerate(test_cases, 1):
        question = test["question"]
        expected = test["expected"]

        # Baseline retrieval
        start = time.perf_counter()

        baseline_results = store.search(
            question,
            top_k=TOP_K
        )

        baseline_latency = (
            time.perf_counter() - start
        ) * 1000

        # Reranked retrieval
        start = time.perf_counter()

        candidates = store.search(
            question,
            top_k=CANDIDATE_K
        )

        reranked_results = reranker.rerank(
            question,
            candidates
        )

        reranked_latency = (
            time.perf_counter() - start
        ) * 1000

        baseline_metrics = calculate_metrics(
            baseline_results,
            expected
        )

        reranked_metrics = calculate_metrics(
            reranked_results,
            expected
        )

        for name, metrics, latency in [
            ("baseline", baseline_metrics, baseline_latency),
            ("reranked", reranked_metrics, reranked_latency)
        ]:
            totals[name]["hits"] += metrics["hit"]
            totals[name]["mrr"] += metrics["reciprocal_rank"]
            totals[name]["precision"] += (
                metrics["document_precision"]
            )
            totals[name]["latency"] += latency

        print(
            f"Question {number}: "
            f"Baseline={'PASS' if baseline_metrics['hit'] else 'FAIL'} | "
            f"Reranked={'PASS' if reranked_metrics['hit'] else 'FAIL'}"
        )

    count = len(test_cases)

    report = {}

    for name, values in totals.items():
        report[name] = {
            "total_questions": count,
            "hit_at_3": values["hits"] / count,
            "mrr_at_3": values["mrr"] / count,
            "document_precision": values["precision"] / count,
            "average_latency_ms": values["latency"] / count
        }

    report["changes"] = {
        metric: report["reranked"][metric] - report["baseline"][metric]
        for metric in [
            "hit_at_3",
            "mrr_at_3",
            "document_precision",
            "average_latency_ms"
        ]
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n====================================")
    print("   FINSIGHT AI RERANKER EVALUATION")
    print("====================================")

    for name, result in [
        ("BASELINE", report["baseline"]),
        ("RERANKED", report["reranked"])
    ]:
        print(f"\n{name}")
        print(f"Hit@3: {result['hit_at_3']:.2%}")
        print(f"MRR@3: {result['mrr_at_3']:.3f}")
        print(
            f"Document Precision: "
            f"{result['document_precision']:.2%}"
        )
        print(
            f"Average Latency: "
            f"{result['average_latency_ms']:.2f} ms"
        )

    print("\nCHANGES (Reranked - Baseline)")

    print(
        f"Hit@3: {report['changes']['hit_at_3']:+.2%}"
    )

    print(
        f"MRR@3: {report['changes']['mrr_at_3']:+.3f}"
    )

    print(
        "Document Precision: "
        f"{report['changes']['document_precision']:+.2%}"
    )

    print(
        "Average Latency: "
        f"{report['changes']['average_latency_ms']:+.2f} ms"
    )

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    evaluate()