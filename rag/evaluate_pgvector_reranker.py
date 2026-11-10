
import json
import time
from pathlib import Path

from rag.chat_ui import prepare_knowledge_base, create_collection_id
from rag.pgvector_store import PgVectorStore
from rag.reranker import Reranker
from rag.evaluate_retrieval import load_test_cases, calculate_metrics

REPORT = Path("rag/evaluation/pgvector_reranker_results.json")


def main():
    documents, _ = prepare_knowledge_base()
    collection_id = create_collection_id(documents)

    store = PgVectorStore(collection_id=collection_id)
    reranker = Reranker(top_k=3)
    tests = load_test_cases()

    report = {"collection_id": collection_id, "questions": []}

    for index, test in enumerate(tests, 1):
        question = test["question"]
        expected = test["expected"]

        start = time.perf_counter()
        candidates = store.search(question, top_k=10)
        retrieval_ms = (time.perf_counter() - start) * 1000

        baseline = candidates[:3]

        start = time.perf_counter()
        reranked = reranker.rerank(question, candidates)
        reranking_ms = (time.perf_counter() - start) * 1000

        before = calculate_metrics(baseline, expected, 3)
        after = calculate_metrics(reranked, expected, 3)

        report["questions"].append({
            "question": question,
            "expected": expected,
            "baseline": before,
            "reranked": after,
            "retrieval_ms": retrieval_ms,
            "reranking_ms": reranking_ms,
            "baseline_sources": [r["source"] for r in baseline],
            "reranked_sources": [r["source"] for r in reranked],
        })

        print(
            f"{index}/{len(tests)} "
            f"Baseline={before['hit']} "
            f"Reranked={after['hit']}"
        )

    count = len(tests)

    def summary(key):
        rows = [r[key] for r in report["questions"]]
        return {
            "hit_at_3": sum(r["hit"] for r in rows) / count,
            "mrr_at_3": sum(r["reciprocal_rank"] for r in rows) / count,
        }

    report["baseline"] = summary("baseline")
    report["reranked"] = summary("reranked")
    report["average_reranking_ms"] = (
        sum(r["reranking_ms"] for r in report["questions"]) / count
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nBASELINE:", report["baseline"])
    print("RERANKED:", report["reranked"])
    print("Average reranking overhead (ms):", round(report["average_reranking_ms"], 2))
    print("Saved:", REPORT)


if __name__ == "__main__":
    main()
