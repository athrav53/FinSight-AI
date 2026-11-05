import json
import time
from pathlib import Path

from rag.document_loader import load_documents, chunk_text
from rag.vector_store import VectorStore


# ==========================================
# PROJECT PATHS
# ==========================================

RAG_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = RAG_DIR / "knowledge_base"
EVALUATION_DIR = RAG_DIR / "evaluation"

EVALUATION_FILE = EVALUATION_DIR / "retrieval_questions.json"
REPORT_FILE = EVALUATION_DIR / "comparison_results.json"

BASELINE_K = 3
EXPERIMENT_K = 5


# ==========================================
# LOAD EVALUATION QUESTIONS
# ==========================================

def load_test_cases():

    with open(EVALUATION_FILE, "r", encoding="utf-8-sig") as file:
        test_cases = json.load(file)

    if not test_cases:
        raise ValueError("Evaluation dataset is empty.")

    return test_cases


# ==========================================
# PREPARE KNOWLEDGE BASE
# ==========================================

def prepare_chunks():

    print("Loading knowledge base...")

    documents = load_documents(str(KNOWLEDGE_BASE))
    chunks = []

    for document in documents:

        text_chunks = chunk_text(
            document["text"],
            chunk_size=120,
            overlap=25,
        )

        for index, text in enumerate(text_chunks):

            chunks.append({
                "chunk_id": f"{document['source']}_{index}",
                "source": document["source"],
                "chunk_index": index,
                "text": text,
            })

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    return chunks


# ==========================================
# CALCULATE RETRIEVAL METRICS
# ==========================================

def calculate_metrics(results, expected, top_k):

    retrieved = [
        result["source"]
        for result in results[:top_k]
    ]

    rank = next(
        (
            index
            for index, source in enumerate(retrieved, start=1)
            if source == expected
        ),
        None,
    )

    hit = rank is not None

    reciprocal_rank = 1 / rank if hit else 0.0

    unique_retrieved = list(dict.fromkeys(retrieved))

    relevant_count = sum(
        source == expected
        for source in unique_retrieved
    )

    precision = (
        relevant_count / len(unique_retrieved)
        if unique_retrieved
        else 0.0
    )

    recall = 1.0 if hit else 0.0

    return {
        "hit": int(hit),
        "reciprocal_rank": reciprocal_rank,
        "document_precision": precision,
        "document_recall": recall,
    }


# ==========================================
# EVALUATE A CONFIGURATION
# ==========================================

def evaluate_configuration(vector_store, test_cases, top_k):

    totals = {
        "hit": 0,
        "reciprocal_rank": 0.0,
        "document_precision": 0.0,
        "document_recall": 0.0,
    }

    common_k_totals = {
        "hit": 0,
        "reciprocal_rank": 0.0,
    }

    latencies_ms = []

    for number, test in enumerate(test_cases, start=1):

        question = test["question"]
        expected = test["expected"]

        start = time.perf_counter()

        results = vector_store.search(
            question,
            top_k=top_k,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies_ms.append(elapsed_ms)

        metrics = calculate_metrics(
            results,
            expected,
            top_k,
        )

        for key in totals:
            totals[key] += metrics[key]

        # Fair comparison: both configurations evaluated at K=3.
        common_metrics = calculate_metrics(
            results,
            expected,
            min(BASELINE_K, top_k),
        )

        common_k_totals["hit"] += common_metrics["hit"]
        common_k_totals["reciprocal_rank"] += (
            common_metrics["reciprocal_rank"]
        )

        status = "PASS" if metrics["hit"] else "FAIL"

        print(
            f"Question {number}: {status} "
            f"| Latency: {elapsed_ms:.2f} ms"
        )

    total = len(test_cases)

    return {
        "top_k": top_k,
        "total_questions": total,
        "hit_at_k": totals["hit"] / total,
        "mrr_at_k": totals["reciprocal_rank"] / total,
        "document_precision": totals["document_precision"] / total,
        "document_recall": totals["document_recall"] / total,
        "hit_at_3": common_k_totals["hit"] / total,
        "mrr_at_3": common_k_totals["reciprocal_rank"] / total,
        "average_latency_ms": sum(latencies_ms) / total,
        "passed": totals["hit"],
        "failed": total - totals["hit"],
    }


# ==========================================
# COMPARE RESULTS
# ==========================================

def evaluate():

    test_cases = load_test_cases()

    print(f"Loaded evaluation questions: {len(test_cases)}")

    chunks = prepare_chunks()

    if not chunks:
        raise ValueError("No knowledge-base chunks found.")

    print("\nBuilding vector store...")

    vector_store = VectorStore()
    vector_store.add_documents(chunks)

    print("Vector store ready!")

    print("\n========== BASELINE: TOP 3 ==========")

    baseline = evaluate_configuration(
        vector_store,
        test_cases,
        BASELINE_K,
    )

    print("\n========== EXPERIMENT: TOP 5 ==========")

    experiment = evaluate_configuration(
        vector_store,
        test_cases,
        EXPERIMENT_K,
    )

    comparison = {
        "experiment": "top_k_comparison",
        "baseline": baseline,
        "experiment_results": experiment,
        "changes": {
            "hit_at_3": (
                experiment["hit_at_3"] - baseline["hit_at_3"]
            ),
            "mrr_at_3": (
                experiment["mrr_at_3"] - baseline["mrr_at_3"]
            ),
            "average_latency_ms": (
                experiment["average_latency_ms"]
                - baseline["average_latency_ms"]
            ),
        },
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    print("\n====================================")
    print("    FINSIGHT AI RAG COMPARISON")
    print("====================================")

    for name, result in [
        ("BASELINE", baseline),
        ("EXPERIMENT", experiment),
    ]:

        print(f"\n{name} (Top {result['top_k']})")
        print(f"Hit@K: {result['hit_at_k']:.2%}")
        print(f"MRR@K: {result['mrr_at_k']:.3f}")
        print(f"Hit@3: {result['hit_at_3']:.2%}")
        print(f"MRR@3: {result['mrr_at_3']:.3f}")
        print(
            f"Document Precision: "
            f"{result['document_precision']:.2%}"
        )
        print(
            f"Document Recall: "
            f"{result['document_recall']:.2%}"
        )
        print(
            f"Average Latency: "
            f"{result['average_latency_ms']:.2f} ms"
        )

    print("\nCHANGES (Experiment - Baseline)")
    print(
        f"Hit@3: {comparison['changes']['hit_at_3']:+.2%}"
    )
    print(
        f"MRR@3: {comparison['changes']['mrr_at_3']:+.3f}"
    )
    print(
        "Average Latency: "
        f"{comparison['changes']['average_latency_ms']:+.2f} ms"
    )

    print(f"\nReport saved to: {REPORT_FILE}")
    print("====================================")


if __name__ == "__main__":
    evaluate()