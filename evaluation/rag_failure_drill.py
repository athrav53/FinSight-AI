"""
FinSight AI - RAG Failure Diagnosis Drill

Breaks the retrieval step on purpose and shows how the
evaluation set reveals the cause. Practice for "a RAG system
returns wrong answers - find out why".

Scenarios:
    baseline        chunk 120 / overlap 25, top-3, threshold 0.25
    tiny_chunks     chunk 20 / overlap 0   -> context cut into fragments
    high_threshold  threshold 0.90         -> almost everything filtered out
    missing_doc     emergency_fund.txt removed from the index
    top1            top-1 only             -> one wrong neighbour = miss

Retrieval only (embeddings via Ollama, no answer generation),
so each run takes about 1-3 minutes on CPU.

Usage:
    python -m evaluation.rag_failure_drill --scenario baseline
    python -m evaluation.rag_failure_drill --scenario missing_doc
    python -m evaluation.rag_failure_drill --chunk-size 60 --threshold 0.4
"""

import argparse
import json
from pathlib import Path

from rag.document_loader import load_documents, chunk_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = PROJECT_ROOT / "rag" / "knowledge_base"
QUESTIONS_FILE = PROJECT_ROOT / "rag" / "evaluation" / "retrieval_questions.json"

SCENARIOS = {
    "baseline":       {"chunk_size": 120, "overlap": 25, "top_k": 3, "threshold": 0.25, "drop_doc": None},
    "tiny_chunks":    {"chunk_size": 20,  "overlap": 0,  "top_k": 3, "threshold": 0.25, "drop_doc": None},
    "high_threshold": {"chunk_size": 120, "overlap": 25, "top_k": 3, "threshold": 0.90, "drop_doc": None},
    "missing_doc":    {"chunk_size": 120, "overlap": 25, "top_k": 3, "threshold": 0.25, "drop_doc": "emergency_fund.txt"},
    "top1":           {"chunk_size": 120, "overlap": 25, "top_k": 1, "threshold": 0.25, "drop_doc": None},
}


def load_questions(path=QUESTIONS_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_chunks(chunk_size, overlap, drop_doc=None, folder=KNOWLEDGE_BASE):
    chunks = []

    for document in load_documents(str(folder)):
        if drop_doc and document["source"] == drop_doc:
            continue

        for index, text in enumerate(chunk_text(document["text"], chunk_size, overlap)):
            chunks.append({
                "chunk_id": f"{document['source']}_{index}",
                "source": document["source"],
                "chunk_index": index,
                "text": text,
            })

    return chunks


def build_store(chunks):
    from rag.vector_store import VectorStore

    store = VectorStore()
    store.add_documents(chunks)
    return store


def score_question(results, expected, threshold):
    """
    results: search results (source, similarity_score), best first.

    Returns retrieval facts for one question:
    - rank: position of the expected document in the raw results (or None)
    - kept: results that pass the similarity threshold
    - answered: the expected document survives the threshold
    - abstained: nothing passes the threshold (system says "not enough info")
    """

    sources = [r["source"] for r in results]
    rank = sources.index(expected) + 1 if expected in sources else None
    kept = [r for r in results if r["similarity_score"] >= threshold]

    return {
        "rank": rank,
        "top_score": round(results[0]["similarity_score"], 3) if results else None,
        "answered": any(r["source"] == expected for r in kept),
        "abstained": not kept,
        "retrieved": sources,
    }


def summarise(rows):
    total = len(rows)

    return {
        "questions": total,
        "hit_at_k": round(sum(r["rank"] is not None for r in rows) / total, 4),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / total, 4),
        "answerable_after_threshold": round(sum(r["answered"] for r in rows) / total, 4),
        "abstained": sum(r["abstained"] for r in rows),
        "failures": [r for r in rows if not r["answered"]],
    }


def run(config, store_builder=build_store):
    chunks = build_chunks(config["chunk_size"], config["overlap"], config["drop_doc"])
    store = store_builder(chunks)

    rows = []
    for item in load_questions():
        results = store.search(item["question"], top_k=config["top_k"])
        rows.append({"question": item["question"], "expected": item["expected"],
                     **score_question(results, item["expected"], config["threshold"])})

    report = summarise(rows)
    report["chunks"] = len(chunks)
    return report


def print_report(name, config, report):
    print(f"\n===== RAG DRILL: {name} =====")
    print(f"Config: {config}")
    print(f"Chunks indexed: {report['chunks']}")
    print(f"Hit@{config['top_k']}: {report['hit_at_k']:.1%}   MRR: {report['mrr']:.3f}")
    print(f"Answerable after threshold: {report['answerable_after_threshold']:.1%}   "
          f"Abstained: {report['abstained']}/{report['questions']}")

    if report["failures"]:
        print("\nFailing questions (diagnose these):")
        for f in report["failures"][:15]:
            print(f"  - {f['question'][:50]:<50} expected={f['expected']:<28} "
                  f"rank={f['rank']} top_score={f['top_score']} got={f['retrieved'][:2]}")


def main():
    parser = argparse.ArgumentParser(description="Break RAG retrieval on purpose and diagnose it.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--drop-doc")
    args = parser.parse_args()

    config = dict(SCENARIOS[args.scenario])
    for key in ("chunk_size", "overlap", "top_k", "threshold", "drop_doc"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    print_report(args.scenario, config, run(config))


if __name__ == "__main__":
    main()