
from rag.chat_ui import (
    prepare_knowledge_base,
    create_collection_id,
)
from rag.pgvector_store import PgVectorStore
from rag.evaluate_retrieval import (
    load_test_cases,
    evaluate_configuration,
)

def main():
    documents, chunks = prepare_knowledge_base()
    collection_id = create_collection_id(documents)

    print(f"Documents: {len(documents)}")
    print(f"Local chunks: {len(chunks)}")
    print(f"Collection: {collection_id}")

    store = PgVectorStore(collection_id=collection_id)
    test_cases = load_test_cases()

    print(f"Evaluation questions: {len(test_cases)}")

    print("\n=== BASELINE TOP-3 ===")
    baseline = evaluate_configuration(store, test_cases, 3)

    print("\n=== EXPERIMENT TOP-5 ===")
    experiment = evaluate_configuration(store, test_cases, 5)

    for name, result in [
        ("Baseline", baseline),
        ("Experiment", experiment),
    ]:
        print(f"\n{name}:")
        print(f"Hit@K: {result['hit_at_k']:.2%}")
        print(f"MRR@K: {result['mrr_at_k']:.3f}")
        print(f"Hit@3: {result['hit_at_3']:.2%}")
        print(f"MRR@3: {result['mrr_at_3']:.3f}")
        print(f"Latency: {result['average_latency_ms']:.2f} ms")

if __name__ == "__main__":
    main()
