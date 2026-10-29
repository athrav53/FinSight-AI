import json
from pathlib import Path

from rag.document_loader import load_documents, chunk_text
from rag.vector_store import VectorStore


# ==========================================
# PROJECT PATHS
# ==========================================

RAG_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = RAG_DIR / "knowledge_base"
EVALUATION_FILE = RAG_DIR / "evaluation" / "retrieval_questions.json"

TOP_K = 3


# ==========================================
# LOAD EVALUATION QUESTIONS
# ==========================================

def load_test_cases():

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
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
# EVALUATE RETRIEVAL
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

    hits = 0
    reciprocal_rank_sum = 0
    precision_sum = 0
    recall_sum = 0

    for number, test in enumerate(test_cases, start=1):

        question = test["question"]
        expected = test["expected"]

        results = vector_store.search(
            question,
            top_k=TOP_K,
        )

        retrieved = [
            result["source"]
            for result in results
        ]

        # First relevant result position
        rank = next(
            (
                index
                for index, source in enumerate(retrieved, start=1)
                if source == expected
            ),
            None,
        )

        hit = rank is not None

        if hit:
            hits += 1
            reciprocal_rank_sum += 1 / rank

        # Document-level precision:
        # Duplicate chunks from the same file count once.
        unique_retrieved = list(dict.fromkeys(retrieved))

        relevant_count = sum(
            source == expected
            for source in unique_retrieved
        )

        precision = (
            relevant_count / len(unique_retrieved)
            if unique_retrieved
            else 0
        )

        # One expected document per question
        recall = 1.0 if hit else 0.0

        precision_sum += precision
        recall_sum += recall

        status = "PASS" if hit else "FAIL"

        print(f"\nQuestion {number}: {status}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {retrieved}")
        print(f"Rank: {rank if rank is not None else 'Not found'}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

    # ======================================
    # FINAL RESULTS
    # ======================================

    total = len(test_cases)

    print("\n====================================")
    print("     FINSIGHT AI RAG EVALUATION")
    print("====================================")

    print(f"Total Questions: {total}")
    print(f"Hit@{TOP_K}: {hits / total:.2%}")
    print(f"MRR@{TOP_K}: {reciprocal_rank_sum / total:.3f}")
    print(f"Document Precision: {precision_sum / total:.2%}")
    print(f"Document Recall: {recall_sum / total:.2%}")
    print(f"Passed: {hits}")
    print(f"Failed: {total - hits}")

    print("====================================")


if __name__ == "__main__":
    evaluate()