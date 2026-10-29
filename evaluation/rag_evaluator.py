"""
FinSight AI - Advanced RAG Evaluation Pipeline

Features:
- JSON-based test cases
- Retrieval Hit@1 and Hit@3
- Mean Reciprocal Rank (MRR)
- Citation validation
- Unsupported-question abstention detection
- LLM-based grounding evaluation
- Hallucination flagging
- Category-wise evaluation
- Question-by-question progress tracking
- Response-time measurement
- CSV and JSON exports

Note:
Grounding scores are estimates from an LLM judge.
They are not guarantees of factual correctness.
"""

# ============================================================
# IMPORTS
# ============================================================

import csv
import json
import re
import time

from datetime import datetime
from pathlib import Path

from rag.document_loader import (
    load_documents,
    prepare_document_chunks,
)

from rag.vector_store import VectorStore

from rag.rag_pipeline import generate_rag_answer

from evaluation.grounding_evaluator import evaluate_grounding


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "rag" / "knowledge_base"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

TEST_CASES_PATH = EVALUATION_DIR / "test_cases.json"

RESULTS_DIR = EVALUATION_DIR / "results"


# ============================================================
# LOAD EVALUATION DATASET
# ============================================================

def load_test_cases():

    if not TEST_CASES_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {TEST_CASES_PATH}"
        )

    with open(
        TEST_CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    # Support either:
    # {"test_cases": [...]} or a direct list of test cases.
    if isinstance(dataset, dict):
        test_cases = dataset.get("test_cases", [])
    elif isinstance(dataset, list):
        test_cases = dataset
    else:
        raise ValueError(
            "Evaluation dataset must be a JSON object or list."
        )

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("No evaluation test cases found.")

    required_fields = {
        "id",
        "question",
        "answerable",
        "expected_source",
    }

    seen_ids = set()

    for test_case in test_cases:

        if not isinstance(test_case, dict):
            raise ValueError("Each test case must be a JSON object.")

        missing = required_fields - test_case.keys()

        if missing:
            raise ValueError(
                f"Test case missing required fields: {missing}"
            )

        test_id = test_case["id"]

        if test_id in seen_ids:
            raise ValueError(f"Duplicate test case ID: {test_id}")

        seen_ids.add(test_id)

        if not isinstance(test_case["question"], str):
            raise ValueError(
                f"Question must be a string for test ID {test_id}"
            )

        if not test_case["question"].strip():
            raise ValueError(
                f"Question cannot be empty for test ID {test_id}"
            )

        if not isinstance(test_case["answerable"], bool):
            raise ValueError(
                f"'answerable' must be a boolean for test ID {test_id}"
            )

        expected_source = test_case["expected_source"]

        if test_case["answerable"]:

            if not isinstance(expected_source, str):
                raise ValueError(
                    f"Answerable test {test_id} needs expected_source."
                )

            if not expected_source.strip():
                raise ValueError(
                    f"Expected source cannot be empty for {test_id}"
                )

        else:

            if expected_source is not None:
                raise ValueError(
                    f"Unsupported test {test_id} must have "
                    "expected_source=null."
                )

    return test_cases


# ============================================================
# EXTRACT RETRIEVED SOURCE NAMES
# ============================================================

def extract_source_names(sources):

    document_names = []

    for source in sources:

        if not isinstance(source, dict):
            continue

        document_name = (
            source.get("document")
            or source.get("source")
            or source.get("filename")
            or source.get("file_name")
        )

        if document_name:

            document_names.append(
                Path(str(document_name)).name
            )

    return document_names


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def calculate_retrieval_metrics(
    expected_source,
    retrieved_documents,
):

    if not expected_source:

        return {
            "retrieval_hit_at_1": None,
            "retrieval_hit_at_3": None,
            "reciprocal_rank": None,
            "expected_source_rank": None,
        }

    expected_source = Path(str(expected_source)).name

    rank = None

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):

        if document == expected_source:
            rank = index
            break

    return {
        "retrieval_hit_at_1": int(rank == 1),
        "retrieval_hit_at_3": int(
            rank is not None and rank <= 3
        ),
        "reciprocal_rank": (
            1.0 / rank if rank is not None else 0.0
        ),
        "expected_source_rank": rank,
    }


# ============================================================
# CITATION VALIDATION
# ============================================================

def check_citation_validity(answer, sources):

    citations = re.findall(
        r"\[Source\s+(\d+)\]",
        str(answer),
        flags=re.IGNORECASE,
    )

    if not citations or not sources:
        return False

    valid_numbers = set(
        range(1, len(sources) + 1)
    )

    cited_numbers = {
        int(number) for number in citations
    }

    return cited_numbers.issubset(valid_numbers)


# ============================================================
# ABSTENTION DETECTION
# ============================================================

def check_abstention(answer, status=None):

    answer_lower = str(answer).strip().lower()

    abstention_phrases = [
        "couldn't find enough information",
        "could not find enough information",
        "not enough information",
        "insufficient information",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not provided in the documents",
        "not available in the documents",
        "not mentioned in the documents",
        "not found in the provided documents",
        "do not contain this information",
        "don't contain this information",
    ]

    phrase_detected = any(
        phrase in answer_lower
        for phrase in abstention_phrases
    )

    status_detected = str(status).lower() in {
        "abstained",
        "abstention",
        "insufficient_context",
        "no_relevant_documents",
    }

    return phrase_detected or status_detected


# ============================================================
# EXTRACT SOURCE TEXTS
# ============================================================

def extract_source_texts(sources):

    source_texts = []

    for source in sources:

        if not isinstance(source, dict):
            continue

        source_text = source.get("text", "")

        if isinstance(source_text, str) and source_text.strip():
            source_texts.append(source_text.strip())

    return source_texts


# ============================================================
# GROUNDING EVALUATION
# ============================================================

def run_grounding_evaluation(
    question,
    answer,
    sources,
    execution_success,
    abstention_detected,
):

    if not execution_success:

        return {
            "grounded": None,
            "score": None,
            "explanation": "RAG execution failed.",
            "status": "skipped_execution_error",
        }

    if abstention_detected:

        return {
            "grounded": None,
            "score": None,
            "explanation": "Answer abstained.",
            "status": "skipped_abstention",
        }

    source_texts = extract_source_texts(sources)

    if not source_texts:

        return {
            "grounded": None,
            "score": None,
            "explanation": "No source texts available.",
            "status": "no_sources",
        }

    try:

        result = evaluate_grounding(
            question=question,
            answer=answer,
            source_texts=source_texts,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "Grounding evaluator must return a dictionary."
            )

        return result

    except Exception as error:

        return {
            "grounded": None,
            "score": None,
            "explanation": str(error),
            "status": "judge_error",
        }


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(test_case, vector_store):

    question = test_case["question"]

    expected_source = test_case["expected_source"]

    answerable = test_case["answerable"]

    # --------------------------------------------------------
    # STEP A: GENERATE RAG ANSWER
    # --------------------------------------------------------

    start_time = time.perf_counter()

    execution_success = True

    try:

        result = generate_rag_answer(
            question,
            vector_store,
            top_k=3,
            min_similarity=0.25,
        )

        answer = str(result.get("answer", ""))

        sources = result.get("sources", []) or []

        status = result.get("status", "")

        model = result.get("model", "")

        error_message = ""

    except Exception as error:

        execution_success = False

        answer = ""

        sources = []

        status = "execution_error"

        model = ""

        error_message = str(error)

    rag_response_time = round(
        time.perf_counter() - start_time,
        2,
    )

    # --------------------------------------------------------
    # STEP B: RETRIEVAL METRICS
    # --------------------------------------------------------

    retrieved_documents = extract_source_names(sources)

    retrieval_metrics = calculate_retrieval_metrics(
        expected_source,
        retrieved_documents,
    )

    # --------------------------------------------------------
    # STEP C: ABSTENTION DETECTION
    # --------------------------------------------------------

    abstention_detected = (
        check_abstention(answer, status)
        if execution_success
        else False
    )

    # --------------------------------------------------------
    # STEP D: CITATION VALIDATION
    # --------------------------------------------------------

    if not execution_success or abstention_detected:
        citation_valid = None
    else:
        citation_valid = check_citation_validity(
            answer,
            sources,
        )

    # --------------------------------------------------------
    # STEP E: GROUNDING EVALUATION
    # --------------------------------------------------------

    judge_start_time = time.perf_counter()

    grounding_result = run_grounding_evaluation(
        question=question,
        answer=answer,
        sources=sources,
        execution_success=execution_success,
        abstention_detected=abstention_detected,
    )

    judge_response_time = round(
        time.perf_counter() - judge_start_time,
        2,
    )

    grounded = grounding_result.get("grounded")

    grounding_score = grounding_result.get("score")

    grounding_explanation = grounding_result.get(
        "explanation",
        "",
    )

    grounding_status = grounding_result.get(
        "status",
        "unknown",
    )

    hallucination_flag = (
        not grounded
        if isinstance(grounded, bool)
        else None
    )

    # --------------------------------------------------------
    # STEP F: HEURISTIC TEST RESULT
    # --------------------------------------------------------

    if not execution_success:

        heuristic_passed = False

    elif not answerable:

        heuristic_passed = abstention_detected

    elif abstention_detected:

        heuristic_passed = False

    else:

        retrieval_passed = (
            retrieval_metrics["retrieval_hit_at_3"] == 1
        )

        citation_passed = citation_valid is True

        grounding_passed = (
            grounding_status == "success"
            and grounded is True
        )

        heuristic_passed = (
            retrieval_passed
            and citation_passed
            and grounding_passed
            and bool(answer.strip())
        )

    # --------------------------------------------------------
    # STEP G: TOTAL TIME
    # --------------------------------------------------------

    total_evaluation_time = round(
        rag_response_time + judge_response_time,
        2,
    )

    # --------------------------------------------------------
    # STEP H: SAVE QUESTION RESULT
    # --------------------------------------------------------

    return {

        "id": test_case["id"],

        "category": test_case.get(
            "category",
            "unknown",
        ),

        "question": question,

        "answerable": answerable,

        "expected_source": expected_source,

        "answer": answer,

        "retrieved_documents": ", ".join(
            retrieved_documents
        ),

        "retrieval_hit_at_1": retrieval_metrics[
            "retrieval_hit_at_1"
        ],

        "retrieval_hit_at_3": retrieval_metrics[
            "retrieval_hit_at_3"
        ],

        "reciprocal_rank": retrieval_metrics[
            "reciprocal_rank"
        ],

        "expected_source_rank": retrieval_metrics[
            "expected_source_rank"
        ],

        "citation_valid": citation_valid,

        "abstention_detected": abstention_detected,

        "grounded": grounded,

        "grounding_score": grounding_score,

        "grounding_explanation": grounding_explanation,

        "grounding_status": grounding_status,

        "hallucination_flag": hallucination_flag,

        "heuristic_test_passed": bool(
            heuristic_passed
        ),

        "execution_success": execution_success,

        "response_time_seconds": rag_response_time,

        "judge_response_time_seconds": judge_response_time,

        "total_evaluation_time_seconds": total_evaluation_time,

        "status": status,

        "model": model,

        "grounding_review": (
            "llm_judge_completed"
            if grounding_status == "success"
            else "manual_review_required"
        ),

        "error": error_message,
    }


# ============================================================
# SAFE AVERAGE
# ============================================================

def safe_average(values):

    valid_values = []

    for value in values:

        if value is None:
            continue

        try:
            valid_values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not valid_values:
        return None

    return round(
        sum(valid_values) / len(valid_values),
        4,
    )


# ============================================================
# CALCULATE EVALUATION SUMMARY
# ============================================================

def calculate_summary(results):

    total_questions = len(results)

    successful_results = [
        result
        for result in results
        if result["execution_success"]
    ]

    answerable_results = [
        result
        for result in successful_results
        if result["answerable"]
    ]

    unsupported_results = [
        result
        for result in successful_results
        if not result["answerable"]
    ]

    answered_results = [
        result
        for result in answerable_results
        if not result["abstention_detected"]
    ]

    grounded_results = [
        result
        for result in answered_results
        if result["grounding_status"] == "success"
        and isinstance(result["grounded"], bool)
    ]

    # Retrieval metrics

    retrieval_hit_at_1 = safe_average([
        result["retrieval_hit_at_1"]
        for result in answerable_results
    ])

    retrieval_hit_at_3 = safe_average([
        result["retrieval_hit_at_3"]
        for result in answerable_results
    ])

    mean_reciprocal_rank = safe_average([
        result["reciprocal_rank"]
        for result in answerable_results
    ])

    # Citation validity

    citation_validity_rate = safe_average([
        int(result["citation_valid"])
        for result in answered_results
        if result["citation_valid"] is not None
    ])

    # Abstention rate

    unsupported_abstention_rate = safe_average([
        int(result["abstention_detected"])
        for result in unsupported_results
    ])

    # Grounding metrics

    grounding_success_rate = safe_average([
        int(result["grounded"])
        for result in grounded_results
    ])

    average_grounding_score = safe_average([
        result["grounding_score"]
        for result in grounded_results
    ])

    hallucination_flag_rate = safe_average([
        int(result["hallucination_flag"])
        for result in grounded_results
        if result["hallucination_flag"] is not None
    ])

    # Overall heuristic pass rate

    heuristic_test_pass_rate = safe_average([
        int(result["heuristic_test_passed"])
        for result in results
    ])

    # Performance

    average_response_time = safe_average([
        result["response_time_seconds"]
        for result in successful_results
    ])

    average_judge_time = safe_average([
        result["judge_response_time_seconds"]
        for result in successful_results
    ])

    average_total_time = safe_average([
        result["total_evaluation_time_seconds"]
        for result in successful_results
    ])

    # Category-wise results

    category_results = {}

    categories = sorted({
        result["category"]
        for result in results
    })

    for category in categories:

        matching = [
            result
            for result in results
            if result["category"] == category
        ]

        category_results[category] = {

            "total": len(matching),

            "passed": sum(
                bool(result["heuristic_test_passed"])
                for result in matching
            ),

            "pass_rate": safe_average([
                int(result["heuristic_test_passed"])
                for result in matching
            ]),
        }

    return {

        "total_questions": total_questions,

        "successful_executions": len(
            successful_results
        ),

        "retrieval_hit_at_1": retrieval_hit_at_1,

        "retrieval_hit_at_3": retrieval_hit_at_3,

        "mean_reciprocal_rank": mean_reciprocal_rank,

        "citation_validity_rate": citation_validity_rate,

        "unsupported_abstention_rate": (
            unsupported_abstention_rate
        ),

        "grounding_success_rate": grounding_success_rate,

        "average_grounding_score": average_grounding_score,

        "hallucination_flag_rate": hallucination_flag_rate,

        "grounding_evaluated_questions": len(
            grounded_results
        ),

        "grounding_eligible_questions": len(
            answered_results
        ),

        "heuristic_test_pass_rate": heuristic_test_pass_rate,

        "average_response_time_seconds": average_response_time,

        "average_judge_time_seconds": average_judge_time,

        "average_total_evaluation_time_seconds": (
            average_total_time
        ),

        "category_results": category_results,

        "grounding": (
            "LLM judge estimates; manual review recommended"
        ),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, summary):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = RESULTS_DIR / (
        f"rag_results_{timestamp}.csv"
    )

    json_path = RESULTS_DIR / (
        f"rag_summary_{timestamp}.json"
    )

    if results:

        fieldnames = list(results[0].keys())

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(results)

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return csv_path, json_path


# ============================================================
# PRINT QUESTION RESULT
# ============================================================

def print_question_result(index, total, result):

    print("\n" + "=" * 60, flush=True)

    print(
        f"QUESTION {index}/{total}",
        flush=True,
    )

    print("=" * 60, flush=True)

    print(
        f"Question: {result['question']}",
        flush=True,
    )

    print(
        f"Category: {result['category']}",
        flush=True,
    )

    print(
        f"Execution Success: {result['execution_success']}",
        flush=True,
    )

    print(
        f"Retrieval Hit@1: {result['retrieval_hit_at_1']}",
        flush=True,
    )

    print(
        f"Retrieval Hit@3: {result['retrieval_hit_at_3']}",
        flush=True,
    )

    print(
        f"Citation Valid: {result['citation_valid']}",
        flush=True,
    )

    print(
        f"Abstention Detected: {result['abstention_detected']}",
        flush=True,
    )

    print(
        f"Grounded: {result['grounded']}",
        flush=True,
    )

    print(
        f"Grounding Score: {result['grounding_score']}",
        flush=True,
    )

    print(
        f"Grounding Status: {result['grounding_status']}",
        flush=True,
    )

    print(
        f"Hallucination Flag: {result['hallucination_flag']}",
        flush=True,
    )

    print(
        f"Heuristic Test Passed: "
        f"{result['heuristic_test_passed']}",
        flush=True,
    )

    print(
        f"RAG Response Time: "
        f"{result['response_time_seconds']} seconds",
        flush=True,
    )

    print(
        f"Judge Response Time: "
        f"{result['judge_response_time_seconds']} seconds",
        flush=True,
    )

    print(
        f"Total Evaluation Time: "
        f"{result['total_evaluation_time_seconds']} seconds",
        flush=True,
    )

    if result["error"]:

        print(
            f"Error: {result['error']}",
            flush=True,
        )

    if result["grounding_explanation"]:

        print(
            f"Judge Explanation: "
            f"{result['grounding_explanation']}",
            flush=True,
        )


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def run_evaluation():

    print("=" * 60, flush=True)

    print(
        "FINSIGHT AI - ADVANCED RAG EVALUATION",
        flush=True,
    )

    print("=" * 60, flush=True)

    # --------------------------------------------------------
    # LOAD TEST CASES
    # --------------------------------------------------------

    print(
        "\nLoading evaluation dataset...",
        flush=True,
    )

    test_cases = load_test_cases()

    total = len(test_cases)

    print(
        f"Total test questions: {total}",
        flush=True,
    )

    # --------------------------------------------------------
    # LOAD KNOWLEDGE BASE
    # --------------------------------------------------------

    print(
        "\nLoading knowledge base...",
        flush=True,
    )

    documents = load_documents(
        str(KNOWLEDGE_BASE_DIR)
    )

    print(
        f"Documents loaded: {len(documents)}",
        flush=True,
    )

    chunks = prepare_document_chunks(
        documents,
        chunk_size=120,
        overlap=25,
    )

    print(
        f"Chunks indexed: {len(chunks)}",
        flush=True,
    )

    if not chunks:
        raise ValueError(
            "No document chunks available."
        )

    # --------------------------------------------------------
    # BUILD VECTOR STORE
    # --------------------------------------------------------

    print(
        "\nBuilding vector store...",
        flush=True,
    )

    vector_store = VectorStore()

    vector_store.add_documents(chunks)

    print(
        "Vector store ready.",
        flush=True,
    )

    # --------------------------------------------------------
    # RUN 25-QUESTION EVALUATION
    # --------------------------------------------------------

    results = []

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):

        print(
            "\n" + "-" * 60,
            flush=True,
        )

        print(
            f"STARTING QUESTION {index}/{total}",
            flush=True,
        )

        print(
            f"Question: {test_case['question']}",
            flush=True,
        )

        print(
            "-" * 60,
            flush=True,
        )

        question_start_time = time.perf_counter()

        result = evaluate_question(
            test_case,
            vector_store,
        )

        question_elapsed_time = round(
            time.perf_counter() - question_start_time,
            2,
        )

        results.append(result)

        print(
            f"\nQUESTION {index} COMPLETED "
            f"in {question_elapsed_time} seconds",
            flush=True,
        )

        print_question_result(
            index,
            total,
            result,
        )

    # --------------------------------------------------------
    # CALCULATE SUMMARY
    # --------------------------------------------------------

    summary = calculate_summary(results)

    print(
        "\n" + "=" * 60,
        flush=True,
    )

    print(
        "EVALUATION SUMMARY",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}",
            flush=True,
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    csv_path, json_path = save_results(
        results,
        summary,
    )

    print(
        "\nResults saved successfully.",
        flush=True,
    )

    print(
        f"CSV: {csv_path}",
        flush=True,
    )

    print(
        f"JSON: {json_path}",
        flush=True,
    )

    return results, summary


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_evaluation()