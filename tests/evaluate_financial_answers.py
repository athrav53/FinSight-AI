"""
FinSight AI - Financial Answer Accuracy Evaluation

Tests:
1. Database tool execution
2. LLM answer generation
3. Financial amount accuracy
4. Hallucinated amount detection
5. Response latency

Uses actual PostgreSQL data and Ollama.
"""

import json
import re
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from agent.financial_agent import run_agent


# =====================================================
# EVALUATION QUESTIONS
# =====================================================

EVALUATION_CASES = [
    {
        "question": "What is my total spending?",
        "expected_tool": "get_total_expenses"
    },
    {
        "question": "Show my 3 largest transactions with their amounts.",
        "expected_tool": "get_top_expenses"
    },
    {
        "question": "Show my spending totals by category.",
        "expected_tool": "get_category_spending"
    },
    {
        "question": "Show my monthly spending totals.",
        "expected_tool": "get_monthly_spending"
    },
    {
        "question": "Show my top 3 merchants and their spending amounts.",
        "expected_tool": "get_top_merchants"
    },
    {
        "question": "Find my Amazon transactions and their amounts.",
        "expected_tool": "search_transactions"
    }
]


# =====================================================
# NORMALIZE FINANCIAL AMOUNTS
# =====================================================

def normalize_amount(value):
    """
    Convert an amount into a two-decimal Decimal.
    """

    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError, TypeError):
        return None


def extract_answer_amounts(answer):
    """
    Extract financial amounts from an AI-generated answer.

    Supports:
    ₹27,265.15
    INR 27265.15
    Rs. 27265.15
    27265.15

    Only extracts numbers with exactly two decimal places.
    This avoids most dates and transaction counts.
    """

    pattern = r"(?<![\w.])(?:₹\s*|INR\s*|Rs\.?\s*)?(-?\d[\d,]*\.\d{2})(?![\d.])"

    matches = re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE
    )

    amounts = []

    for match in matches:

        cleaned = match.replace(",", "").strip()

        amount = normalize_amount(cleaned)

        if amount is not None:
            amounts.append(amount)

    return amounts


# =====================================================
# EXTRACT DATABASE AMOUNTS
# =====================================================

def extract_database_amounts(data):
    """
    Recursively extract values from fields that represent
    monetary amounts.

    Ignores IDs, limits, transaction counts and dates.
    """

    amounts = []

    monetary_keys = {
        "amount",
        "total",
        "total_amount",
        "total_expenses",
        "total_spending",
        "total_debit",
        "total_debits",
        "total_spent",
        "spending",
        "balance",
        "monthly_total",
        "category_total",
        "merchant_total"
    }

    def walk(value):

        if isinstance(value, dict):

            for key, item in value.items():

                if key.lower() in monetary_keys:

                    amount = normalize_amount(item)

                    if amount is not None:
                        amounts.append(amount)

                elif isinstance(item, (dict, list)):
                    walk(item)

        elif isinstance(value, list):

            for item in value:
                walk(item)

    walk(data)

    return amounts


# =====================================================
# EVALUATE ONE ANSWER
# =====================================================

def evaluate_answer(result, expected_tool):

    if result.get("status") != "success":

        return {
            "passed": False,
            "reason": "Agent execution failed.",
            "database_amounts": [],
            "answer_amounts": []
        }

    if result.get("tool") != expected_tool:

        return {
            "passed": False,
            "reason": "Incorrect tool selected.",
            "database_amounts": [],
            "answer_amounts": []
        }

    database_amounts = extract_database_amounts(
        result.get("tool_result", {})
    )

    answer_amounts = extract_answer_amounts(
        result.get("answer", "")
    )

    if not database_amounts:

        return {
            "passed": False,
            "reason": (
                "No database monetary fields detected. "
                "Check the actual tool-result schema."
            ),
            "database_amounts": [],
            "answer_amounts": [
                str(value) for value in answer_amounts
            ]
        }

    database_set = set(database_amounts)
    answer_set = set(answer_amounts)

    # Every amount mentioned by the LLM must exist in
    # the database tool result.
    invented_amounts = answer_set - database_set

    # At least one database amount must appear in the answer.
    matched_amounts = answer_set & database_set

    passed = (
        len(invented_amounts) == 0
        and len(matched_amounts) > 0
    )

    if invented_amounts:
        reason = "Answer contains amounts not found in database result."

    elif not matched_amounts:
        reason = "No matching financial amount found in answer."

    else:
        reason = "Financial amounts matched database result."

    return {
        "passed": passed,
        "reason": reason,
        "database_amounts": [
            str(value) for value in database_amounts
        ],
        "answer_amounts": [
            str(value) for value in answer_amounts
        ],
        "invented_amounts": [
            str(value) for value in invented_amounts
        ]
    }


# =====================================================
# RUN EVALUATION
# =====================================================

def evaluate_financial_answers():

    print("\n===== FINANCIAL ANSWER ACCURACY =====")

    results = []
    passed_count = 0

    start_time = time.perf_counter()

    for index, case in enumerate(EVALUATION_CASES, start=1):

        question = case["question"]

        print(f"\n[{index}/{len(EVALUATION_CASES)}] {question}")

        case_start = time.perf_counter()

        try:

            agent_result = run_agent(question)

            evaluation = evaluate_answer(
                agent_result,
                case["expected_tool"]
            )

            error = None

        except Exception as exc:

            agent_result = {
                "status": "error",
                "answer": str(exc)
            }

            evaluation = {
                "passed": False,
                "reason": "Evaluation exception.",
                "database_amounts": [],
                "answer_amounts": []
            }

            error = str(exc)

        elapsed = round(
            time.perf_counter() - case_start,
            2
        )

        if evaluation["passed"]:
            passed_count += 1

        print("Tool   :", agent_result.get("tool"))
        print("Answer :", agent_result.get("answer"))
        print("Result :", "PASS" if evaluation["passed"] else "FAIL")
        print("Reason :", evaluation["reason"])
        print("Time   :", elapsed, "seconds")

        results.append({
            "question": question,
            "expected_tool": case["expected_tool"],
            "actual_tool": agent_result.get("tool"),
            "agent_status": agent_result.get("status"),
            "answer": agent_result.get("answer"),
            "evaluation": evaluation,
            "response_time_seconds": elapsed,
            "error": error
        })

    total = len(EVALUATION_CASES)

    accuracy = round(
        (passed_count / total) * 100,
        2
    )

    total_time = round(
        time.perf_counter() - start_time,
        2
    )

    report = {
        "evaluation_name": "Financial Answer Accuracy",
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "accuracy_percent": accuracy,
        "total_time_seconds": total_time,
        "average_latency_seconds": round(total_time / total, 2),
        "results": results
    }

    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "financial_answer_evaluation.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("FINAL FINANCIAL ANSWER EVALUATION")
    print("=" * 55)

    print(f"Total Questions : {total}")
    print(f"Passed          : {passed_count}")
    print(f"Failed          : {total - passed_count}")
    print(f"Accuracy        : {accuracy}%")
    print(f"Total Time      : {total_time}s")
    print(f"Average Latency : {total_time / total:.2f}s")
    print(f"Report Saved    : {output_file}")

    return report


if __name__ == "__main__":
    evaluate_financial_answers()