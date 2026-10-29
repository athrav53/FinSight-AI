"""
FinSight AI - Agent Tool Selection Evaluation

Evaluates how accurately Ollama selects
the correct PostgreSQL financial analytics tool.

Does not execute database queries.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from agent.financial_agent import select_tool


# =====================================================
# EVALUATION DATASET
# =====================================================

EVALUATION_CASES = [
    # Total expenses
    {
        "question": "How much did I spend in total?",
        "expected_tool": "get_total_expenses"
    },
    {
        "question": "What is my overall expenditure?",
        "expected_tool": "get_total_expenses"
    },
    {
        "question": "Calculate my total debit amount.",
        "expected_tool": "get_total_expenses"
    },

    # Top expenses
    {
        "question": "Show my 5 largest expenses.",
        "expected_tool": "get_top_expenses"
    },
    {
        "question": "What were my biggest transactions?",
        "expected_tool": "get_top_expenses"
    },
    {
        "question": "List my top 3 highest payments.",
        "expected_tool": "get_top_expenses"
    },

    # Category spending
    {
        "question": "Show my spending by category.",
        "expected_tool": "get_category_spending"
    },
    {
        "question": "Which expense category costs the most?",
        "expected_tool": "get_category_spending"
    },
    {
        "question": "Give me a category-wise expense breakdown.",
        "expected_tool": "get_category_spending"
    },

    # Monthly spending
    {
        "question": "How much did I spend each month?",
        "expected_tool": "get_monthly_spending"
    },
    {
        "question": "Compare my spending across months.",
        "expected_tool": "get_monthly_spending"
    },
    {
        "question": "Show my monthly expenditure trend.",
        "expected_tool": "get_monthly_spending"
    },

    # Top merchants
    {
        "question": "Which merchants received the most money?",
        "expected_tool": "get_top_merchants"
    },
    {
        "question": "Show my top 5 merchants by spending.",
        "expected_tool": "get_top_merchants"
    },
    {
        "question": "Where do I spend the most money?",
        "expected_tool": "get_top_merchants"
    },

    # Transaction search
    {
        "question": "Show my Amazon transactions.",
        "expected_tool": "search_transactions"
    },
    {
        "question": "Find transactions related to Croma.",
        "expected_tool": "search_transactions"
    },
    {
        "question": "Search for payments to Pizza Hut.",
        "expected_tool": "search_transactions"
    },
]


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_agent():
    """
    Run questions through Ollama and compare
    selected tools against expected tools.
    """

    print("\n===== FINSIGHT AI AGENT EVALUATION =====")

    results = []
    correct = 0

    start_time = time.perf_counter()

    for index, case in enumerate(EVALUATION_CASES, start=1):

        question = case["question"]
        expected_tool = case["expected_tool"]

        print(f"\n[{index}/{len(EVALUATION_CASES)}] {question}")

        case_start = time.perf_counter()

        try:
            selected = select_tool(question)

            predicted_tool = selected["tool"]
            arguments = selected["arguments"]

            passed = predicted_tool == expected_tool

            error = None

        except Exception as exc:
            predicted_tool = None
            arguments = {}
            passed = False
            error = str(exc)

        elapsed = round(
            time.perf_counter() - case_start,
            2
        )

        if passed:
            correct += 1

        print(f"Expected : {expected_tool}")
        print(f"Predicted: {predicted_tool}")
        print(f"Result   : {'PASS' if passed else 'FAIL'}")
        print(f"Time     : {elapsed}s")

        if error:
            print(f"Error    : {error}")

        results.append({
            "question": question,
            "expected_tool": expected_tool,
            "predicted_tool": predicted_tool,
            "arguments": arguments,
            "passed": passed,
            "response_time_seconds": elapsed,
            "error": error
        })

    total = len(EVALUATION_CASES)

    accuracy = round(
        (correct / total) * 100,
        2
    ) if total else 0

    total_time = round(
        time.perf_counter() - start_time,
        2
    )

    report = {
        "evaluation_name": "FinSight AI Agent Tool Selection",
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy_percent": accuracy,
        "total_time_seconds": total_time,
        "results": results
    }

    # Save evaluation report
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "agent_evaluation.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)

    print(f"Total Questions : {total}")
    print(f"Correct         : {correct}")
    print(f"Incorrect       : {total - correct}")
    print(f"Accuracy        : {accuracy}%")
    print(f"Total Time      : {total_time}s")
    print(f"Report Saved    : {output_file}")

    return report


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    evaluate_agent()