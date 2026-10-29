"""
FinSight AI - Unseen Agent Evaluation

Evaluates:
1. Tool selection accuracy
2. Argument extraction accuracy
3. Response latency
4. Error handling

Uses Ollama but does not execute PostgreSQL queries.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from agent.financial_agent import select_tool


# =====================================================
# UNSEEN EVALUATION DATASET
# =====================================================

EVALUATION_CASES = [

    # TOTAL EXPENSES
    {
        "question": "What is the sum of all my debit transactions?",
        "expected_tool": "get_total_expenses",
        "expected_arguments": {}
    },
    {
        "question": "Tell me my cumulative spending.",
        "expected_tool": "get_total_expenses",
        "expected_arguments": {}
    },
    {
        "question": "How much money have I spent altogether?",
        "expected_tool": "get_total_expenses",
        "expected_arguments": {}
    },
    {
        "question": "What is the total value of my expenses?",
        "expected_tool": "get_total_expenses",
        "expected_arguments": {}
    },
    {
        "question": "Calculate my overall debit spending.",
        "expected_tool": "get_total_expenses",
        "expected_arguments": {}
    },

    # TOP EXPENSES
    {
        "question": "Show me my 3 most expensive transactions.",
        "expected_tool": "get_top_expenses",
        "expected_arguments": {"limit": 3}
    },
    {
        "question": "List my 7 highest individual expenses.",
        "expected_tool": "get_top_expenses",
        "expected_arguments": {"limit": 7}
    },
    {
        "question": "Which 2 payments were the largest?",
        "expected_tool": "get_top_expenses",
        "expected_arguments": {"limit": 2}
    },
    {
        "question": "Display my 10 biggest debit transactions.",
        "expected_tool": "get_top_expenses",
        "expected_arguments": {"limit": 10}
    },
    {
        "question": "Show my 4 largest purchases.",
        "expected_tool": "get_top_expenses",
        "expected_arguments": {"limit": 4}
    },

    # CATEGORY SPENDING
    {
        "question": "Break down my expenses by category.",
        "expected_tool": "get_category_spending",
        "expected_arguments": {}
    },
    {
        "question": "Which spending category is the most expensive?",
        "expected_tool": "get_category_spending",
        "expected_arguments": {}
    },
    {
        "question": "Compare expenses across categories.",
        "expected_tool": "get_category_spending",
        "expected_arguments": {}
    },
    {
        "question": "How is my spending distributed across categories?",
        "expected_tool": "get_category_spending",
        "expected_arguments": {}
    },
    {
        "question": "Give me a category-level expense summary.",
        "expected_tool": "get_category_spending",
        "expected_arguments": {}
    },

    # MONTHLY SPENDING
    {
        "question": "Provide a month-by-month expense summary.",
        "expected_tool": "get_monthly_spending",
        "expected_arguments": {}
    },
    {
        "question": "Which month had the highest expenditure?",
        "expected_tool": "get_monthly_spending",
        "expected_arguments": {}
    },
    {
        "question": "How does my expenditure change every month?",
        "expected_tool": "get_monthly_spending",
        "expected_arguments": {}
    },
    {
        "question": "Show monthly totals for my debit transactions.",
        "expected_tool": "get_monthly_spending",
        "expected_arguments": {}
    },
    {
        "question": "Compare September and October spending.",
        "expected_tool": "get_monthly_spending",
        "expected_arguments": {}
    },

    # TOP MERCHANTS
    {
        "question": "Name my 3 highest-spending merchants.",
        "expected_tool": "get_top_merchants",
        "expected_arguments": {"limit": 3}
    },
    {
        "question": "List the 7 businesses receiving the most money.",
        "expected_tool": "get_top_merchants",
        "expected_arguments": {"limit": 7}
    },
    {
        "question": "Which 2 merchants account for my largest spending?",
        "expected_tool": "get_top_merchants",
        "expected_arguments": {"limit": 2}
    },
    {
        "question": "Rank my top 10 merchants by total spending.",
        "expected_tool": "get_top_merchants",
        "expected_arguments": {"limit": 10}
    },
    {
        "question": "Show my 4 most expensive merchants.",
        "expected_tool": "get_top_merchants",
        "expected_arguments": {"limit": 4}
    },

    # TRANSACTION SEARCH
    {
        "question": "Find my Netflix payments.",
        "expected_tool": "search_transactions",
        "expected_arguments": {"search_term": "Netflix"}
    },
    {
        "question": "Show transactions involving Big Bazaar.",
        "expected_tool": "search_transactions",
        "expected_arguments": {"search_term": "Big Bazaar"}
    },
    {
        "question": "Find payments associated with Spotify.",
        "expected_tool": "search_transactions",
        "expected_arguments": {"search_term": "Spotify"}
    },
    {
        "question": "Search for transactions from Croma.",
        "expected_tool": "search_transactions",
        "expected_arguments": {"search_term": "Croma"}
    },
    {
        "question": "Show me all Pizza Hut transactions.",
        "expected_tool": "search_transactions",
        "expected_arguments": {"search_term": "Pizza Hut"}
    }
]


# =====================================================
# NORMALIZE ARGUMENTS
# =====================================================

def normalize_arguments(tool_name, arguments):
    """
    Apply default limits before comparing arguments.
    """

    arguments = dict(arguments)

    if tool_name in (
        "get_top_expenses",
        "get_top_merchants"
    ):
        arguments.setdefault("limit", 5)

    if tool_name == "search_transactions":
        arguments.setdefault("limit", 10)

    return arguments


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_agent_unseen():

    print("\n===== FINSIGHT AI UNSEEN EVALUATION =====")

    results = []

    correct_tools = 0
    correct_arguments = 0
    fully_correct = 0

    start_time = time.perf_counter()

    for index, case in enumerate(EVALUATION_CASES, start=1):

        question = case["question"]
        expected_tool = case["expected_tool"]

        expected_arguments = normalize_arguments(
            expected_tool,
            case["expected_arguments"]
        )

        print(f"\n[{index}/{len(EVALUATION_CASES)}] {question}")

        case_start = time.perf_counter()

        try:
            prediction = select_tool(question)

            predicted_tool = prediction["tool"]

            predicted_arguments = normalize_arguments(
                predicted_tool,
                prediction["arguments"]
            )

            tool_correct = predicted_tool == expected_tool

            arguments_correct = (
                predicted_arguments == expected_arguments
            )

            error = None

        except Exception as exc:

            predicted_tool = None
            predicted_arguments = {}

            tool_correct = False
            arguments_correct = False

            error = str(exc)

        elapsed = round(
            time.perf_counter() - case_start,
            2
        )

        if tool_correct:
            correct_tools += 1

        if arguments_correct and tool_correct:
            correct_arguments += 1

        passed = tool_correct and arguments_correct

        if passed:
            fully_correct += 1

        print(f"Expected Tool : {expected_tool}")
        print(f"Predicted Tool: {predicted_tool}")
        print(f"Expected Args : {expected_arguments}")
        print(f"Predicted Args: {predicted_arguments}")
        print(f"Result        : {'PASS' if passed else 'FAIL'}")
        print(f"Time          : {elapsed}s")

        if error:
            print(f"Error         : {error}")

        results.append({
            "question": question,
            "expected_tool": expected_tool,
            "predicted_tool": predicted_tool,
            "expected_arguments": expected_arguments,
            "predicted_arguments": predicted_arguments,
            "tool_correct": tool_correct,
            "arguments_correct": arguments_correct,
            "passed": passed,
            "response_time_seconds": elapsed,
            "error": error
        })

    total = len(EVALUATION_CASES)

    tool_accuracy = round(
        100 * correct_tools / total, 2
    )

    argument_accuracy = round(
        100 * correct_arguments / total, 2
    )

    full_accuracy = round(
        100 * fully_correct / total, 2
    )

    total_time = round(
        time.perf_counter() - start_time, 2
    )

    report = {
        "evaluation_name": "FinSight AI Unseen Evaluation",
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "correct_tools": correct_tools,
        "correct_tool_and_arguments": fully_correct,
        "tool_accuracy_percent": tool_accuracy,
        "argument_accuracy_percent": argument_accuracy,
        "full_accuracy_percent": full_accuracy,
        "total_time_seconds": total_time,
        "average_latency_seconds": round(total_time / total, 2),
        "results": results
    }

    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "agent_unseen_evaluation.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("FINAL UNSEEN EVALUATION RESULTS")
    print("=" * 55)

    print(f"Total Questions       : {total}")
    print(f"Correct Tool Selection: {correct_tools}")
    print(f"Tool Accuracy         : {tool_accuracy}%")
    print(f"Tool + Args Accuracy  : {full_accuracy}%")
    print(f"Total Time            : {total_time}s")
    print(f"Average Latency       : {total_time / total:.2f}s")
    print(f"Report Saved          : {output_file}")

    return report


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    evaluate_agent_unseen()