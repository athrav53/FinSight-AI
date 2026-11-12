"""
FinSight AI - Transaction Categorisation Evaluation

Compares rules, LLM and hybrid categorisation on a labelled
set of bank narrations (evaluation/categorization_cases.json).

Metrics per mode:
- accuracy            correct / total
- rule_coverage       share of cases a rule matched (rules / hybrid)
- llm_calls           how many cases needed the model
- fallbacks           invalid output, model down or no match
- avg_latency_ms      average time per case
- errors              every wrong prediction (for diagnosis)

Usage:
    python -m evaluation.categorization_evaluator                 # rules + hybrid + llm
    python -m evaluation.categorization_evaluator --modes rules   # no Ollama needed
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from ai.categorizer import categorize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = PROJECT_ROOT / "evaluation" / "categorization_cases.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

MODES = ("rules", "hybrid", "llm")


def load_cases(path=CASES_FILE):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"]


def evaluate_mode(cases, mode, categorize_fn=categorize, show_progress=False):
    """Run one mode over all cases and return its metrics."""

    rows = []

    if show_progress:
        print(f"\n--- Running mode: {mode} ({len(cases)} cases) ---", flush=True)

    for number, case in enumerate(cases, start=1):
        start = time.perf_counter()

        result = categorize_fn(
            case["description"], case["transaction_type"], mode=mode
        )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        correct = result.category == case["expected_category"]

        rows.append({
            "id": case["id"],
            "description": case["description"],
            "expected": case["expected_category"],
            "predicted": result.category,
            "source": result.source,
            "error": result.error,
            "correct": correct,
            "latency_ms": latency_ms,
        })

        if show_progress:
            mark = "OK " if correct else "BAD"
            print(
                f"[{number:>2}/{len(cases)}] {mark} {case['description'][:38]:<38} "
                f"-> {result.category:<15} ({result.source}, {latency_ms / 1000:.1f}s)",
                flush=True,
            )

    return summarise(mode, rows)


def summarise(mode, rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)

    return {
        "mode": mode,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "rule_coverage": round(sum(r["source"] == "rules" for r in rows) / total, 4) if total else 0.0,
        "llm_calls": sum(r["source"] == "llm" or r["error"] in {"invalid_output", "model_unavailable", "category_type_mismatch"} for r in rows),
        "fallbacks": sum(r["source"] == "fallback" for r in rows),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / total, 2) if total else 0.0,
        "errors": [
            {k: r[k] for k in ("id", "description", "expected", "predicted", "source", "error")}
            for r in rows if not r["correct"]
        ],
        "rows": rows,
    }


def print_comparison(results):
    print("\n===== TRANSACTION CATEGORISATION: COMPARISON =====\n")
    print(f"{'Mode':<8} {'Accuracy':>9} {'Correct':>9} {'Rule cov.':>10} {'LLM calls':>10} {'Fallbacks':>10} {'Avg ms':>10}")

    for r in results:
        print(
            f"{r['mode']:<8} {r['accuracy']:>8.1%} {r['correct']:>4}/{r['total']:<4} "
            f"{r['rule_coverage']:>9.1%} {r['llm_calls']:>10} {r['fallbacks']:>10} {r['avg_latency_ms']:>10.1f}"
        )

    for r in results:
        if r["errors"]:
            print(f"\nWrong in {r['mode']}:")
            for e in r["errors"]:
                print(f"  {e['id']}: {e['description'][:45]:<45} expected={e['expected']:<15} got={e['predicted']} ({e['source']})")


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"categorization_{stamp}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate transaction categorisation.")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()

    cases = load_cases()
    results = [evaluate_mode(cases, mode, show_progress=True) for mode in args.modes]

    print_comparison(results)
    print(f"\nSaved: {save_results(results)}")


if __name__ == "__main__":
    main()