
"""
FinSight AI - LLM Observability

Records every LLM call (RAG chatbot and financial agent) as one
JSON line in logs/llm_metrics.jsonl, and summarises them:
call count, failure rate, average and p95 latency, tokens and
estimated cost, overall and per feature.

Privacy:
- Does not log prompts or generated answers
- Does not log API credentials
- Stores only operational metadata (explicit allowlist)

Reliability:
- Logging failures never interrupt the user's request.
"""

import contextvars
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = Path(os.getenv("LLM_METRICS_FILE", LOG_DIR / "llm_metrics.jsonl"))

logger = logging.getLogger(__name__)

# Set by the API middleware for each HTTP request, so every LLM
# call can be linked to the request that caused it.
current_request_id = contextvars.ContextVar("current_request_id", default=None)

ALLOWED_FIELDS = (
    "feature",
    "request_id",
    "model",
    "status",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "ollama_duration_ms",
    "error_type",
)


# =====================================================
# RECORDING
# =====================================================

def log_llm_metrics(metrics):
    """
    Append one LLM event to the JSONL log.

    Logging failures must not interrupt the main pipeline.
    """

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Explicit allowlist prevents accidental logging of
        # prompts, answers or secrets.
        for field in ALLOWED_FIELDS:
            if field in metrics:
                event[field] = metrics[field]

        event.setdefault("feature", "unknown")

        if "request_id" not in event and current_request_id.get():
            event["request_id"] = current_request_id.get()

        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

        logger.info(
            "LLM call | feature=%s | status=%s | latency_ms=%s | tokens=%s | request_id=%s",
            event.get("feature"), event.get("status"), event.get("latency_ms"),
            event.get("total_tokens"), event.get("request_id"),
        )

    except (OSError, TypeError, ValueError):
        logger.warning("Could not write LLM observability event.")


# =====================================================
# SUMMARY
# =====================================================

def percentile(values, pct):
    """Nearest-rank percentile (pct 0-100). None for an empty list."""

    if not values:
        return None

    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))

    return ordered[rank - 1]


def price_per_million(kind):
    """USD per 1M tokens from .env (LLM_PRICE_INPUT_PER_1M / OUTPUT). Default 0."""

    try:
        return max(0.0, float(os.getenv(f"LLM_PRICE_{kind}_PER_1M", "0")))
    except ValueError:
        return 0.0


def summarise(events):
    """Summary for a list of events."""

    total = len(events)
    failures = sum(1 for e in events if e.get("status") != "success")

    latencies = [
        e["latency_ms"] for e in events
        if e.get("status") == "success" and isinstance(e.get("latency_ms"), (int, float))
    ]

    prompt_tokens = sum(e.get("prompt_tokens") or 0 for e in events)
    completion_tokens = sum(e.get("completion_tokens") or 0 for e in events)

    cost = (
        prompt_tokens * price_per_million("INPUT")
        + completion_tokens * price_per_million("OUTPUT")
    ) / 1_000_000

    return {
        "total_calls": total,
        "failed_calls": failures,
        "failure_rate": round(failures / total, 4) if total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": percentile(latencies, 95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": round(cost, 6),
    }


def read_events(hours=24):
    """Events from the last `hours` hours. Skips unreadable lines."""

    if not isinstance(hours, (int, float)) or not 0 < hours <= 24 * 90:
        raise ValueError("hours must be between 1 and 2160.")

    if not LOG_FILE.exists():
        return []

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    events = []

    with LOG_FILE.open(encoding="utf-8") as file:
        for line in file:
            try:
                event = json.loads(line)
                if datetime.fromisoformat(event["timestamp"]) >= since:
                    events.append(event)
            except (ValueError, KeyError, TypeError):
                continue

    return events


def summarize_llm_metrics(hours=24):
    """Overall summary plus per-feature summaries."""

    events = read_events(hours)

    by_feature = {}
    for event in events:
        by_feature.setdefault(event.get("feature", "unknown"), []).append(event)

    return {
        "window_hours": int(hours),
        **summarise(events),
        "by_feature": {
            name: summarise(items) for name, items in sorted(by_feature.items())
        },
    }


if __name__ == "__main__":
    print(json.dumps(summarize_llm_metrics(24), indent=2))