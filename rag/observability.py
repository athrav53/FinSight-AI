
"""
FinSight AI - LLM Observability

Records LLM performance and failures as JSONL.

Privacy:
- Does not log prompts or generated answers
- Does not log API credentials
- Stores only operational metadata
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "llm_metrics.jsonl"

logger = logging.getLogger(__name__)


def log_llm_metrics(metrics):
    """
    Append one LLM event to the JSONL log.

    Logging failures must not interrupt
    the main RAG pipeline.
    """

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Explicit allowlist prevents accidental
        # logging of prompts, answers or secrets.
        allowed_fields = (
            "model",
            "status",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "ollama_duration_ms",
            "error_type",
        )

        event = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat()
        }

        for field in allowed_fields:
            if field in metrics:
                event[field] = metrics[field]

        with LOG_FILE.open(
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(event, ensure_ascii=False)
                + "\n"
            )

    except (OSError, TypeError, ValueError):
        logger.warning(
            "Could not write LLM observability event."
        )


if __name__ == "__main__":

    sample_metrics = {
        "model": "test-model",
        "status": "success",
        "latency_ms": 123.45,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "ollama_duration_ms": 120.0,
    }

    log_llm_metrics(sample_metrics)

    print("Observability test completed.")
    print("Log file:", LOG_FILE)
