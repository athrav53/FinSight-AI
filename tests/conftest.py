"""
Shared pytest settings for FinSight AI.

LLM observability events written during tests go to a temporary
file, never to the real logs/llm_metrics.jsonl.
"""

import pytest

from rag import observability


@pytest.fixture(autouse=True)
def isolated_llm_metrics_log(tmp_path, monkeypatch):
    monkeypatch.setattr(observability, "LOG_FILE", tmp_path / "llm_metrics.jsonl")