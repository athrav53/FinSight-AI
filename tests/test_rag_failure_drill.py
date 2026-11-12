"""
FinSight AI - RAG failure drill tests (no Ollama needed: fake store)
"""

from evaluation import rag_failure_drill as drill


def result(source, score):
    return {"source": source, "similarity_score": score}


def test_score_question_hit_and_threshold():
    rows = [result("a.txt", 0.8), result("b.txt", 0.5)]

    scored = drill.score_question(rows, "b.txt", threshold=0.6)

    assert scored["rank"] == 2
    assert scored["answered"] is False      # retrieved, but filtered by threshold
    assert scored["abstained"] is False


def test_score_question_abstains_when_nothing_passes():
    scored = drill.score_question([result("a.txt", 0.3)], "a.txt", threshold=0.9)
    assert scored["abstained"] is True


def test_summarise_metrics():
    rows = [
        {"rank": 1, "answered": True, "abstained": False},
        {"rank": 2, "answered": True, "abstained": False},
        {"rank": None, "answered": False, "abstained": True},
        {"rank": 1, "answered": True, "abstained": False},
    ]

    report = drill.summarise(rows)

    assert report["hit_at_k"] == 0.75
    assert report["mrr"] == round((1 + 0.5 + 1) / 4, 4)
    assert report["abstained"] == 1
    assert len(report["failures"]) == 1


def test_missing_doc_scenario_removes_the_document():
    chunks = drill.build_chunks(120, 25, drop_doc="emergency_fund.txt")
    sources = {c["source"] for c in chunks}

    assert "emergency_fund.txt" not in sources
    assert len(sources) >= 10


def test_tiny_chunks_create_many_more_chunks():
    assert len(drill.build_chunks(20, 0)) > 3 * len(drill.build_chunks(120, 25))


def test_run_with_fake_store():
    class FakeStore:
        def search(self, question, top_k):
            return [result("budgeting_guide.txt", 0.9)][:top_k]

    report = drill.run(drill.SCENARIOS["baseline"], store_builder=lambda chunks: FakeStore())

    assert report["questions"] == 40
    assert 0 < report["hit_at_k"] < 1          # only budgeting questions hit