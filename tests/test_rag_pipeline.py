"""
FinSight AI - RAG pipeline unit tests

Ollama and the embedding model are replaced with fakes,
so these tests run in seconds with no AI model installed.

Run:
    python -m pytest tests/test_rag_pipeline.py -v
"""

import json
from unittest import mock

import pytest
import requests

from rag import rag_pipeline as rp
from rag.vector_store import VectorStore


CHUNKS = [
    {
        "chunk_id": "emergency_fund.txt::0",
        "source": "emergency_fund.txt",
        "chunk_index": 0,
        "similarity_score": 0.82,
        "text": (
            "An emergency fund is money reserved for unexpected "
            "expenses. Example target: ₹1,50,000."
        ),
    },
    {
        "chunk_id": "budgeting_guide.txt::0",
        "source": "budgeting_guide.txt",
        "chunk_index": 0,
        "similarity_score": 0.41,
        "text": "A monthly budget is a plan for income and expenses.",
    },
    {
        "chunk_id": "python.txt::0",
        "source": "python.txt",
        "chunk_index": 0,
        "similarity_score": 0.10,
        "text": "Python is a programming language.",
    },
]


class FakeVectorStore(VectorStore):
    """Returns fixed chunks instead of calling the embedding model."""

    def __init__(self, chunks):
        super().__init__()
        self._fixed = chunks

    def search(self, question, top_k=3):
        return [dict(chunk) for chunk in self._fixed[:top_k]]


def ollama_reply(text):
    """A fake HTTP response from Ollama containing `text`."""

    response = mock.Mock()
    response.raise_for_status = lambda: None
    response.json.return_value = {"response": text}
    return response


def ask(model_text, chunks=CHUNKS, structured=False):
    """Run the pipeline with a fake Ollama reply."""

    with mock.patch.object(rp, "STRUCTURED_OUTPUT", structured), \
         mock.patch.object(
             rp.requests,
             "post",
             return_value=ollama_reply(model_text)
         ) as post:

        result = rp.generate_rag_answer(
            "What is an emergency fund?",
            FakeVectorStore(chunks),
        )

    return result, post


# ---------------------------------------------------------
# Plain-text mode (original behaviour)
# ---------------------------------------------------------

def test_valid_answer_returns_only_cited_sources():

    result, _ = ask(
        "Money kept for unexpected expenses. [Source 1]"
    )

    assert result["status"] == "success"

    assert [
        s["document"] for s in result["sources"]
    ] == ["emergency_fund.txt"]


def test_retrieved_sources_include_everything_in_rank_order():

    result, _ = ask(
        "Money kept for unexpected expenses. [Source 1]"
    )

    ranked = [
        r["document"]
        for r in result["retrieved_sources"]
    ]

    assert ranked == [
        "emergency_fund.txt",
        "budgeting_guide.txt",
        "python.txt",
    ]


def test_amount_formats_are_normalised():

    result, _ = ask(
        "Keep Rs. 1,50,000.00 aside. [Source 1]"
    )

    assert result["status"] == "success"


def test_invented_amount_is_rejected():

    result, _ = ask(
        "Keep ₹5,000 aside. [Source 1]"
    )

    assert result["status"] == "validation_failed"

    assert result["sources"] == []


def test_missing_citation_is_rejected():

    result, _ = ask(
        "Money kept for unexpected expenses."
    )

    assert result["status"] == "validation_failed"


def test_citation_to_filtered_out_source_is_rejected():

    # Source 3 has similarity 0.10,
    # below the 0.25 threshold.

    result, _ = ask(
        "Python is a language. [Source 3]"
    )

    assert result["status"] == "validation_failed"


def test_insufficient_answer_with_curly_apostrophe():

    result, _ = ask(
        "I couldn’t find enough information in the provided "
        "documents to answer that question."
    )

    assert result["status"] == "insufficient_context"

    assert result["sources"] == []


def test_no_relevant_chunks_skips_the_model():

    result, post = ask(
        "unused",
        chunks=[CHUNKS[2]]
    )

    assert result["status"] == "insufficient_context"

    post.assert_not_called()


def test_ollama_down_returns_model_unavailable():

    with mock.patch.object(
        rp.requests,
        "post",
        side_effect=requests.ConnectionError()
    ):

        result = rp.generate_rag_answer(
            "Question?",
            FakeVectorStore(CHUNKS)
        )

    assert result["status"] == "model_unavailable"

    assert result["retrieved_sources"]


def test_empty_question_is_rejected():

    with pytest.raises(ValueError):

        rp.generate_rag_answer(
            "   ",
            FakeVectorStore(CHUNKS)
        )


# ---------------------------------------------------------
# Structured-output mode (JSON + Pydantic)
# ---------------------------------------------------------

def test_structured_answer_is_accepted():

    reply = json.dumps({
        "enough_information": True,
        "answer": (
            "Money kept for unexpected expenses. [Source 1]"
        ),
        "citations": [1],
    })

    result, post = ask(
        reply,
        structured=True
    )

    assert result["status"] == "success"

    assert result["output_mode"] == "structured"

    # Ollama received a JSON schema.
    assert "properties" in (
        post.call_args.kwargs["json"]["format"]
    )


def test_structured_citations_without_markers_are_added():

    reply = json.dumps({
        "enough_information": True,
        "answer": "Money kept for unexpected expenses.",
        "citations": [1],
    })

    result, _ = ask(
        reply,
        structured=True
    )

    assert result["status"] == "success"

    assert "[Source 1]" in result["answer"]


def test_structured_invalid_citation_is_still_rejected():

    reply = json.dumps({
        "enough_information": True,
        "answer": "Money kept for unexpected expenses.",
        "citations": [7],
    })

    result, _ = ask(
        reply,
        structured=True
    )

    assert result["status"] == "validation_failed"


def test_structured_not_enough_information():

    reply = json.dumps({
        "enough_information": False,
        "answer": "anything",
        "citations": [],
    })

    result, _ = ask(
        reply,
        structured=True
    )

    assert result["status"] == "insufficient_context"


def test_invalid_json_falls_back_to_plain_text():

    replies = [
        ollama_reply("this is not json"),
        ollama_reply(
            "Money kept for unexpected expenses. [Source 1]"
        ),
    ]

    with mock.patch.object(
        rp,
        "STRUCTURED_OUTPUT",
        True
    ), mock.patch.object(
        rp.requests,
        "post",
        side_effect=replies
    ) as post:

        result = rp.generate_rag_answer(
            "Question?",
            FakeVectorStore(CHUNKS)
        )

    assert result["status"] == "success"

    assert result["output_mode"] == "plain_fallback"

    assert post.call_count == 2

    # The retry should use plain text.
    assert "format" not in (
        post.call_args.kwargs["json"]
    )