
"""
FinSight AI - FastAPI Application

Endpoints:
- GET  /health   Service status
- POST /ask      RAG answer with source documents
- POST /agent    Financial agent (API key required, user-scoped)
- GET  /metrics  LLM observability summary (API key required)
"""

import os
import secrets
import logging

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All environment-specific settings (database, OLLAMA_BASE_URL,
# API key) come from .env / environment variables.
load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=True
)

RAG_COLLECTION_ID = os.getenv(
    "FINSIGHT_RAG_COLLECTION_ID",
    "finsight_509b0086ca64b0adf3ebd2e0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 2. IMPORT PROJECT COMPONENTS
# ============================================================

from rag.pgvector_store import PgVectorStore
from rag.rag_pipeline import generate_rag_answer
from agent.financial_agent import run_agent
from api.request_middleware import request_tracking_middleware
from rag.observability import summarize_llm_metrics


# ============================================================
# 3. GLOBAL VECTOR STORE
# ============================================================

vector_store = None


def initialize_knowledge_base():
    global vector_store

    if not RAG_COLLECTION_ID.strip():
        raise ValueError(
            "RAG collection ID cannot be empty."
        )

    vector_store = PgVectorStore(
        collection_id=RAG_COLLECTION_ID
    )

    logger.info(
        "PostgreSQL pgvector initialized. Collection: %s",
        RAG_COLLECTION_ID
    )


# ============================================================
# 4. FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store

    try:
        initialize_knowledge_base()

    except Exception:
        vector_store = None
        logger.exception(
            "Failed to initialize PostgreSQL pgvector"
        )

    yield

    vector_store = None


# ============================================================
# 5. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FinSight AI",
    description=(
        "Financial AI Assistant using FastAPI, "
        "PostgreSQL pgvector, RAG and Ollama."
    ),
    version="1.4.0",
    lifespan=lifespan
)


# Register request tracking middleware
app.middleware("http")(request_tracking_middleware)


# ============================================================
# 6. REQUEST AND RESPONSE MODELS
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str
    status: str


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "application": "FinSight AI",
        "knowledge_base_ready": vector_store is not None
    }


# ============================================================
# 8. RAG QUESTION ANSWERING
# ============================================================

@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(request: QuestionRequest):

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not available."
        )

    try:
        result = generate_rag_answer(
            question=request.question,
            vector_store=vector_store,
            top_k=3,
            min_similarity=0.25,
            use_reranker=False
        )

        # Pipeline sources are dicts; the API returns document names.
        sources = [
            source.get("document", "") if isinstance(source, dict) else str(source)
            for source in result.get("sources", [])
        ]

        return QuestionResponse(
            answer=result.get("answer", ""),
            sources=list(dict.fromkeys(sources)),
            model=result.get("model", "gemma3:4b"),
            status=result.get("status", "unknown")
        )

    except Exception:
        logger.exception(
            "RAG question answering failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process RAG question."
        )


# ============================================================
# 9. API KEY AUTHENTICATION
# ============================================================

def get_authenticated_user(
    x_api_key: str | None = Header(default=None)
) -> str:

    expected_api_key = os.getenv(
        "FINSIGHT_API_KEY"
    )

    demo_user_id = os.getenv(
        "FINSIGHT_DEMO_USER_ID",
        "demo"
    )

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured."
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        expected_api_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )

    return demo_user_id


# ============================================================
# 10. FINANCIAL AI AGENT
# ============================================================

@app.post("/agent")
def ask_financial_agent(
    request: QuestionRequest,
    user_id: str = Depends(
        get_authenticated_user
    )
):

    try:
        result = run_agent(
            question=request.question,
            user_id=user_id
        )

        return result

    except PermissionError:
        logger.warning(
            "Financial agent permission denied"
        )

        raise HTTPException(
            status_code=403,
            detail="Permission denied."
        )

    except ConnectionError:
        logger.exception(
            "Financial agent connection failed"
        )

        raise HTTPException(
            status_code=503,
            detail="Financial agent service unavailable."
        )

    except Exception:
        logger.exception(
            "Financial agent request failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Financial agent request failed."
        )


# ============================================================
# 11. LLM OBSERVABILITY METRICS
# ============================================================

@app.get("/metrics")
def llm_metrics(
    hours: int = Query(24, ge=1, le=2160),
    user_id: str = Depends(get_authenticated_user)
):
    """
    LLM calls in the last `hours` hours: count, failure rate,
    average and p95 latency, tokens and estimated cost,
    overall and per feature (rag, agent). Requires the API key.
    """

    try:
        return summarize_llm_metrics(hours)

    except Exception:
        logger.exception("Could not read LLM metrics")

        raise HTTPException(
            status_code=503,
            detail="Monitoring data is unavailable."
        )
