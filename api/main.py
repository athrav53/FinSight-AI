

"""
FinSight AI - FastAPI Application

REST API endpoints:
- GET /health: Application health check
- POST /ask: RAG document question answering
- POST /agent: Financial AI Agent

Existing Streamlit application remains unchanged.

Security:
- Financial queries require an authenticated user.
- A configured demo API key is supported for local testing.
- The client cannot choose an arbitrary user_id.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
import secrets

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field

from rag.document_loader import (
    load_documents,
    prepare_document_chunks,
)
from rag.vector_store import VectorStore
from rag.rag_pipeline import generate_rag_answer

from agent.financial_agent import run_agent


# ==========================================
# CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "rag" / "knowledge_base"

vector_store = None


# ==========================================
# REQUEST AND RESPONSE MODELS
# ==========================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )


class QuestionResponse(BaseModel):
    answer: str
    sources: list
    model: str
    status: str


# ==========================================
# API AUTHENTICATION
# ==========================================

def get_authenticated_user(
    x_api_key: str | None = Header(default=None)
) -> str:
    """
    Authenticate the request using a configured API key.

    This is suitable for a single-user local demo.
    Production multi-user authentication should use
    verified identity tokens or server-side sessions.
    """

    expected_key = os.getenv("FINSIGHT_API_KEY")
    demo_user_id = os.getenv("FINSIGHT_DEMO_USER_ID")

    if not expected_key or not demo_user_id:
        raise HTTPException(
            status_code=503,
            detail="Financial API authentication is not configured."
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return demo_user_id


# ==========================================
# INITIALIZE KNOWLEDGE BASE
# ==========================================

def initialize_knowledge_base():
    global vector_store

    documents = load_documents(KNOWLEDGE_BASE_DIR)

    if not documents:
        logger.warning(
            "No documents found in knowledge base."
        )
        return

    chunks = prepare_document_chunks(
        documents,
        chunk_size=120,
        overlap=25
    )

    if not chunks:
        logger.warning(
            "No document chunks generated."
        )
        return

    store = VectorStore()
    store.add_documents(chunks)

    vector_store = store

    logger.info(
        "Knowledge base initialized successfully."
    )


# ==========================================
# APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize_knowledge_base()

    except Exception:
        logger.exception(
            "Knowledge base initialization failed."
        )

    yield


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="FinSight AI API",
    description=(
        "REST API for financial document "
        "question answering and AI-powered "
        "financial analytics."
    ),
    version="1.2.0",
    lifespan=lifespan
)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "application": "FinSight AI",
        "knowledge_base_ready": (
            vector_store is not None
        )
    }


# ==========================================
# RAG QUESTION ANSWERING
# ==========================================

@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(request: QuestionRequest):

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge base is not available. "
                "Check Ollama and document indexing."
            )
        )

    try:
        result = generate_rag_answer(
            question=request.question,
            vector_store=vector_store,
            top_k=3,
            min_similarity=0.25
        )

        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            model=result["model"],
            status=result["status"]
        )

    except Exception:
        logger.exception(
            "RAG question answering failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question."
        )


# ==========================================
# FINANCIAL AI AGENT
# ==========================================

@app.post("/agent")
def ask_financial_agent(
    request: QuestionRequest,
    user_id: str = Depends(get_authenticated_user)
):
    """
    Execute the financial AI Agent for the
    authenticated demo user only.

    The user_id is determined by the server,
    not supplied by the request body.
    """

    try:
        result = run_agent(
            question=request.question,
            user_id=user_id
        )

        status = result.get("status")

        if status == "error":

            if result.get("error_type") == "UserScopeRequired":
                raise HTTPException(
                    status_code=403,
                    detail=result.get(
                        "answer",
                        "User scope is required."
                    )
                )

            raise HTTPException(
                status_code=503,
                detail=result.get(
                    "answer",
                    "Financial agent is unavailable."
                )
            )

        if status == "tool_error":
            raise HTTPException(
                status_code=503,
                detail=result.get(
                    "answer",
                    "Financial tool execution failed."
                )
            )

        return result

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Financial agent API failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the agent request."
        )
