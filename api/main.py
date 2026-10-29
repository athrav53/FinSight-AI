"""
FinSight AI - FastAPI Application

Provides REST API endpoints for:
- Application health checks
- RAG-based document question answering

Existing Streamlit application remains unchanged.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.document_loader import (
    load_documents,
    prepare_document_chunks,
)
from rag.vector_store import VectorStore
from rag.rag_pipeline import generate_rag_answer


# ==========================================
# CONFIGURATION
# ==========================================

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
# INITIALIZE KNOWLEDGE BASE
# ==========================================

def initialize_knowledge_base():
    global vector_store

    documents = load_documents(
        KNOWLEDGE_BASE_DIR
    )

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
        "question answering using local RAG."
    ),
    version="1.0.0",
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