
"""
FinSight AI - Document-Based RAG Chatbot

Features:
- Streamlit chatbot interface
- PostgreSQL pgvector (default)
- Manual NumPy fallback
- Persistent PostgreSQL document embeddings
- Isolated document collections
- Ollama embeddings and Gemma 3 answers
- Source citations and chat history
- Synthetic demonstration documents

Use only public or synthetic documents.
Do not upload personal bank statements.
"""

import hashlib
import sys
from pathlib import Path

import streamlit as st


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.document_loader import (
    load_documents,
    prepare_document_chunks,
)
from rag.vector_store import VectorStore
from rag.pgvector_store import PgVectorStore
from rag.rag_pipeline import generate_rag_answer


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "rag" / "knowledge_base"

PGVECTOR_OPTION = "PostgreSQL pgvector"
NUMPY_OPTION = "NumPy (in-memory)"


# =====================================================
# CREATE SAMPLE DOCUMENTS
# =====================================================

def create_sample_documents():
    """
    Create synthetic educational documents.

    Existing files are not overwritten.
    """

    KNOWLEDGE_BASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_documents = {
        "budgeting_guide.txt": """
MONTHLY BUDGETING GUIDE

A monthly budget is a plan for managing income
and expenses over a month.

A budget helps people understand where money
is going and identify spending priorities.

Common expense categories include housing,
groceries, transport, utilities, healthcare,
and entertainment.

Needs are essential expenses such as basic
housing, food, and utilities.

Wants are optional expenses such as some
entertainment purchases and nonessential shopping.

A simple budgeting process includes:

1. Review recorded income.
2. Identify recurring expenses.
3. Group expenses into categories.
4. Decide spending priorities.
5. Review the budget regularly.

A budget should be adjusted when circumstances
or expenses change.
""",
        "emergency_fund.txt": """
EMERGENCY FUND GUIDE

An emergency fund is money reserved for
unexpected necessary expenses.

Examples include urgent repairs, unexpected
medical bills, and temporary income disruption.

An emergency fund can reduce the need to
borrow money for unexpected expenses.

The appropriate amount depends on individual
circumstances, essential expenses, and financial
responsibilities.

Emergency savings should generally be accessible
when needed.

An emergency fund is different from money
reserved for planned purchases.
""",
        "spending_analysis.txt": """
SPENDING ANALYSIS GUIDE

Spending analysis involves reviewing recorded
transactions to understand expense patterns.

Transactions can be grouped into categories
such as groceries, transport, housing, shopping,
and entertainment.

Comparing expenses across months can help
identify changes in spending.

Large transactions should be reviewed in context.

A high expense in one category does not
automatically indicate poor financial management.

Missing income records do not prove that a
person has no income.

Transaction datasets may be incomplete.

Financial conclusions should reflect only
the available records.

Spending analysis can support budgeting,
expense monitoring, and financial awareness.
""",
    }

    created_files = []

    for filename, content in sample_documents.items():
        file_path = KNOWLEDGE_BASE_DIR / filename

        if not file_path.exists():
            file_path.write_text(
                content.strip(),
                encoding="utf-8",
            )
            created_files.append(filename)

    return created_files


# =====================================================
# COLLECTION IDENTIFICATION
# =====================================================

def create_collection_id(documents):
    """
    Generate a stable collection ID from document
    filenames and contents.

    Identical documents reuse the same collection.
    Changed documents receive a new collection ID.
    """

    digest = hashlib.sha256()

    for document in sorted(
        documents,
        key=lambda item: item["source"],
    ):
        source = document["source"]
        content = document["text"]

        digest.update(source.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(content.encode("utf-8"))
        digest.update(b"\x00")

    # Include the embedding model and chunking configuration.
    digest.update(b"nomic-embed-text|chunk120|overlap25")

    return "finsight_" + digest.hexdigest()[:24]


# =====================================================
# DOCUMENT LOADING
# =====================================================

def prepare_knowledge_base():
    """
    Load documents and prepare their chunks.
    """

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            "Knowledge base folder does not exist. "
            "Create demo documents first."
        )

    documents = load_documents(KNOWLEDGE_BASE_DIR)

    if not documents:
        raise ValueError(
            "No supported documents found. "
            "Add .txt or .md files."
        )

    chunks = prepare_document_chunks(
        documents,
        chunk_size=120,
        overlap=25,
    )

    if not chunks:
        raise ValueError(
            "No document chunks were generated."
        )

    return documents, chunks


# =====================================================
# BUILD KNOWLEDGE BASE
# =====================================================

def build_knowledge_base(
    storage_backend=PGVECTOR_OPTION,
):
    """
    Build the selected vector index.

    PostgreSQL failures are reported directly.
    NumPy is used only when explicitly selected.
    """

    documents, chunks = prepare_knowledge_base()

    if storage_backend == PGVECTOR_OPTION:
        collection_id = create_collection_id(documents)

        store = PgVectorStore(
            collection_id=collection_id,
        )

        # Avoid unnecessary embedding generation
        # when the complete collection already exists.
        with __import__(
            "db.connection",
            fromlist=["get_connection"],
        ).get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id
                    FROM rag_chunks
                    WHERE collection_id = %s
                    """,
                    (collection_id,),
                )
                existing_ids = {
                    row[0] for row in cursor.fetchall()
                }

        expected_ids = {
            chunk["chunk_id"] for chunk in chunks
        }

        if existing_ids == expected_ids:
            indexed_count = len(chunks)
            reused_index = True
        else:
            indexed_count = store.add_documents(chunks)
            reused_index = False

    elif storage_backend == NUMPY_OPTION:
        store = VectorStore()
        indexed_count = store.add_documents(chunks)
        collection_id = None
        reused_index = False

    else:
        raise ValueError(
            f"Unsupported storage backend: {storage_backend}"
        )

    return {
        "store": store,
        "document_count": len(documents),
        "chunk_count": indexed_count,
        "collection_id": collection_id,
        "storage_backend": storage_backend,
        "reused_index": reused_index,
    }


# =====================================================
# SESSION STATE
# =====================================================

def initialize_chat_state():
    defaults = {
        "rag_vector_store": None,
        "rag_messages": [],
        "rag_document_count": 0,
        "rag_chunk_count": 0,
        "rag_collection_id": None,
        "rag_active_backend": None,
        "rag_selected_backend": PGVECTOR_OPTION,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_active_knowledge_base():
    """
    Clear the current session's index reference.

    Does not delete PostgreSQL records.
    """

    st.session_state["rag_vector_store"] = None
    st.session_state["rag_document_count"] = 0
    st.session_state["rag_chunk_count"] = 0
    st.session_state["rag_collection_id"] = None
    st.session_state["rag_active_backend"] = None
    st.session_state["rag_messages"] = []


# =====================================================
# DISPLAY SOURCES
# =====================================================

def display_sources(sources):
    if not sources:
        return

    with st.expander(
        "View supporting document passages"
    ):
        for source in sources:
            st.markdown(
                f"**{source['label']} — "
                f"{source['document']}**"
            )

            st.caption(
                "Similarity score: "
                f"{source['similarity_score']}"
            )

            if "rerank_score" in source:
                st.caption(
                    "Rerank score: "
                    f"{source['rerank_score']}"
                )

            st.write(source["text"])


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

def display_chat_history():
    for message in st.session_state["rag_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            display_sources(message.get("sources", []))


# =====================================================
# PROCESS QUESTION
# =====================================================

def process_question(question):
    store = st.session_state["rag_vector_store"]

    if store is None:
        st.warning(
            "Please build the knowledge base first."
        )
        return

    st.session_state["rag_messages"].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching documents and generating answer..."
        ):
            try:
                result = generate_rag_answer(
                    question=question,
                    vector_store=store,
                    top_k=3,
                    min_similarity=0.25,
                )

                print(
                    "RAG OUTPUT MODE:",
                    result.get("output_mode"),
                )

            except Exception as error:
                st.error(
                    "RAG processing failed: "
                    f"{error}"
                )
                return

        st.markdown(result["answer"])

        display_sources(result["sources"])

        if result["status"] != "success":
            st.info(
                f"Response status: {result['status']}"
            )

    st.session_state["rag_messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "status": result["status"],
    })


# =====================================================
# MAIN STREAMLIT INTERFACE
# =====================================================

def show_rag_chat():
    initialize_chat_state()

    st.title("📚 FinSight AI — Document Chatbot")

    st.write(
        "Ask financial education questions "
        "using locally indexed documents."
    )

    st.info(
        "This chatbot uses local Ollama AI. "
        "Answers are based on retrieved documents, "
        "but should still be verified against "
        "the displayed source passages."
    )

    st.subheader("Knowledge Base")

    st.caption(
        "Supported formats: TXT and Markdown"
    )

    # -----------------------------------------
    # STORAGE BACKEND
    # -----------------------------------------

    selected_backend = st.selectbox(
        "Vector Storage Backend",
        options=[
            PGVECTOR_OPTION,
            NUMPY_OPTION,
        ],
        key="rag_selected_backend",
        help=(
            "PostgreSQL stores embeddings persistently. "
            "NumPy stores embeddings in memory. "
            "Fallback is manual."
        ),
    )

    if selected_backend == PGVECTOR_OPTION:
        st.caption(
            "PostgreSQL pgvector selected — "
            "persistent vector storage."
        )
    else:
        st.caption(
            "NumPy selected — "
            "in-memory vector storage."
        )

    active_backend = st.session_state[
        "rag_active_backend"
    ]

    if (
        active_backend is not None
        and active_backend != selected_backend
    ):
        st.warning(
            "You changed the storage backend. "
            "Build the knowledge base again "
            "to activate the selected backend."
        )

    # -----------------------------------------
    # KNOWLEDGE BASE CONTROLS
    # -----------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "📄 Create Demo Documents",
            use_container_width=True,
        ):
            try:
                created = create_sample_documents()

                if created:
                    st.success(
                        f"Created {len(created)} "
                        "demo documents."
                    )
                else:
                    st.info(
                        "Demo documents already exist."
                    )

            except Exception as error:
                st.error(
                    f"Document creation failed: {error}"
                )

    with col2:
        if st.button(
            "🔄 Build Knowledge Base",
            use_container_width=True,
        ):
            with st.spinner(
                "Preparing knowledge base..."
            ):
                try:
                    result = build_knowledge_base(
                        storage_backend=selected_backend,
                    )

                    st.session_state[
                        "rag_vector_store"
                    ] = result["store"]

                    st.session_state[
                        "rag_document_count"
                    ] = result["document_count"]

                    st.session_state[
                        "rag_chunk_count"
                    ] = result["chunk_count"]

                    st.session_state[
                        "rag_collection_id"
                    ] = result["collection_id"]

                    st.session_state[
                        "rag_active_backend"
                    ] = result["storage_backend"]

                    # Old answers may refer to
                    # another document index.
                    st.session_state["rag_messages"] = []

                    if result["reused_index"]:
                        st.success(
                            "Existing PostgreSQL knowledge "
                            "base loaded successfully!"
                        )
                    else:
                        st.success(
                            "Knowledge base built successfully!"
                        )

                except Exception as error:
                    st.error(
                        "Knowledge base error: "
                        f"{error}"
                    )

                    st.info(
                        "No automatic fallback was performed. "
                        "If PostgreSQL is unavailable, "
                        "select NumPy manually and rebuild."
                    )

    with col3:
        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
        ):
            st.session_state["rag_messages"] = []
            st.rerun()

    # -----------------------------------------
    # INDEX STATUS
    # -----------------------------------------

    store = st.session_state["rag_vector_store"]

    if store is not None:
        st.success(
            "Knowledge base ready | "
            f"Documents: "
            f"{st.session_state['rag_document_count']} | "
            f"Chunks: "
            f"{st.session_state['rag_chunk_count']}"
        )

        st.caption(
            "Active storage: "
            f"{st.session_state['rag_active_backend']}"
        )

        collection_id = st.session_state[
            "rag_collection_id"
        ]

        if collection_id:
            st.caption(
                f"Collection ID: {collection_id}"
            )

    else:
        st.warning(
            "Knowledge base not indexed yet."
        )

    st.divider()

    # -----------------------------------------
    # CHAT INTERFACE
    # -----------------------------------------

    st.subheader("Ask FinSight AI")

    display_chat_history()

    backend_matches = (
        st.session_state["rag_active_backend"]
        == selected_backend
    )

    question = st.chat_input(
        "Ask a question about your documents...",
        disabled=(
            store is None
            or not backend_matches
        ),
    )

    if question:
        process_question(question)


# =====================================================
# STANDALONE EXECUTION
# =====================================================

if __name__ == "__main__":
    show_rag_chat()
