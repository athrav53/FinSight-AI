"""
FinSight AI - Document-Based RAG Chatbot

Features:
- Streamlit chatbot interface
- Local Ollama embeddings
- Gemma 3 question answering
- Document indexing
- Source citations
- Conversation history
- Sample educational documents
- No paid API required

Run:
    python -m streamlit run rag/chat_ui.py

Important:
Use public or synthetic documents only.
Do not add real bank statements or personal information.
"""

import sys
from pathlib import Path

import streamlit as st


# =====================================================
# PROJECT PATH CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================
# IMPORT EXISTING RAG MODULES
# =====================================================

from rag.document_loader import (
    load_documents,
    prepare_document_chunks,
)

from rag.vector_store import VectorStore

from rag.rag_pipeline import generate_rag_answer


# =====================================================
# DOCUMENT CONFIGURATION
# =====================================================

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "rag" / "knowledge_base"


# =====================================================
# CREATE SAMPLE DOCUMENTS
# =====================================================

def create_sample_documents():
    """
    Create synthetic educational documents.

    These documents are for testing only.
    They are not personalized financial advice.
    """

    KNOWLEDGE_BASE_DIR.mkdir(
        parents=True,
        exist_ok=True
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
"""
    }

    created_files = []

    for filename, content in sample_documents.items():

        file_path = KNOWLEDGE_BASE_DIR / filename

        if not file_path.exists():

            file_path.write_text(
                content.strip(),
                encoding="utf-8"
            )

            created_files.append(filename)

    return created_files


# =====================================================
# LOAD AND INDEX DOCUMENTS
# =====================================================

def build_knowledge_base():
    """
    Load documents, create chunks, and
    build an in-memory vector index.
    """

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            "Knowledge base folder does not exist."
        )

    documents = load_documents(
        KNOWLEDGE_BASE_DIR
    )

    if not documents:
        raise ValueError(
            "No supported documents found. "
            "Add .txt or .md files."
        )

    chunks = prepare_document_chunks(
        documents,
        chunk_size=120,
        overlap=25
    )

    if not chunks:
        raise ValueError(
            "No document chunks were generated."
        )

    store = VectorStore()

    indexed_count = store.add_documents(
        chunks
    )

    return {
        "store": store,
        "document_count": len(documents),
        "chunk_count": indexed_count
    }


# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

def initialize_chat_state():

    if "rag_vector_store" not in st.session_state:
        st.session_state["rag_vector_store"] = None

    if "rag_messages" not in st.session_state:
        st.session_state["rag_messages"] = []

    if "rag_document_count" not in st.session_state:
        st.session_state["rag_document_count"] = 0

    if "rag_chunk_count" not in st.session_state:
        st.session_state["rag_chunk_count"] = 0


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

def display_chat_history():

    for message in st.session_state["rag_messages"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander(
                    "View supporting document passages"
                ):

                    for source in message["sources"]:

                        st.markdown(
                            f"**{source['label']} — "
                            f"{source['document']}**"
                        )

                        st.caption(
                            "Similarity score: "
                            f"{source['similarity_score']}"
                        )

                        st.write(
                            source["text"]
                        )


# =====================================================
# PROCESS USER QUESTION
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
        "content": question
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
                    min_similarity=0.25
                )
                print("RAG OUTPUT MODE:", result.get("output_mode"))


            except Exception as error:

                st.error(
                    "RAG processing failed: "
                    f"{error}"
                )

                return

        st.markdown(
            result["answer"]
        )

        if result["sources"]:

            with st.expander(
                "View supporting document passages"
            ):

                for source in result["sources"]:

                    st.markdown(
                        f"**{source['label']} — "
                        f"{source['document']}**"
                    )

                    st.caption(
                        "Similarity score: "
                        f"{source['similarity_score']}"
                    )

                    st.write(
                        source["text"]
                    )

        if result["status"] != "success":

            st.info(
                f"Response status: {result['status']}"
            )

    st.session_state["rag_messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "status": result["status"]
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

    # -----------------------------------------
    # KNOWLEDGE BASE CONTROLS
    # -----------------------------------------

    st.subheader("Knowledge Base")

    st.caption(
        "Supported formats: TXT and Markdown"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📄 Create Demo Documents",
            use_container_width=True
        ):

            created = create_sample_documents()

            if created:

                st.success(
                    f"Created {len(created)} demo documents."
                )

            else:

                st.info(
                    "Demo documents already exist."
                )

    with col2:

        if st.button(
            "🔄 Build Knowledge Base",
            use_container_width=True
        ):

            with st.spinner(
                "Generating embeddings and building index..."
            ):

                try:

                    result = build_knowledge_base()

                    st.session_state["rag_vector_store"] = (
                        result["store"]
                    )

                    st.session_state["rag_document_count"] = (
                        result["document_count"]
                    )

                    st.session_state["rag_chunk_count"] = (
                        result["chunk_count"]
                    )

                    # Previous answers may refer to
                    # a different document index.
                    st.session_state["rag_messages"] = []

                    st.success(
                        "Knowledge base built successfully!"
                    )

                except Exception as error:

                    st.error(
                        f"Knowledge base error: {error}"
                    )

    with col3:

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
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

    question = st.chat_input(
        "Ask a question about your documents...",
        disabled=(store is None)
    )

    if question:

        process_question(question)


# =====================================================
# STANDALONE EXECUTION
# =====================================================

if __name__ == "__main__":

    show_rag_chat()