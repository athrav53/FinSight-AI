"""
FinSight AI - Agent Streamlit Interface

The user whose transactions the agent reads is chosen here,
in the app, and passed to run_agent(). The language model
never decides whose data to read.
"""

import streamlit as st

from agent.financial_agent import run_agent


EXAMPLE_QUESTIONS = [
    "How much did I spend in total?",
    "Show me my top 5 largest expenses.",
    "Show my spending by category.",
    "Compare my monthly spending.",
    "Where do I spend the most money?",
    "Find my Amazon transactions.",
]

ALL_USERS = "All users"


def load_user_ids():
    """User IDs available in the database (empty list if unavailable)."""

    try:
        from analytics.queries import get_users

        return [row[0] for row in get_users()]

    except Exception:
        return []


def show_execution_details(result):
    """Expandable panel: tool, scope, validation, timings, raw data."""

    with st.expander("View agent execution"):

        st.write("**Selected tool:**", result.get("tool"))
        st.write("**Data scope:**", result.get("user_scope", "all users"))

        source = result.get("answer_source")

        if source == "model":
            st.write("**Answer check:** ✅ every amount matches the database")
        elif source == "template_after_validation_failure":
            st.write(
                "**Answer check:** ⚠️ the model wrote amounts that are not in "
                "the database result, so the answer was rebuilt from the "
                "database instead:",
                ", ".join(result.get("invented_amounts", [])),
            )
        elif source == "template_model_unavailable":
            st.write(
                "**Answer check:** ℹ️ the local model was unavailable, so the "
                "answer was built directly from the database result."
            )

        if result.get("performance"):
            st.write("**Timings (seconds):**")
            st.json(result["performance"])

        if result.get("tool_result") is not None:
            st.write("**Database result:**")
            st.json(result["tool_result"])


def render_agent_ui():
    """
    Display the AI financial analytics agent.
    """

    st.title("🤖 FinSight AI Agent")

    st.write(
        "Ask questions about your financial transactions "
        "using natural language."
    )

    st.info(
        "This demo uses synthetic financial data "
        "stored in PostgreSQL. Every amount in an answer is checked "
        "against the database before it is shown."
    )

    # ---------------------------------------------
    # WHOSE DATA: chosen by the app, not the model
    # ---------------------------------------------

    user_ids = load_user_ids()

    selected = st.selectbox(
        "Answer questions about",
        [ALL_USERS] + user_ids,
        key="agent_user_scope",
        help=(
            "The agent only reads this user's transactions. "
            "The AI model cannot change this."
        ),
    )

    user_id = None if selected == ALL_USERS else selected

    # Clear the chat when the user scope changes, so earlier
    # answers about another user are not mixed in.
    if st.session_state.get("agent_scope") != selected:
        st.session_state["agent_scope"] = selected
        st.session_state["agent_messages"] = []

    # Example questions
    with st.expander("Example questions"):
        for example in EXAMPLE_QUESTIONS:
            st.markdown(f"- {example}")

    # Chat history
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    for message in st.session_state["agent_messages"]:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    question = st.chat_input(
        "Ask your financial question...",
        key="agent_question"
    )

    if question:

        st.session_state["agent_messages"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Analysing your transactions..."
            ):

                result = run_agent(question, user_id=user_id)

            status = result.get("status")

            if status == "success":

                st.markdown(result["answer"])

                show_execution_details(result)

            elif status == "unsupported":

                st.warning(result.get("answer"))

            else:

                st.error(
                    result.get(
                        "answer",
                        "Agent processing failed."
                    )
                )

        st.session_state["agent_messages"].append({
            "role": "assistant",
            "content": result.get(
                "answer",
                "Agent processing failed."
            )
        })


if __name__ == "__main__":
    render_agent_ui()