"""
FinSight AI - RAG Evaluation Dashboard

Displays saved RAG evaluation results.

Features:
- Evaluation KPI cards
- Retrieval metrics
- Citation validity
- Unsupported question abstention
- Response-time analysis
- Question-wise evaluation results
- Grounding and hallucination metrics (when available)
- CSV and JSON downloads

This dashboard reads saved evaluation results.
It does not rerun the Ollama models.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# PATH CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# =====================================================
# LOAD LATEST EVALUATION RESULTS
# =====================================================

def load_latest_evaluation():

    if not RESULTS_DIR.exists():
        return None, None, "Evaluation results folder not found."

    csv_files = sorted(
        RESULTS_DIR.glob("rag_results_*.csv"),
        reverse=True
    )

    if not csv_files:
        return None, None, "No evaluation CSV files found."

    for csv_path in csv_files:

        timestamp = csv_path.stem.replace("rag_results_", "", 1)

        json_path = RESULTS_DIR / f"rag_summary_{timestamp}.json"

        if not json_path.exists():
            continue

        try:
            results_df = pd.read_csv(csv_path)

            with open(json_path, "r", encoding="utf-8") as file:
                summary = json.load(file)

            if not isinstance(summary, dict):
                continue

            return results_df, summary, None

        except (OSError, ValueError, json.JSONDecodeError):
            continue

    return None, None, "No matching evaluation files found."


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def format_percentage(value):

    if value is None or pd.isna(value):
        return "Not Evaluated"

    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "Not Evaluated"


def format_decimal(value):

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.3f}"
    except (ValueError, TypeError):
        return "N/A"


def format_seconds(value):

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.2f} s"
    except (ValueError, TypeError):
        return "N/A"


def display_value(value, default="Not Evaluated"):

    if value is None:
        return default

    if isinstance(value, (list, dict)):
        return json.dumps(value, indent=2, ensure_ascii=False)

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return str(value)


def has_grounding_results(summary):

    evaluated = summary.get("grounding_evaluated_questions")

    if evaluated is not None:
        try:
            return int(evaluated) > 0
        except (ValueError, TypeError):
            return False

    return False


# =====================================================
# MAIN DASHBOARD
# =====================================================

def show_evaluation_dashboard():

    st.title("🧪 FinSight AI — RAG Evaluation")

    st.caption(
        "Evaluation dashboard for retrieval accuracy, "
        "answer quality, citations and response time."
    )

    st.info(
        "Metrics are loaded from saved evaluation files. "
        "This page does not rerun Ollama evaluations."
    )

    results_df, summary, error = load_latest_evaluation()

    if error:

        st.warning(error)

        st.code(
            "python -u -m evaluation.rag_evaluator",
            language="powershell"
        )

        return

    # =================================================
    # EVALUATION OVERVIEW
    # =================================================

    st.subheader("📊 Evaluation Overview")

    total_questions = summary.get("total_questions", 0)

    successful = summary.get("successful_executions", 0)

    heuristic_pass = summary.get("heuristic_test_pass_rate")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Questions", total_questions)

    with col2:
        st.metric("Successful Executions", successful)

    with col3:
        st.metric(
            "Heuristic Pass Rate",
            format_percentage(heuristic_pass)
        )

    if total_questions:
        try:
            success_rate = successful / total_questions
            st.progress(min(max(float(success_rate), 0.0), 1.0))
            st.caption(
                f"Execution success rate: {success_rate * 100:.1f}%"
            )
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # =================================================
    # RETRIEVAL PERFORMANCE
    # =================================================

    st.divider()

    st.subheader("🔎 Retrieval Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Retrieval Hit@1",
        format_percentage(summary.get("retrieval_hit_at_1"))
    )

    col2.metric(
        "Retrieval Hit@3",
        format_percentage(summary.get("retrieval_hit_at_3"))
    )

    col3.metric(
        "Mean Reciprocal Rank",
        format_decimal(summary.get("mean_reciprocal_rank"))
    )

    retrieval_data = {
        "Metric": [
            "Hit@1",
            "Hit@3",
            "MRR"
        ],
        "Score": [
            summary.get("retrieval_hit_at_1"),
            summary.get("retrieval_hit_at_3"),
            summary.get("mean_reciprocal_rank")
        ]
    }

    retrieval_df = pd.DataFrame(retrieval_data)

    retrieval_df["Score"] = pd.to_numeric(
        retrieval_df["Score"],
        errors="coerce"
    )

    retrieval_df = retrieval_df.dropna(subset=["Score"])

    if not retrieval_df.empty:

        fig = px.bar(
            retrieval_df,
            x="Metric",
            y="Score",
            text_auto=".2f",
            title="Retrieval Evaluation Metrics",
            labels={"Score": "Score (0–1)"}
        )

        fig.update_yaxes(range=[0, 1.05])

        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Hit@1 and Hit@3 measure whether the expected "
        "document was retrieved. MRR rewards higher "
        "ranking of the expected document."
    )

    # =================================================
    # ANSWER QUALITY
    # =================================================

    st.divider()

    st.subheader("🧠 Answer Quality")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Citation Validity",
        format_percentage(
            summary.get("citation_validity_rate")
        )
    )

    col2.metric(
        "Unsupported Abstention",
        format_percentage(
            summary.get("unsupported_abstention_rate")
        )
    )

    col3.metric(
        "Average Response Time",
        format_seconds(
            summary.get("average_response_time_seconds")
        )
    )

    st.caption(
        "Citation validity checks whether citation references "
        "are valid, not whether the answer is factually correct. "
        "Abstention is checked using heuristics."
    )

    # =================================================
    # GROUNDING AND HALLUCINATION
    # =================================================

    st.divider()

    st.subheader("🛡️ Grounding & Hallucination Analysis")

    if has_grounding_results(summary):

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Grounding Success Rate",
            format_percentage(
                summary.get("grounding_success_rate")
            )
        )

        col2.metric(
            "Average Grounding Score",
            format_decimal(
                summary.get("average_grounding_score")
            )
        )

        col3.metric(
            "Hallucination Flag Rate",
            format_percentage(
                summary.get("hallucination_flag_rate")
            )
        )

        st.caption(
            "Grounding and hallucination metrics are "
            "estimates from an automated LLM judge. "
            "They are not proof of factual correctness."
        )

    else:

        st.info(
            "Grounding and hallucination metrics: "
            "Not Evaluated in this saved run."
        )

    # =================================================
    # RESPONSE TIME CHART
    # =================================================

    st.divider()

    st.subheader("⏱️ Response Time Analysis")

    required_columns = {
        "id",
        "response_time_seconds"
    }

    if required_columns.issubset(results_df.columns):

        chart_df = results_df[
            ["id", "response_time_seconds"]
        ].copy()

        chart_df["response_time_seconds"] = pd.to_numeric(
            chart_df["response_time_seconds"],
            errors="coerce"
        )

        chart_df = chart_df.dropna(
            subset=["response_time_seconds"]
        )

        if not chart_df.empty:

            chart_df["Question"] = (
                "Q" + chart_df["id"].astype(str)
            )

            fig = px.bar(
                chart_df,
                x="Question",
                y="response_time_seconds",
                title="Response Time by Question",
                labels={
                    "response_time_seconds":
                    "Response Time (seconds)"
                },
                text_auto=".2f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            fastest = chart_df["response_time_seconds"].min()
            slowest = chart_df["response_time_seconds"].max()
            average = chart_df["response_time_seconds"].mean()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fastest Response",
                format_seconds(fastest)
            )

            col2.metric(
                "Slowest Response",
                format_seconds(slowest)
            )

            col3.metric(
                "Average Response",
                format_seconds(average)
            )

        else:
            st.info("No valid response-time data found.")

    else:
        st.info("Response-time columns are unavailable.")

    # =================================================
    # CATEGORY ANALYSIS
    # =================================================

    if "category" in results_df.columns:

        st.divider()

        st.subheader("📂 Question Category Analysis")

        category_counts = (
            results_df["category"]
            .fillna("Unspecified")
            .value_counts()
            .reset_index()
        )

        category_counts.columns = ["Category", "Questions"]

        fig = px.bar(
            category_counts,
            x="Category",
            y="Questions",
            title="Questions by Category",
            text_auto=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # QUESTION-WISE RESULTS
    # =================================================

    st.divider()

    st.subheader("📋 Individual Test Results")

    preferred_columns = [
        "id",
        "category",
        "question",
        "answerable",
        "retrieval_hit_at_1",
        "retrieval_hit_at_3",
        "reciprocal_rank",
        "citation_valid",
        "abstention_detected",
        "grounded",
        "grounding_score",
        "hallucination_flag",
        "heuristic_test_passed",
        "response_time_seconds",
        "status"
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in results_df.columns
    ]

    if available_columns:

        st.dataframe(
            results_df[available_columns],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No question-wise results available.")

    # =================================================
    # DETAILED QUESTION REVIEW
    # =================================================

    st.divider()

    st.subheader("💬 Question & Answer Review")

    for _, row in results_df.iterrows():

        question_id = display_value(
            row.get("id"),
            default="Unknown"
        )

        question = display_value(
            row.get("question"),
            default="Question unavailable"
        )

        with st.expander(
            f"Question {question_id}: {question}"
        ):

            st.markdown("**Generated Answer**")

            st.write(
                display_value(
                    row.get("answer"),
                    default="No answer recorded."
                )
            )

            st.markdown("**Retrieved Documents**")

            st.write(
                display_value(
                    row.get("retrieved_documents"),
                    default="None"
                )
            )

            st.markdown("**Citation Validity**")

            st.write(
                display_value(
                    row.get("citation_valid"),
                    default="Not Evaluated"
                )
            )

            st.markdown("**Grounding Review**")

            grounding_status = display_value(
                row.get("grounding_status"),
                default="Not Evaluated"
            )

            st.write(f"Status: {grounding_status}")

            if grounding_status.lower() == "success":

                st.write(
                    "Grounded:",
                    display_value(row.get("grounded"))
                )

                st.write(
                    "Grounding Score:",
                    display_value(row.get("grounding_score"))
                )

                st.write(
                    "Judge Explanation:",
                    display_value(
                        row.get("grounding_explanation")
                    )
                )

            elif "grounding_review" in results_df.columns:

                st.write(
                    display_value(
                        row.get("grounding_review")
                    )
                )

    # =================================================
    # EXPORT REPORTS
    # =================================================

    st.divider()

    st.subheader("⬇️ Export Evaluation Results")

    csv_data = results_df.to_csv(
        index=False
    ).encode("utf-8")

    json_data = json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="📥 Download Evaluation CSV",
            data=csv_data,
            file_name="finsight_rag_evaluation.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        st.download_button(
            label="📥 Download Summary JSON",
            data=json_data,
            file_name="finsight_rag_summary.json",
            mime="application/json",
            use_container_width=True
        )

    # =================================================
    # LIMITATIONS
    # =================================================

    st.divider()

    st.warning(
        "Evaluation limitations: The current saved "
        "evaluation contains a small test dataset. "
        "High retrieval scores on five questions do not "
        "establish production-level reliability. "
        "Citation validity is not factual grounding, "
        "and automated grounding checks require "
        "manual verification."
    )


# =====================================================
# STANDALONE EXECUTION
# =====================================================

if __name__ == "__main__":

    st.set_page_config(
        page_title="FinSight AI - RAG Evaluation",
        page_icon="🧪",
        layout="wide"
    )

    show_evaluation_dashboard()