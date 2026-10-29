"""
FinSight AI
Financial Analytics + Generative AI + RAG + AI Agent

Technologies:
Python, PostgreSQL, Streamlit, Pandas, Plotly, Ollama

Use synthetic financial data during development.
"""

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import hashlib
import html
import json
import math
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db.connection import get_connection

from modules.statement_upload import show_statement_upload

from analytics.financial_insights import (
    generate_financial_insights,
)

from ai.llm_insights import (
    build_financial_payload,
    generate_llm_insights,
)

from rag.chat_ui import show_rag_chat

from evaluation.evaluation_dashboard import (
    show_evaluation_dashboard,
)

from agent.agent_ui import render_agent_ui


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FinSight AI | Financial Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# DESIGN TOKENS (adapt to light or dark theme)
# =====================================================
# Streamlit can run in light or dark mode (config.toml, or the
# user's Settings menu). Every colour below has a light and a dark
# version so text and charts stay readable in both.
# Chart colours follow a colour-blind-checked palette.


def detect_theme():
    """
    Return "dark" or "light".

    1. The theme set in .streamlit/config.toml ([theme] base = ...) wins.
    2. Otherwise use the theme the browser reports (Streamlit 1.46+).
    3. Otherwise Streamlit's default, light.
    """

    configured = st.get_option("theme.base")

    if configured in ("dark", "light"):
        return configured

    try:
        theme_type = st.context.theme.type
        if theme_type in ("dark", "light"):
            return theme_type
    except Exception:
        pass

    return "light"


THEME = detect_theme()

TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "surface_2": "#f3f2ee",
        "border": "rgba(11,11,11,0.12)",
        "text": "#0b0b0b",
        "text_2": "#52514e",
        "muted": "#6f6d68",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "accent_soft": "rgba(42,120,214,0.10)",
        "accent_mid": "rgba(42,120,214,0.16)",
    },
    "dark": {
        "surface": "#1a1a19",
        "surface_2": "#222220",
        "border": "rgba(255,255,255,0.10)",
        "text": "#f0efec",
        "text_2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "accent_soft": "rgba(57,135,229,0.10)",
        "accent_mid": "rgba(57,135,229,0.18)",
    },
}

SERIES_BY_THEME = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}

COLORS = dict(TOKENS[THEME])

# Status colours are the same in both themes.
COLORS.update({
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "warning": "#c98500" if THEME == "light" else "#fab219",
})

SERIES = SERIES_BY_THEME[THEME]

EXPENSE_COLOR = SERIES[0]
INCOME_COLOR = SERIES[1]
OTHER_COLOR = "#a3a19b" if THEME == "light" else "#5f5e5a"


def with_alpha(hex_color, alpha):
    """#3987e5 + 0.12 -> rgba(57,135,229,0.12)"""

    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    return f"rgba({red},{green},{blue},{alpha})"


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    f"""
    <style>

    .block-container {{
        padding-top: 3.2rem;
        padding-bottom: 3rem;
        max-width: 1480px;
    }}

    h1, h2, h3, h4 {{
        letter-spacing: -0.02em;
    }}

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {{
        border-right: 1px solid {COLORS["border"]};
    }}

    .st-key-finsight_navigation div[role="radiogroup"] label {{
        padding: 0.45rem 0.7rem;
        border-radius: 10px;
        margin-bottom: 2px;
        width: 100%;
        transition: background 0.15s ease;
    }}

    .st-key-finsight_navigation div[role="radiogroup"] label:hover {{
        background: {COLORS["accent_soft"]};
    }}

    .st-key-finsight_navigation div[role="radiogroup"] label:has(input:checked) {{
        background: {COLORS["accent_mid"]};
        box-shadow: inset 3px 0 0 {SERIES[0]};
    }}

    .fs-brand {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.2rem 0 0.2rem 0;
    }}

    .fs-brand-logo {{
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        font-size: 1.25rem;
        background: linear-gradient(135deg, {SERIES[0]}, {SERIES[2]});
    }}

    .fs-brand-name {{
        font-weight: 800;
        font-size: 1.15rem;
        line-height: 1.1;
        color: {COLORS["text"]};
    }}

    .fs-brand-tag {{
        font-size: 0.75rem;
        color: {COLORS["muted"]};
    }}

    .fs-side-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {COLORS["muted"]};
        font-weight: 700;
        margin: 0.4rem 0 0.2rem 0;
    }}

    /* ---------- Header ---------- */

    .fs-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        flex-wrap: wrap;
        padding: 1.1rem 1.3rem;
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        background:
            radial-gradient(circle at 0% 0%, {COLORS["accent_mid"]}, transparent 45%),
            {COLORS["surface"]};
        margin-bottom: 1.1rem;
    }}

    .fs-eyebrow {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: {COLORS["muted"]};
        font-weight: 700;
    }}

    .fs-title {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {COLORS["text"]};
        margin: 0.1rem 0 0.15rem 0;
    }}

    .fs-subtitle {{
        color: {COLORS["text_2"]};
        font-size: 0.95rem;
    }}

    .fs-pills {{
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        align-items: center;
    }}

    .fs-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        border: 1px solid {COLORS["border"]};
        background: {COLORS["surface_2"]};
        color: {COLORS["text_2"]};
        font-size: 0.8rem;
        white-space: nowrap;
    }}

    .fs-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }}

    /* ---------- Section titles ---------- */

    .fs-section {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {COLORS["text"]};
        margin: 1.2rem 0 0.1rem 0;
    }}

    .fs-section-sub {{
        color: {COLORS["muted"]};
        font-size: 0.82rem;
        margin-bottom: 0.6rem;
    }}

    /* ---------- KPI cards (st.metric) ---------- */

    [data-testid="stMetric"] {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 118px;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {{
        font-size: 0.8rem;
        color: {COLORS["text_2"]} !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }}

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {{
        font-weight: 750;
        font-size: 1.65rem;
        color: {COLORS["text"]} !important;
    }}

    /* ---------- Cards / charts ---------- */

    div[data-testid="stPlotlyChart"] {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 4px 2px;
    }}

    .fs-card {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }}

    .fs-observation {{
        display: flex;
        gap: 0.6rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid {COLORS["grid"]};
        color: {COLORS["text_2"]};
        font-size: 0.92rem;
    }}

    .fs-observation:last-child {{
        border-bottom: none;
    }}

    .fs-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
    }}

    /* ---------- Tabs ---------- */

    button[data-baseweb="tab"] p {{
        font-size: 0.95rem;
        font-weight: 600;
    }}

    .fs-footer {{
        color: {COLORS["muted"]};
        font-size: 0.78rem;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {COLORS["grid"]};
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

PAGES = [
    "📊 Dashboard",
    "📚 Document Chatbot",
    "🤖 AI Agent",
    "🧪 RAG Evaluation",
]

st.sidebar.markdown(
    """
    <div class="fs-brand">
        <div class="fs-brand-logo">💰</div>
        <div>
            <div class="fs-brand-name">FinSight AI</div>
            <div class="fs-brand-tag">Financial Intelligence Platform</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

st.sidebar.markdown(
    '<div class="fs-side-label">Navigation</div>',
    unsafe_allow_html=True,
)

selected_page = st.sidebar.radio(
    "Navigation",
    PAGES,
    key="finsight_navigation",
    label_visibility="collapsed",
)

st.sidebar.divider()


# =====================================================
# DOCUMENT CHATBOT
# =====================================================

if selected_page == "📚 Document Chatbot":

    st.sidebar.caption(
        "Powered by local Ollama and RAG"
    )

    show_rag_chat()

    st.stop()


# =====================================================
# AI AGENT
# =====================================================

if selected_page == "🤖 AI Agent":

    st.sidebar.caption(
        "Powered by local Ollama + PostgreSQL tools"
    )

    render_agent_ui()

    st.stop()


# =====================================================
# RAG EVALUATION
# =====================================================

if selected_page == "🧪 RAG Evaluation":

    st.sidebar.caption(
        "RAG Performance Monitoring"
    )

    show_evaluation_dashboard()

    st.stop()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def money(value):
    """
    Format amounts in Indian Rupees with Indian digit grouping,
    e.g. 4834629.2 -> ₹48,34,629.20
    """

    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    rupees = int(value)
    paise = int(round((value - rupees) * 100))

    if paise == 100:
        rupees += 1
        paise = 0

    digits = str(rupees)

    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []

        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]

        if head:
            groups.insert(0, head)

        digits = ",".join(groups) + "," + tail

    return f"{sign}₹{digits}.{paise:02d}"


def money_short(value):
    """Compact Indian format for chart labels: ₹950, ₹12.5K, ₹4.8L, ₹1.2Cr."""

    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1e7:
        text = f"{value / 1e7:.1f}Cr"
    elif value >= 1e5:
        text = f"{value / 1e5:.1f}L"
    elif value >= 1e3:
        text = f"{value / 1e3:.1f}K"
    else:
        text = f"{value:.0f}"

    return f"{sign}₹{text.replace('.0', '')}"


def money_kpi(value):
    """KPI card value: ₹75.43 L / ₹1.20 Cr for large amounts, full value below 1 lakh."""

    value = float(value)
    sign = "-" if value < 0 else ""
    size = abs(value)

    if size >= 1e7:
        return f"{sign}₹{size / 1e7:.2f} Cr"

    if size >= 1e5:
        return f"{sign}₹{size / 1e5:.2f} L"

    return money(value)


def percent_change(current, previous):
    """Change from previous to current as a percentage, or None."""

    if previous is None or previous == 0:
        return None

    return (current - previous) / abs(previous) * 100


def delta_text(change):
    """Text for st.metric delta, or None to hide it."""

    if change is None or math.isnan(change):
        return None

    return f"{change:+.1f}% vs previous period"


def section(title, subtitle=None):
    """Consistent section heading."""

    st.markdown(
        f'<div class="fs-section">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<div class="fs-section-sub">{html.escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )


# =====================================================
# DATABASE CONNECTION
# =====================================================

def load_database_data():
    """
    Retrieve financial transactions from PostgreSQL.

    Read-only operation.
    """

    query = """
        SELECT
            id,
            user_id,
            account_number,
            transaction_date,
            transaction_type,
            category,
            merchant,
            description,
            amount,
            balance

        FROM bank_transactions

        ORDER BY
            transaction_date DESC,
            id DESC
    """

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT current_database()"
            )

            database_name = cursor.fetchone()[0]

            cursor.execute(query)

            columns = [
                column.name
                for column in cursor.description
            ]

            records = cursor.fetchall()

    df = pd.DataFrame(
        records,
        columns=columns,
    )

    return database_name, df


# =====================================================
# DATA PREPROCESSING
# =====================================================

def prepare_transactions(raw_df):
    """
    Clean transaction data without modifying
    the original database records.
    """

    cleaned = raw_df.copy()

    if cleaned.empty:
        return cleaned

    cleaned["transaction_date"] = pd.to_datetime(
        cleaned["transaction_date"],
        errors="coerce",
    )

    cleaned["amount"] = pd.to_numeric(
        cleaned["amount"],
        errors="coerce",
    )

    cleaned["balance"] = pd.to_numeric(
        cleaned["balance"],
        errors="coerce",
    )

    cleaned["transaction_type"] = (
        cleaned["transaction_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for column in [
        "user_id",
        "category",
        "merchant",
        "description",
    ]:

        cleaned[column] = (
            cleaned[column]
            .fillna("")
            .astype(str)
        )

    cleaned = cleaned.dropna(
        subset=[
            "transaction_date",
            "amount",
        ]
    )

    cleaned = cleaned[
        cleaned["transaction_type"].isin(
            ["debit", "credit"]
        )
    ].copy()

    return cleaned


# =====================================================
# CHART STYLING
# =====================================================

def style_chart(fig, title=None, height=360, legend=True):
    """Apply the shared theme (light or dark) to a Plotly figure."""

    fig.update_layout(
        template="plotly_dark" if THEME == "dark" else "plotly_white",
        height=height,
        title=dict(
            text=title or "",
            font=dict(size=15, color=COLORS["text"]),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="system-ui, -apple-system, Segoe UI, sans-serif",
            color=COLORS["text_2"],
            size=12,
        ),
        margin=dict(l=16, r=16, t=56 if title else 16, b=16),
        hoverlabel=dict(
            bgcolor=COLORS["surface_2"],
            bordercolor=COLORS["axis"],
            font=dict(color=COLORS["text"], size=13),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(color=COLORS["text_2"]),
        ),
        showlegend=legend,
        bargap=0.28,
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=COLORS["axis"],
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["muted"]),
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        gridwidth=1,
        linecolor=COLORS["axis"],
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["muted"]),
        zeroline=False,
    )

    return fig


def round_bar_ends(fig):
    """Rounded bar ends (needs Plotly 5.19+; ignored on older versions)."""

    try:
        fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    except (ValueError, TypeError):
        pass

    return fig


def render_chart(fig, key):

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={"displayModeBar": False},
    )


def rupee_axis(fig, axis="y", values=None):
    """Show compact ₹ tick labels (₹50K, ₹1.2L) on a value axis."""

    if values is None or len(values) == 0:
        return fig

    top = max(float(max(values)), 1.0)
    step = 10 ** math.floor(math.log10(top))

    for multiplier in (1, 2, 2.5, 5, 10):
        if top / (step * multiplier) <= 5:
            step = step * multiplier
            break

    ticks = [step * i for i in range(0, int(top // step) + 2)]

    settings = dict(
        tickvals=ticks,
        ticktext=[money_short(tick) for tick in ticks],
    )

    if axis == "y":
        fig.update_yaxes(**settings)
    else:
        fig.update_xaxes(**settings)

    return fig


# =====================================================
# LOAD TRANSACTIONS
# =====================================================

database_online = True
database_name = "unavailable"

try:

    database_name, raw_df = load_database_data()

    df = prepare_transactions(raw_df)

except Exception:

    database_online = False

    df = pd.DataFrame()


# =====================================================
# DASHBOARD HEADER
# =====================================================

if database_online:
    status_dot = COLORS["good"]
    status_text = "Database online"
else:
    status_dot = COLORS["critical"]
    status_text = "Database offline"

record_text = f"{len(df):,} transactions" if database_online else "No data loaded"

st.markdown(
    f"""
    <div class="fs-header">
        <div>
            <div class="fs-eyebrow">Financial intelligence workspace</div>
            <div class="fs-title">💰 FinSight AI</div>
            <div class="fs-subtitle">
                Explore transactions, discover spending patterns,
                and generate explainable AI insights.
            </div>
        </div>
        <div class="fs-pills">
            <span class="fs-pill">
                <span class="fs-dot" style="background:{status_dot}"></span>
                {html.escape(status_text)}
            </span>
            <span class="fs-pill">🗄️ {html.escape(str(database_name))}</span>
            <span class="fs-pill">📄 {html.escape(record_text)}</span>
            <span class="fs-pill">🧪 Synthetic data</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not database_online:

    st.error(
        "Unable to load transactions. "
        "Check that PostgreSQL is running "
        "and your database configuration is correct."
    )

    with st.expander("Troubleshooting"):

        st.code(
            "Check .env settings, "
            "db/connection.py, "
            "and the bank_transactions table."
        )

    st.stop()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.markdown(
    '<div class="fs-side-label">Filters</div>',
    unsafe_allow_html=True,
)

filtered_df = df.copy()
base_df = df.copy()          # same filters except dates (for comparisons)
start_date = end_date = None

if not df.empty:

    users = sorted(
        user
        for user in df["user_id"].unique()
        if user.strip()
    )

    selected_user = st.sidebar.selectbox(
        "👤 User",
        ["All Users"] + users,
    )

    min_date = df["transaction_date"].min().date()
    max_date = df["transaction_date"].max().date()

    date_range = st.sidebar.date_input(
        "📅 Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    categories = sorted(
        category
        for category in df["category"].unique()
        if category.strip()
    )

    selected_categories = st.sidebar.multiselect(
        "🏷️ Categories",
        categories,
        placeholder="All categories",
    )

    selected_type = st.sidebar.selectbox(
        "🔁 Transaction type",
        ["All", "Expenses", "Income"],
    )

    if selected_user != "All Users":
        base_df = base_df[base_df["user_id"] == selected_user]

    if selected_categories:
        base_df = base_df[base_df["category"].isin(selected_categories)]

    if selected_type == "Expenses":
        base_df = base_df[base_df["transaction_type"] == "debit"]
    elif selected_type == "Income":
        base_df = base_df[base_df["transaction_type"] == "credit"]

    filtered_df = base_df.copy()

    if (
        isinstance(date_range, (list, tuple))
        and len(date_range) == 2
    ):

        start_date, end_date = date_range

        filtered_df = filtered_df[
            filtered_df["transaction_date"]
            .dt.date
            .between(start_date, end_date)
        ].copy()

    elif (
        isinstance(date_range, (list, tuple))
        and len(date_range) == 1
    ):

        st.sidebar.info(
            "Select an end date to apply "
            "the date filter."
        )

    st.sidebar.caption(
        f"Showing {len(filtered_df):,} "
        f"of {len(df):,} transactions"
    )

else:

    st.info(
        "No valid transactions are available yet. "
        "You can upload a synthetic statement in the Upload tab."
    )


# =====================================================
# KPI CALCULATIONS
# =====================================================

def totals(frame):
    """Income, expenses, net and savings rate for a set of rows."""

    if frame.empty:
        return 0.0, 0.0, 0.0, None

    expenses = float(frame.loc[frame["transaction_type"] == "debit", "amount"].sum())
    income = float(frame.loc[frame["transaction_type"] == "credit", "amount"].sum())
    net = income - expenses
    rate = (net / income * 100) if income > 0 else None

    return income, expenses, net, rate


debits = filtered_df[
    filtered_df["transaction_type"] == "debit"
].copy() if not filtered_df.empty else pd.DataFrame(columns=df.columns)

credits = filtered_df[
    filtered_df["transaction_type"] == "credit"
].copy() if not filtered_df.empty else pd.DataFrame(columns=df.columns)

total_income, total_spending, net_cash_flow, savings_rate = totals(filtered_df)

# Previous period of the same length, with the same user/category/type filters.
previous_df = pd.DataFrame(columns=df.columns)

if start_date is not None and end_date is not None and not base_df.empty:

    period_days = (end_date - start_date).days + 1
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=period_days - 1)

    previous_df = base_df[
        base_df["transaction_date"].dt.date.between(previous_start, previous_end)
    ]

has_previous = not previous_df.empty

prev_income, prev_spending, prev_net, prev_rate = totals(previous_df)


# =====================================================
# KPI CARDS
# =====================================================

section(
    "Financial overview",
    "Totals for the selected filters"
    + (" · change vs the previous period of the same length" if has_previous else ""),
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Transactions",
    f"{len(filtered_df):,}",
    delta=delta_text(percent_change(len(filtered_df), len(previous_df))) if has_previous else None,
    delta_color="off",
)

k2.metric(
    "Income",
    money_kpi(total_income),
    help=f"Exact: {money(total_income)}",
    delta=delta_text(percent_change(total_income, prev_income)) if has_previous else None,
)

k3.metric(
    "Expenses",
    money_kpi(total_spending),
    help=f"Exact: {money(total_spending)}",
    delta=delta_text(percent_change(total_spending, prev_spending)) if has_previous else None,
    delta_color="inverse",
)

k4.metric(
    "Net cash flow",
    money_kpi(net_cash_flow),
    help=f"Exact: {money(net_cash_flow)}",
    delta=delta_text(percent_change(net_cash_flow, prev_net)) if has_previous else None,
)

k5.metric(
    "Savings rate",
    f"{savings_rate:.1f}%" if savings_rate is not None else "—",
    help="Net cash flow as a share of income for the selected period.",
)

if credits.empty and not filtered_df.empty:

    st.info(
        "No income transactions were found "
        "in the selected records. "
        "This does not mean the user has no income."
    )

if filtered_df.empty:

    st.warning(
        "No transactions match the selected filters."
    )


# =====================================================
# CHARTS - ROW 1: TREND + CATEGORY DISTRIBUTION
# =====================================================

section("Spending analytics", "Hover over any chart for exact values")

row1_left, row1_right = st.columns([3, 2])

with row1_left:

    if not debits.empty:

        weekly = (
            debits
            .set_index("transaction_date")
            .resample("W-MON", label="left", closed="left")["amount"]
            .sum()
            .reset_index()
        )

        # A week is partial when the selected date range covers
        # fewer than 7 of its days (first/last week of the range).
        range_start = pd.Timestamp(start_date) if start_date else debits["transaction_date"].min()
        range_end = pd.Timestamp(end_date) if end_date else debits["transaction_date"].max()

        week_start = weekly["transaction_date"]
        week_end = (week_start + timedelta(days=6)).clip(upper=range_end)
        covered_days = (week_end - week_start.clip(lower=range_start)).dt.days + 1

        weekly["partial"] = covered_days < 7

        hover_labels = [
            money(amount) + (" (partial week)" if partial else "")
            for amount, partial in zip(weekly["amount"], weekly["partial"])
        ]

        fig = go.Figure(
            go.Scatter(
                x=weekly["transaction_date"],
                y=weekly["amount"],
                mode="lines+markers",
                line=dict(color=EXPENSE_COLOR, width=2.5, shape="spline", smoothing=0.6),
                marker=dict(
                    size=8,
                    color=[COLORS["surface"] if partial else EXPENSE_COLOR for partial in weekly["partial"]],
                    line=dict(width=2, color=EXPENSE_COLOR),
                ),
                fill="tozeroy",
                fillcolor=with_alpha(EXPENSE_COLOR, 0.12),
                customdata=hover_labels,
                hovertemplate="Week of %{x|%d %b %Y}<br><b>%{customdata}</b><extra></extra>",
                name="Expenses",
            )
        )

        style_chart(fig, "Weekly spending trend", height=400, legend=False)
        rupee_axis(fig, "y", weekly["amount"])
        fig.update_xaxes(tickformat="%d %b")
        fig.update_layout(hovermode="x unified")

        render_chart(fig, "weekly_spending_chart")

        if weekly["partial"].any():
            st.caption(
                "Hollow points are partial weeks at the edges of the date range, "
                "so their totals are naturally lower."
            )

    else:
        st.info("No spending data available for this chart.")

with row1_right:

    if not debits.empty:

        category_totals = (
            debits.assign(
                category=debits["category"].replace("", "Uncategorized")
            )
            .groupby("category", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
        )

        # A donut stays readable with at most 6 segments:
        # top 5 categories + "Other".
        top = category_totals.head(5)
        rest = category_totals.iloc[5:]["amount"].sum()

        if rest > 0:
            top = pd.concat(
                [top, pd.DataFrame([{"category": "Other", "amount": rest}])],
                ignore_index=True,
            )

        colors = [
            OTHER_COLOR if name == "Other" else SERIES[i]
            for i, name in enumerate(top["category"])
        ]

        fig = go.Figure(
            go.Pie(
                labels=top["category"],
                values=top["amount"],
                hole=0.62,
                sort=False,
                direction="clockwise",
                marker=dict(colors=colors, line=dict(color=COLORS["surface"], width=2)),
                textinfo="percent",
                textposition="outside",
                textfont=dict(color=COLORS["text_2"], size=12),
                customdata=[money(value) for value in top["amount"]],
                hovertemplate="<b>%{label}</b><br>%{customdata}<br>%{percent}<extra></extra>",
            )
        )

        style_chart(fig, "Spending by category", height=400)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
            margin=dict(l=24, r=24, t=56, b=16),
            annotations=[dict(
                text=f"<b>{money_short(category_totals['amount'].sum())}</b><br>"
                     f"<span style='font-size:11px;color:{COLORS['muted']}'>total spent</span>",
                showarrow=False,
                font=dict(size=18, color=COLORS["text"]),
            )],
        )

        render_chart(fig, "category_donut_chart")

        with st.expander("All categories (table)"):
            st.dataframe(
                category_totals.assign(
                    share=category_totals["amount"] / category_totals["amount"].sum()
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "category": "Category",
                    "amount": st.column_config.NumberColumn("Spent (₹)", format="%.2f"),
                    "share": st.column_config.ProgressColumn(
                        "Share", min_value=0, max_value=1, format="%.2f"
                    ),
                },
            )

    else:
        st.info("No category data available.")


# =====================================================
# CHARTS - ROW 2: TOP MERCHANTS + INCOME VS EXPENSES
# =====================================================

row2_left, row2_right = st.columns(2)

with row2_left:

    if not debits.empty:

        top_merchants = (
            debits.assign(
                merchant=debits["merchant"].replace("", "Unknown Merchant")
            )
            .groupby("merchant", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
            .head(8)
            .sort_values("amount")
        )

        fig = go.Figure(
            go.Bar(
                x=top_merchants["amount"],
                y=top_merchants["merchant"],
                orientation="h",
                marker=dict(color=EXPENSE_COLOR),
                text=[money_short(value) for value in top_merchants["amount"]],
                textposition="outside",
                textfont=dict(color=COLORS["text_2"]),
                customdata=[money(value) for value in top_merchants["amount"]],
                hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>",
                cliponaxis=False,
            )
        )

        style_chart(fig, "Top merchants by spending", legend=False)
        rupee_axis(fig, "x", top_merchants["amount"])
        fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
        fig.update_yaxes(showgrid=False)
        fig.update_layout(margin=dict(l=16, r=56, t=56, b=16))
        round_bar_ends(fig)

        render_chart(fig, "top_merchants_chart")

    else:
        st.info("No merchant spending data available.")

with row2_right:

    if not filtered_df.empty:

        monthly = filtered_df.copy()
        monthly["month"] = monthly["transaction_date"].dt.to_period("M").dt.to_timestamp()

        monthly_totals = (
            monthly
            .pivot_table(
                index="month",
                columns="transaction_type",
                values="amount",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=["credit", "debit"], fill_value=0)
            .reset_index()
        )

        labels = monthly_totals["month"].dt.strftime("%b %Y")

        fig = go.Figure()

        fig.add_bar(
            x=labels,
            y=monthly_totals["credit"],
            name="Income",
            marker=dict(color=INCOME_COLOR),
            customdata=[money(value) for value in monthly_totals["credit"]],
            hovertemplate="Income<br><b>%{customdata}</b><extra></extra>",
        )

        fig.add_bar(
            x=labels,
            y=monthly_totals["debit"],
            name="Expenses",
            marker=dict(color=EXPENSE_COLOR),
            customdata=[money(value) for value in monthly_totals["debit"]],
            hovertemplate="Expenses<br><b>%{customdata}</b><extra></extra>",
        )

        style_chart(fig, "Income vs expenses by month")
        rupee_axis(
            fig,
            "y",
            pd.concat([monthly_totals["credit"], monthly_totals["debit"]]),
        )
        fig.update_layout(barmode="group", bargroupgap=0.08, hovermode="x unified")
        round_bar_ends(fig)

        render_chart(fig, "income_vs_expenses_chart")

    else:
        st.info("No monthly data available.")


# =====================================================
# TABS: TRANSACTIONS · INSIGHTS · AI REPORT · UPLOAD
# =====================================================

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

tab_transactions, tab_insights, tab_ai, tab_upload = st.tabs([
    "📋 Transaction Explorer",
    "🧠 Financial Insights",
    "🤖 AI Spending Report",
    "📤 Upload Statement",
])


# =====================================================
# TAB 1: TRANSACTION EXPLORER
# =====================================================

with tab_transactions:

    controls = st.columns([3, 1.4, 1, 1])

    search_text = controls[0].text_input(
        "Search",
        placeholder="Search merchant, category or description...",
    )

    sort_column = controls[1].selectbox(
        "Sort by",
        ["Date", "Amount", "Merchant", "Category"],
    )

    sort_order = controls[2].selectbox(
        "Order",
        ["Descending", "Ascending"],
    )

    page_size = controls[3].selectbox(
        "Rows per page",
        [25, 50, 100],
    )

    display_df = filtered_df.copy()

    if search_text.strip() and not display_df.empty:

        searchable_columns = [
            "description",
            "merchant",
            "category",
            "transaction_type",
        ]

        mask = pd.Series(
            False,
            index=display_df.index,
        )

        for column in searchable_columns:

            mask |= (
                display_df[column]
                .astype(str)
                .str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            )

        display_df = display_df[mask].copy()

    sort_map = {
        "Date": ["transaction_date", "id"],
        "Amount": ["amount"],
        "Merchant": ["merchant"],
        "Category": ["category"],
    }

    if not display_df.empty:
        display_df = display_df.sort_values(
            sort_map[sort_column],
            ascending=(sort_order == "Ascending"),
        )

    total_rows = len(display_df)
    total_pages = max(1, math.ceil(total_rows / page_size))

    pager = st.columns([1, 3])

    page_number = pager[0].number_input(
        f"Page (of {total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    start_row = (page_number - 1) * page_size
    page_df = display_df.iloc[start_row:start_row + page_size]

    pager[1].markdown(
        f"<div style='padding-top:2.1rem;color:{COLORS['muted']}'>"
        f"Showing {start_row + 1 if total_rows else 0:,}–"
        f"{min(start_row + page_size, total_rows):,} of {total_rows:,} matching transactions"
        "</div>",
        unsafe_allow_html=True,
    )

    table_df = page_df.assign(
        signed_amount=page_df["amount"].where(
            page_df["transaction_type"] == "credit",
            -page_df["amount"],
        ),
        transaction_type=page_df["transaction_type"].str.capitalize(),
    )

    st.dataframe(
        table_df[[
            "transaction_date",
            "user_id",
            "merchant",
            "category",
            "transaction_type",
            "signed_amount",
            "balance",
            "description",
        ]],
        use_container_width=True,
        hide_index=True,
        height=min(38 * (len(table_df) + 1) + 3, 640),
        column_config={
            "transaction_date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "user_id": "User",
            "merchant": "Merchant",
            "category": "Category",
            "transaction_type": "Type",
            "signed_amount": st.column_config.NumberColumn(
                "Amount (₹)",
                format="%.2f",
                help="Negative = money out, positive = money in",
            ),
            "balance": st.column_config.NumberColumn("Balance (₹)", format="%.2f"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
    )

    st.download_button(
        f"⬇️ Download {total_rows:,} matching transactions (CSV)",
        data=(
            display_df
            .to_csv(index=False)
            .encode("utf-8")
        ),
        file_name="finsight_transactions.csv",
        mime="text/csv",
        disabled=display_df.empty,
    )


# =====================================================
# TAB 2: RULE-BASED FINANCIAL INSIGHTS
# =====================================================

insights = None

with tab_insights:

    try:

        insights = generate_financial_insights(
            filtered_df
        )

        observations = insights.get(
            "observations",
            [],
        )

        if observations:

            items = "".join(
                f'<div class="fs-observation"><span>💡</span>'
                f"<span>{html.escape(str(observation))}</span></div>"
                for observation in observations
            )

            st.markdown(
                f'<div class="fs-card"><div class="fs-section" style="margin-top:0">'
                f"Key observations</div>{items}</div>",
                unsafe_allow_html=True,
            )

        else:

            st.info(
                "No observations available "
                "for the selected transactions."
            )

        details_left, details_right = st.columns(2)

        with details_left:

            section("Highest spending categories")

            if insights.get("top_categories"):

                st.dataframe(
                    pd.DataFrame(
                        insights["top_categories"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "category": "Category",
                        "amount": st.column_config.NumberColumn("Spent (₹)", format="%.2f"),
                    },
                )

            else:

                st.caption(
                    "No category summary available."
                )

        with details_right:

            section(
                "⚠️ Unusually large expenses",
                "Statistical flags only; these do not establish fraud.",
            )

            if insights.get("large_transactions"):

                st.dataframe(
                    pd.DataFrame(
                        insights["large_transactions"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "date": "Date",
                        "amount": st.column_config.NumberColumn("Amount (₹)", format="%.2f"),
                        "category": "Category",
                    },
                )

            else:

                st.caption(
                    "No large expenses flagged."
                )

    except Exception:

        st.error(
            "Financial insights could not be generated "
            "for the selected data."
        )


# =====================================================
# TAB 3: AI SPENDING REPORT
# =====================================================

with tab_ai:

    st.markdown(
        f'<div class="fs-card">'
        f'<div class="fs-section" style="margin-top:0">🤖 AI spending report</div>'
        f'<div style="color:{COLORS["text_2"]};font-size:0.9rem">'
        "All numbers are calculated with pandas first. The local Ollama model only "
        "writes the summary, and every ₹ figure it writes is checked. If Ollama is not "
        "running, a rule-based report is shown instead."
        "</div></div>",
        unsafe_allow_html=True,
    )

    if insights is not None:

        try:

            safe_payload = build_financial_payload(
                insights
            )

            fingerprint_data = {
                "database": database_name,
                "financial_data": safe_payload,
                "transaction_ids": sorted(
                    filtered_df["id"]
                    .astype(str)
                    .tolist()
                ),
            }

            report_fingerprint = hashlib.sha256(
                json.dumps(
                    fingerprint_data,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()

            button_col, preview_col = st.columns([1, 2])

            with button_col:

                generate_clicked = st.button(
                    "✨ Generate AI Financial Report",
                    key="generate_spending_report",
                    disabled=filtered_df.empty,
                    type="primary",
                    use_container_width=True,
                )

            with preview_col:

                with st.expander(
                    "Preview the aggregated data sent to the model"
                ):

                    st.json(
                        safe_payload
                    )

            if generate_clicked:

                with st.spinner(
                    "Analysing your transactions and writing the report..."
                ):

                    result = generate_llm_insights(
                        insights
                    )

                st.session_state[
                    "spending_report"
                ] = result

                st.session_state[
                    "spending_report_hash"
                ] = report_fingerprint

            saved_report = st.session_state.get(
                "spending_report"
            )

            saved_hash = st.session_state.get(
                "spending_report_hash"
            )

            if (
                saved_report
                and saved_hash == report_fingerprint
            ):

                from_ai = saved_report.get("source") == "local_ollama"

                badge_color = COLORS["good"] if from_ai else COLORS["warning"]
                badge_text = "Local AI (Ollama)" if from_ai else "Rule-based fallback"

                with st.container(border=True):

                    st.markdown(
                        f'<span class="fs-badge" style="background:{badge_color}22;'
                        f'color:{badge_color};border:1px solid {badge_color}55">'
                        f"● {badge_text}</span>",
                        unsafe_allow_html=True,
                    )

                    if not from_ai:

                        st.caption(
                            "Ollama was unavailable or its response "
                            "failed validation, so a rule-based report is shown."
                        )

                    st.markdown(
                        saved_report.get(
                            "answer",
                            "No report text returned.",
                        )
                    )

                    details = f"Source: {saved_report.get('model', 'unknown')}"

                    if from_ai:
                        details += (
                            f" · Input tokens: {saved_report.get('input_tokens', 'N/A')}"
                            f" · Output tokens: {saved_report.get('output_tokens', 'N/A')}"
                        )

                    st.caption(details)

            elif saved_report:

                st.info(
                    "Your selected transactions have changed. "
                    "Generate a new report for the current data."
                )

        except Exception:

            st.error(
                "AI report preparation failed. "
                "Please check the financial insights "
                "and Ollama configuration."
            )

    else:

        st.info(
            "Financial insights are needed before an AI report "
            "can be generated."
        )


# =====================================================
# TAB 4: BANK STATEMENT UPLOAD
# =====================================================

with tab_upload:

    show_statement_upload()


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    '<div class="fs-footer">'
    "FinSight AI · Python · PostgreSQL · Streamlit · Pandas · Plotly · Ollama · RAG"
    "<br>Financial insights are informational only. "
    "No personalized investment advice is provided."
    "</div>",
    unsafe_allow_html=True,
)