import csv
from datetime import datetime
import json
import html
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from chatbot.fraud_assistant import (
    build_incident_context,
    build_dashboard_context,
)

from learning.continual_learner import learning_status


# ============================================================
# PAGE CONFIG
# ============================================================



def visible_metric(label, value, delta=None, delta_color="normal", help=None, label_visibility="visible", border=False):
    """Theme-independent metric card with explicit readable colors."""
    delta_html = ""
    if delta is not None:
        cmap = {"normal": "#2563eb", "off": "#64748b", "inverse": "#dc2626"}
        dcolor = cmap.get(str(delta_color), "#2563eb")
        delta_html = f'<div style="margin-top:.25rem;font-size:.75rem;font-weight:700;color:{dcolor} !important;-webkit-text-fill-color:{dcolor} !important;">{delta}</div>'
    help_html = f'<div style="font-size:.7rem;color:#64748b !important;-webkit-text-fill-color:#64748b !important;margin-top:.3rem;">{help}</div>' if help else ""
    st.markdown(
        f"""
        <div style="width:100%;box-sizing:border-box;min-height:96px;padding:1rem 1.05rem;background:#ffffff !important;border:1px solid #dfe5ee !important;border-radius:12px;box-shadow:0 1px 3px rgba(15,23,42,.05);overflow:hidden;">
            <div style="font-size:.78rem;font-weight:700;color:#475569 !important;-webkit-text-fill-color:#475569 !important;line-height:1.25;">{label}</div>
            <div style="font-size:1.72rem;font-weight:800;color:#111827 !important;-webkit-text-fill-color:#111827 !important;line-height:1.15;margin-top:.45rem;white-space:normal;word-break:break-word;">{value}</div>
            {delta_html}
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(
    page_title="Merchant Risk Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = ROOT_DIR / "reports"
DATA_DIR = ROOT_DIR / "data"

INCIDENT_FILE = REPORTS_DIR / "incident_summary.json"
TRANSACTION_FILE = DATA_DIR / "raw" / "transactions.csv"
RESPONSE_FILE = REPORTS_DIR / "merchant_response_actions.csv"
TRANSACTION_RISK_REPORT_FILE = REPORTS_DIR / "transaction_risk_reports.csv"
METRICS_FILE = REPORTS_DIR / "strong_optimized_metrics.json"
AUDIT_FILE = REPORTS_DIR / "prediction_audit.jsonl"

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

# ============================================================
# OLLAMA CONFIGURATION
# ============================================================
#
# The dashboard talks directly to the local Ollama server.
# When Streamlit runs inside Docker, host.docker.internal
# points back to the host machine where Ollama is running.
#
# Override these with environment variables if required:
#   OLLAMA_URL=http://host.docker.internal:11434
#   OLLAMA_MODEL=qwen3.5:4b
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3.5:4b",
)


def _build_ollama_prompt(
    question,
    context,
    conversation_history=None,
):
    """Build the grounded prompt shared by normal and streaming Ollama calls."""

    history = conversation_history or []

    history_text = []
    for message in history[-8:]:
        role = str(
            message.get("role", "user")
        ).upper()
        content = str(
            message.get("content", "")
        ).strip()

        if content:
            history_text.append(
                f"{role}: {content}"
            )

    history_block = (
        "\n".join(history_text)
        if history_text
        else "No previous conversation."
    )

    return f"""
You are the AI Fraud Assistant for Merchant Risk Sentinel.

Your job is to analyze merchant fraud intelligence using ONLY
the evidence supplied in the dashboard context.

Be concise, analytical and operational.
Do not invent transaction IDs, customers, risk scores, causes,
or evidence that is not present in the context.

When recommending an action:
1. State the recommended action.
2. Explain the evidence supporting it.
3. Mention uncertainty when the available evidence is incomplete.

DASHBOARD / INCIDENT CONTEXT:
{context}

RECENT CONVERSATION:
{history_block}

USER QUESTION:
{question}

ANSWER:
"""


def ask_ollama(
    question,
    context,
    conversation_history=None,
):
    """Send the fraud-assistant request directly to Ollama and return the complete answer."""

    prompt = _build_ollama_prompt(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
        },
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    answer = str(
        data.get("response", "")
    ).strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer


def stream_ollama(
    question,
    context,
    conversation_history=None,
):
    """
    Stream the local Ollama response token-by-token.

    Yields text chunks as they arrive so Streamlit can render the
    assistant response live instead of waiting for the full answer.
    """

    prompt = _build_ollama_prompt(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "think": False,
        },
        stream=True,
        timeout=(10, 300),
    )

    response.raise_for_status()

    chunks = []
    received_content = False

    try:
        for raw_line in response.iter_lines(
            decode_unicode=True
        ):
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                # Ollama normally returns one JSON object per line.
                # Ignore malformed/empty transport fragments safely.
                continue

            chunk = str(
                data.get("response", "")
            )

            if chunk:
                received_content = True
                chunks.append(chunk)
                yield chunk

            if data.get("done", False):
                break

    finally:
        response.close()

    if not received_content:
        raise RuntimeError(
            "Ollama returned an empty streaming response."
        )


# ============================================================
# SESSION STATE
# ============================================================

if "selected_incident_id" not in st.session_state:
    st.session_state.selected_incident_id = None

if "live_prediction" not in st.session_state:
    st.session_state.live_prediction = None

if "fraud_chat_messages" not in st.session_state:
    st.session_state.fraud_chat_messages = []

if "fraud_pending_question" not in st.session_state:
    st.session_state.fraud_pending_question = None


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f5f7fa;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }

    /* Keep Streamlit's native text colors intact.
       Broad p/span/label rules can make native widgets invisible. */

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #6b7280;
    }

    .risk-banner {
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        background: #111827;
        margin-bottom: 1.5rem;
    }

    .risk-banner-label {
        color: #d1d5db !important;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .risk-banner-value {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.25rem;
        min-height: 120px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .incident-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        min-height: 250px;
    }

    .incident-card:hover {
        border-color: #9ca3af;
    }

    .incident-id {
        font-size: 1.15rem;
        font-weight: 800;
        color: #111827;
        margin-top: 0.6rem;
    }

    .incident-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #374151;
        margin-top: 0.25rem;
        margin-bottom: 0.8rem;
    }

    .incident-stat {
        font-size: 0.85rem;
        color: #4b5563;
        margin: 0.3rem 0;
    }

    .critical-badge,
    .high-badge,
    .medium-badge,
    .low-badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .critical-badge {
        background: #fee2e2;
        color: #991b1b !important;
    }

    .high-badge {
        background: #ffedd5;
        color: #9a3412 !important;
    }

    .medium-badge {
        background: #fef3c7;
        color: #92400e !important;
    }

    .low-badge {
        background: #dcfce7;
        color: #166534 !important;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }

    .investigation-header {
        background: #111827;
        border-radius: 18px;
        padding: 1.7rem;
        margin-bottom: 1.5rem;
    }

    .signal-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .action-box {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    /* ======================================================
       STREAMLIT BUTTON VISIBILITY
       Explicitly style the button AND every nested element.
       Streamlit renders button labels inside nested p/span/div
       elements, so the global text rule above can otherwise
       make the label nearly invisible on dark buttons.
       ====================================================== */

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    button[kind="secondary"],
    button[kind="primary"] {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #9ca3af !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10) !important;
    }

    div[data-testid="stButton"] > button *,
    div[data-testid="stDownloadButton"] > button *,
    button[kind="secondary"] *,
    button[kind="primary"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover {
        background: #f3f4f6 !important;
        color: #111827 !important;
        border-color: #6b7280 !important;
    }

    div[data-testid="stButton"] > button:focus,
    div[data-testid="stButton"] > button:active,
    div[data-testid="stDownloadButton"] > button:focus,
    div[data-testid="stDownloadButton"] > button:active {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* Force Markdown-rendered labels inside buttons to remain visible. */
    div[data-testid="stButton"] button p,
    div[data-testid="stButton"] button span,
    div[data-testid="stButton"] button div,
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span,
    div[data-testid="stDownloadButton"] button div {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Streamlit 1.62 base button selectors. */
    div[data-testid="stBaseButton-secondary"],
    div[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"] {
        color: #111827 !important;
        background: #ffffff !important;
        border-color: #9ca3af !important;
        opacity: 1 !important;
    }

    div[data-testid="stBaseButton-secondary"] *,
    div[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-secondary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    div[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background: #111827 !important;
        color: #ffffff !important;
    }

    div[data-testid="stBaseButton-primary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /*
       Never allow disabled/hover/focus states to hide button labels.
    */
    button:disabled,
    button:hover,
    button:focus,
    button:active {
        opacity: 1 !important;
    }

    /* ======================================================
       FINAL TEXT VISIBILITY OVERRIDES
       Streamlit 1.62 can apply theme text variables to
       rendered HTML/native elements. Explicitly force all
       dashboard content to a readable color.
       ====================================================== */

    .main .block-container .section-card,
    .main .block-container .section-card *,
    .main .block-container .signal-box,
    .main .block-container .signal-box *,
    .main .block-container .action-box,
    .main .block-container .action-box *,
    .main .block-container .incident-card,
    .main .block-container .incident-card *,
    .main .block-container .hero,
    .main .block-container .hero *,
    .main .block-container .incident-stat,
    .main .block-container .incident-stat *,
    .main .block-container .incident-id,
    .main .block-container .incident-title {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    .main .block-container .hero-subtitle,
    .main .block-container .metric-label {
        color: #6b7280 !important;
        -webkit-text-fill-color: #6b7280 !important;
    }

    .main .block-container .risk-banner,
    .main .block-container .risk-banner * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .main .block-container .risk-banner-label {
        color: #d1d5db !important;
        -webkit-text-fill-color: #d1d5db !important;
    }

    /* Native Streamlit buttons: target every known 1.62 DOM
       variant and force both background and text explicitly. */
    .main .block-container .stButton button,
    .main .block-container .stDownloadButton button,
    .main .block-container div[data-testid="stButton"] button,
    .main .block-container div[data-testid="stDownloadButton"] button,
    .main .block-container button[data-testid*="BaseButton"],
    .main .block-container button[kind="secondary"] {
        background-color: #ffffff !important;
        background-image: none !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        border: 1px solid #9ca3af !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
    }

    .main .block-container .stButton button *,
    .main .block-container .stDownloadButton button *,
    .main .block-container div[data-testid="stButton"] button *,
    .main .block-container div[data-testid="stDownloadButton"] button *,
    .main .block-container button[data-testid*="BaseButton"] *,
    .main .block-container button[kind="secondary"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        fill: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
    }

    .main .block-container .stButton button:hover,
    .main .block-container .stDownloadButton button:hover,
    .main .block-container div[data-testid="stButton"] button:hover,
    .main .block-container div[data-testid="stDownloadButton"] button:hover {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    .main .block-container .stButton button:focus,
    .main .block-container .stButton button:active,
    .main .block-container .stDownloadButton button:focus,
    .main .block-container .stDownloadButton button:active {
        background-color: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* Ensure regular Streamlit markdown is never transparent. */
    .main .block-container [data-testid="stMarkdownContainer"] p,
    .main .block-container [data-testid="stMarkdownContainer"] li,
    .main .block-container [data-testid="stMarkdownContainer"] strong,
    .main .block-container [data-testid="stMarkdownContainer"] em,
    .main .block-container [data-testid="stMarkdownContainer"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Keep captions muted but readable. */
    .main .block-container [data-testid="stCaptionContainer"],
    .main .block-container [data-testid="stCaptionContainer"] * {
        color: #6b7280 !important;
        -webkit-text-fill-color: #6b7280 !important;
        opacity: 1 !important;
    }

    /* Streamlit metrics. */
    .main .block-container [data-testid="stMetricLabel"],
    .main .block-container [data-testid="stMetricLabel"] *,
    .main .block-container [data-testid="stMetricValue"],
    .main .block-container [data-testid="stMetricValue"] *,
    .main .block-container [data-testid="stMetricDelta"],
    .main .block-container [data-testid="stMetricDelta"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Keep icons visible too. */
    div[data-testid="stButton"] button svg,
    div[data-testid="stDownloadButton"] button svg,
    .stButton button svg,
    .stDownloadButton button svg {
        fill: #111827 !important;
        color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* FINAL TEXT VISIBILITY OVERRIDE */
    .main .block-container,
    .main .block-container * {
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
    }

    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6,
    .main .block-container p,
    .main .block-container span,
    .main .block-container div,
    .main .block-container li,
    .main .block-container label,
    .main .block-container strong,
    .main .block-container em {
        -webkit-text-fill-color: #111827 !important;
    }

    .main .block-container [data-testid="stCaptionContainer"],
    .main .block-container [data-testid="stCaptionContainer"] * {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
    }

    .main .block-container [data-testid="stMarkdownContainer"] p,
    .main .block-container [data-testid="stMarkdownContainer"] span,
    .main .block-container [data-testid="stMarkdownContainer"] div,
    .main .block-container [data-testid="stMarkdownContainer"] strong,
    .main .block-container [data-testid="stMarkdownContainer"] li {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    .main .block-container input,
    .main .block-container textarea,
    .main .block-container select,
    .main .block-container [role="combobox"],
    .main .block-container [role="option"] {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* Investigation header values are always dark on the light page. */
    .main .block-container .incident-id,
    .main .block-container .incident-title,
    .main .block-container .incident-stat,
    .main .block-container .section-card,
    .main .block-container .signal-box,
    .main .block-container .action-box,
    .main .block-container .metric-card,
    .main .block-container .metric-label,
    .main .block-container .metric-value {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* Dark banner remains white. */
    .main .block-container .risk-banner,
    .main .block-container .risk-banner * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .main .block-container .risk-banner-label {
        color: #d1d5db !important;
        -webkit-text-fill-color: #d1d5db !important;
    }

    /* Force all visible metric-card text. */
    .main .block-container [data-testid="stMetric"],
    .main .block-container [data-testid="stMetric"] * {
        background: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REFERENCE UI THEME OVERRIDE
# ============================================================
# Streamlit 1.62 can inherit a dark foreground from the active
# theme.  The dashboard is intentionally light in the main area
# and dark in the sidebar, matching the reference design.

st.markdown(
    """
    <style>

    /* ---------- MAIN APPLICATION ---------- */
    :root {
        --primary-color: #2563eb !important;
        --background-color: #f7f9fc !important;
        --secondary-background-color: #ffffff !important;
        --text-color: #111827 !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    section.main,
    .main {
        background: #f7f9fc !important;
        color: #111827 !important;
    }

    .main .block-container {
        max-width: 1480px !important;
        padding-top: 2rem !important;
        padding-left: 2.4rem !important;
        padding-right: 2.4rem !important;
        padding-bottom: 4rem !important;
    }

    /* ---------- ALL NORMAL MAIN TEXT ---------- */
    .main .block-container,
    .main .block-container * {
        --text-color: #111827;
    }

    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6,
    .main .block-container p,
    .main .block-container li,
    .main .block-container label,
    .main .block-container small,
    .main .block-container strong,
    .main .block-container em,
    .main .block-container [data-testid="stMarkdownContainer"],
    .main .block-container [data-testid="stMarkdownContainer"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    .main .block-container [data-testid="stCaptionContainer"],
    .main .block-container [data-testid="stCaptionContainer"] * {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* ---------- TITLE / SUBTITLE ---------- */
    .main .block-container h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em !important;
    }

    .main .block-container h2 {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
    }

    /* ---------- METRIC CARDS ---------- */
    .main [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #dfe5ee !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.2rem !important;
        min-height: 112px !important;
        box-shadow: 0 1px 3px rgba(15,23,42,.04) !important;
    }

    .main [data-testid="stMetricLabel"],
    .main [data-testid="stMetricLabel"] *,
    .main [data-testid="stMetricValue"],
    .main [data-testid="stMetricValue"] *,
    .main [data-testid="stMetricDelta"],
    .main [data-testid="stMetricDelta"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    .main [data-testid="stMetricLabel"] {
        font-size: .84rem !important;
        font-weight: 700 !important;
    }

    .main [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }

    /* ---------- INPUTS / SELECTBOX ---------- */
    .main [data-testid="stTextInput"] input,
    .main [data-testid="stNumberInput"] input,
    .main [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    .main [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    .main [data-testid="stDateInput"] input,
    .main [data-testid="stTimeInput"] input {
        background: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        border-color: #cbd5e1 !important;
        opacity: 1 !important;
    }

    .main [data-testid="stTextInput"] input::placeholder,
    .main [data-testid="stNumberInput"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    .main [data-testid="stSelectbox"] *,
    .main [data-testid="stMultiSelect"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* ---------- BUTTONS ---------- */
    .main .stButton > button,
    .main .stDownloadButton > button,
    .main [data-testid="stButton"] button,
    .main [data-testid="stDownloadButton"] button {
        background: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        border: 1px solid #b8c4d4 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
        box-shadow: 0 1px 3px rgba(15,23,42,.08) !important;
    }

    .main .stButton > button *,
    .main .stDownloadButton > button *,
    .main [data-testid="stButton"] button *,
    .main [data-testid="stDownloadButton"] button * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        fill: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    .main .stButton > button:hover,
    .main .stDownloadButton > button:hover {
        background: #eff6ff !important;
        border-color: #2563eb !important;
        color: #111827 !important;
    }


    /* ======================================================
       TRANSACTION / ACTION WIDGET FIXES
       Keep labels, values and controls visible in the light
       dashboard regardless of the active Streamlit theme.
       ====================================================== */

    .main [data-testid="stTextInput"] label,
    .main [data-testid="stSelectbox"] label,
    .main [data-testid="stNumberInput"] label,
    .main [data-testid="stMultiSelect"] label,
    .main [data-testid="stDateInput"] label,
    .main [data-testid="stTimeInput"] label {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-weight: 700 !important;
    }

    .main [data-testid="stTextInput"] input,
    .main [data-testid="stSelectbox"] input,
    .main [data-testid="stNumberInput"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
    }

    .main [data-baseweb="select"],
    .main [data-baseweb="select"] *,
    .main [role="combobox"],
    .main [role="combobox"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
    }

    /* Investigate buttons: one compact line, no text wrapping. */
    .main div[data-testid="stButton"] button {
        min-width: 112px !important;
        min-height: 38px !important;
        height: 38px !important;
        padding: 0.35rem 0.65rem !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow: hidden !important;
        text-overflow: clip !important;
        font-size: 0.78rem !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .main div[data-testid="stButton"] button p,
    .main div[data-testid="stButton"] button span,
    .main div[data-testid="stButton"] button div {
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow: visible !important;
        line-height: 1 !important;
    }

    /* Custom transaction summary cards. */
    .transaction-summary-card {
        box-sizing: border-box;
        width: 100%;
        min-height: 96px;
        padding: 0.95rem 1rem;
        background: #ffffff !important;
        border: 1px solid #dfe5ee !important;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(15,23,42,.04);
    }

    .transaction-summary-label {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: .45rem;
    }

    .transaction-summary-value {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-size: 1.65rem;
        line-height: 1.05;
        font-weight: 800;
    }

    /* Native dataframe text/header visibility. */
    .main [data-testid="stDataFrame"] *,
    .main [data-testid="stDataEditor"] * {
        opacity: 1 !important;
    }

    /* Keep normal table text dark even when Streamlit theme is dark. */
    .main [data-testid="stDataFrame"] [role="columnheader"],
    .main [data-testid="stDataFrame"] [role="gridcell"] {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }


    .transaction-filter-label {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        font-size: .78rem;
        font-weight: 700;
        line-height: 1.2;
        margin: 0 0 .42rem .05rem;
        opacity: 1 !important;
    }

    /* ---------- TABLES ---------- */
    .main [data-testid="stDataFrame"],
    .main [data-testid="stDataEditor"] {
        background: #ffffff !important;
        border-radius: 12px !important;
    }

    /* ---------- ALERTS / INFO ---------- */
    .main [data-testid="stAlert"],
    .main [data-testid="stNotification"] {
        color: #111827 !important;
        opacity: 1 !important;
    }

    .main [data-testid="stAlert"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* ======================================================
       FINAL VISIBILITY OVERRIDES
       Keep sidebar navigation and AI text readable regardless
       of Streamlit/BaseWeb theme inheritance.
       ====================================================== */

    section[data-testid="stSidebar"] [data-testid="stRadio"] label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label *,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label div,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] * {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) *,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-weight: 700 !important;
    }

    /* AI assistant: force every generated text node to dark text. */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
        mix-blend-mode: normal !important;
    }

    .ai-live-response,
    .ai-live-response * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
        background-color: transparent !important;
    }

    /* AI input text and placeholder. */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea *,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] input * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        opacity: 1 !important;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] span {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        opacity: 1 !important;
    }

    /* ---------- CUSTOM WHITE CARDS ---------- */
    .hero,
    .metric-card,
    .incident-card,
    .section-card,
    .signal-box,
    .action-box {
        color: #111827 !important;
        background: #ffffff !important;
    }

    .hero *,
    .metric-card *,
    .incident-card *,
    .section-card *,
    .signal-box *,
    .action-box * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* ---------- DARK RISK BANNER ---------- */
    .risk-banner {
        background: #111827 !important;
        border-radius: 14px !important;
    }

    .risk-banner,
    .risk-banner * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }


    .risk-banner-label {
        color: #cbd5e1 !important;
        -webkit-text-fill-color: #cbd5e1 !important;
    }

    /* ======================================================
       REFERENCE COMMAND CENTER LAYOUT
       ====================================================== */

    .command-section-title {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-size: 1.45rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 1.15rem 0 .75rem;
    }

    .dashboard-kpi-card {
        box-sizing: border-box;
        width: 100%;
        min-height: 118px;
        padding: 1rem 1.05rem .9rem;
        background: #ffffff !important;
        border: 1px solid #dfe5ee;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(15,23,42,.04);
    }

    .dashboard-kpi-label {
        color: #1f2937 !important;
        -webkit-text-fill-color: #1f2937 !important;
        font-size: .78rem;
        line-height: 1.2;
        font-weight: 700;
        margin-bottom: .65rem;
    }

    .dashboard-kpi-value {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-size: 1.75rem;
        line-height: 1.1;
        font-weight: 800;
        white-space: nowrap;
    }

    .dashboard-kpi-footer {
        font-size: .72rem;
        font-weight: 700;
        margin-top: .55rem;
    }

    .dashboard-kpi-footer.neutral,
    .dashboard-kpi-footer.blue {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
    }

    .dashboard-kpi-footer.red {
        color: #ef4444 !important;
        -webkit-text-fill-color: #ef4444 !important;
    }

    .dashboard-kpi-footer.amber {
        color: #f59e0b !important;
        -webkit-text-fill-color: #f59e0b !important;
    }

    .dashboard-kpi-footer.green {
        color: #22c55e !important;
        -webkit-text-fill-color: #22c55e !important;
    }

    .incident-section-heading {
        display: flex;
        align-items: center;
        gap: .55rem;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-size: 1.55rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.025em;
        margin-top: 1.1rem;
        margin-bottom: .1rem;
    }

    .incident-heading-icon {
        font-size: 1.45rem;
        line-height: 1;
    }

    .incident-table-header {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-size: .69rem;
        font-weight: 800;
        line-height: 1.25;
        min-height: 2.1rem;
        display: flex;
        align-items: flex-end;
        padding: 0 .15rem .35rem;
    }

    .incident-table-cell {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-size: .78rem;
        line-height: 1.3;
        min-height: 44px;
        display: flex;
        align-items: center;
        padding: .15rem;
        font-weight: 600;
    }

    .incident-type-cell {
        font-weight: 500 !important;
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
    }

    .incident-link {
        color: #2563eb !important;
        -webkit-text-fill-color: #2563eb !important;
        font-weight: 800 !important;
    }

    .risk-number {
        font-weight: 800 !important;
    }

    .time-cell {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-size: .72rem !important;
        white-space: nowrap;
    }

    .severity-critical,
    .severity-high,
    .severity-medium,
    .severity-low {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 55px;
        padding: .24rem .45rem;
        border-radius: 6px;
        font-size: .62rem;
        font-weight: 800;
        letter-spacing: .02em;
        line-height: 1;
    }

    .severity-critical {
        background: #fee2e2 !important;
        color: #dc2626 !important;
        -webkit-text-fill-color: #dc2626 !important;
    }

    .severity-high {
        background: #ffedd5 !important;
        color: #ea580c !important;
        -webkit-text-fill-color: #ea580c !important;
    }

    .severity-medium {
        background: #fef3c7 !important;
        color: #d97706 !important;
        -webkit-text-fill-color: #d97706 !important;
    }

    .severity-low {
        background: #dcfce7 !important;
        color: #16a34a !important;
        -webkit-text-fill-color: #16a34a !important;
    }

    .incident-row-divider {
        height: 1px;
        background: #e8edf3 !important;
        margin: 0 .1rem;
    }

    .command-subsection-title {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 1.5rem;
        margin-bottom: .75rem;
    }

    .distribution-title {
        color: #1f2937 !important;
        -webkit-text-fill-color: #1f2937 !important;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: .45rem;
    }

    /* Keep the existing 270px sidebar when expanded,
       but allow Streamlit to collapse it normally. */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 270px !important;
        max-width: 270px !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        max-width: 0 !important;
        width: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: .08rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 7px !important;
        padding: .22rem .4rem !important;
        margin: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(59,130,246,.12) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(
            90deg,
            rgba(59,130,246,.24),
            rgba(59,130,246,.08)
        ) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: .82rem !important;
        font-weight: 650 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: #17233b !important;
        border-color: #30415f !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: .15rem 0 !important;
        min-height: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    @media (max-width: 1100px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .dashboard-kpi-value {
            font-size: 1.35rem;
        }

        .incident-table-cell,
        .incident-table-header {
            font-size: .66rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AI FRAUD ASSISTANT CHAT VISIBILITY + LIVE STREAMING
# ============================================================
#
# Streamlit's chat components can inherit theme foreground colors.
# Keep the fix scoped to the AI chat so the rest of the dashboard
# remains unchanged.
#
st.markdown(
    """
    <style>
    /* ---------- CHAT MESSAGE VISIBILITY ---------- */

    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #dfe5ee !important;
        border-radius: 14px !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.75rem !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Force Qwen's generated text to dark/readable text. */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Generated code blocks remain readable. */
    [data-testid="stChatMessage"] pre {
        background: #f3f4f6 !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }

    [data-testid="stChatMessage"] code {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* ---------- CHAT INPUT ---------- */

    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        background: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        color: #111827 !important;
        background: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stChatInput"] button svg {
        color: #111827 !important;
        fill: #111827 !important;
        opacity: 1 !important;
    }

    /* ======================================================
       GLOBAL STREAMLIT WIDGET VISIBILITY FIX
       These selectors intentionally do NOT depend on .main.
       Streamlit can render widget internals outside that class.
       ====================================================== */

    [data-testid="stChatMessage"] {
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
        mix-blend-mode: normal !important;
    }

    .ai-live-response,
    .ai-live-response * {
        display: block !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background: transparent !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
        font-weight: 500 !important;
        text-shadow: none !important;
        mix-blend-mode: normal !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    .ai-history-response {
        padding: .2rem 0 !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        opacity: 1 !important;
        visibility: visible !important;
        background: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* Live Risk Simulator / native widget labels. */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label *,
    [data-testid="stTextInput"] label,
    [data-testid="stTextInput"] label *,
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] label *,
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] label *,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] label * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stRadio"] [role="radiogroup"] label,
    [data-testid="stRadio"] [role="radiogroup"] label p,
    [data-testid="stRadio"] [role="radiogroup"] label span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stNumberInput"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* ---------- LIVE GENERATION STATUS ---------- */

    .ai-stream-status {
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        margin: .15rem 0 .55rem;
        padding: .35rem .65rem;
        border-radius: 999px;
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        color: #1d4ed8 !important;
        -webkit-text-fill-color: #1d4ed8 !important;
        font-size: .76rem;
        font-weight: 700;
    }

    .ai-stream-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2563eb !important;
        display: inline-block;
        animation: ai-stream-pulse 1s infinite ease-in-out;
    }

    @keyframes ai-stream-pulse {
        0%, 100% {
            opacity: .35;
            transform: scale(.85);
        }
        50% {
            opacity: 1;
            transform: scale(1);
        }
    }


    /* ======================================================
       FINAL GLOBAL TEXT VISIBILITY PATCH
       ====================================================== */

    /* Streamlit alerts / warnings / errors / info */
    [data-testid="stAlert"],
    [data-testid="stAlert"] *,
    [data-testid="stNotification"],
    [data-testid="stNotification"] * {
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] strong,
    [data-testid="stAlert"] em {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* All normal widget labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stMarkdownContainer"] label,
    [data-testid="stMarkdownContainer"] label * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Radio labels outside the sidebar */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label *,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Sidebar navigation remains white */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label *,
    section[data-testid="stSidebar"] [data-testid="stRadio"] p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] span,
    section[data-testid="stSidebar"] [data-testid="stRadio"] div {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Sidebar headings and status text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Text inputs / select boxes / number inputs */
    [data-testid="stTextInput"] label,
    [data-testid="stTextInput"] label *,
    [data-testid="stSelectbox"] label,
    [data-testid="stSelectbox"] label *,
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] label *,
    [data-testid="stTextArea"] label,
    [data-testid="stTextArea"] label *,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] label * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stNumberInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder,
    [data-testid="stSelectbox"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* Buttons: readable dark text on light buttons */
    .main [data-testid="stButton"] button,
    .main [data-testid="stButton"] button p,
    .main [data-testid="stButton"] button span,
    .main [data-testid="stButton"] button div {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* AI chat input */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] input::placeholder {
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background: #ffffff !important;
        caret-color: #111827 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
    }

    /* AI assistant messages: black, always visible */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] *,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        opacity: 1 !important;
        visibility: visible !important;
        mix-blend-mode: normal !important;
        text-shadow: none !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] span,
    .ai-live-response,
    .ai-live-response * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* ======================================================
       END FINAL GLOBAL TEXT VISIBILITY PATCH
       ====================================================== */
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_json(path):

    path = Path(path)

    if not path.exists():
        return []

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return data

    except Exception:
        return []


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_transactions():

    if not TRANSACTION_FILE.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            TRANSACTION_FILE
        )

    except Exception:
        return pd.DataFrame()


def append_transaction_to_csv(transaction):
    """Append one raw transaction to the existing transactions.csv."""
    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {TRANSACTION_FILE}"
        )

    existing_df = pd.read_csv(TRANSACTION_FILE)
    new_row = pd.DataFrame([transaction])

    # Preserve the existing CSV schema exactly.
    for column in existing_df.columns:
        if column not in new_row.columns:
            new_row[column] = ""

    new_row = new_row.reindex(columns=existing_df.columns)

    updated_df = pd.concat(
        [existing_df, new_row],
        ignore_index=True,
    )

    updated_df.to_csv(
        TRANSACTION_FILE,
        index=False,
    )

    # The dashboard caches transactions, so invalidate it immediately.
    load_transactions.clear()


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        if pd.isna(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _risk_report_columns():
    return [
        "report_id",
        "created_at",
        "transaction_id",
        "customer_id",
        "merchant_id",
        "amount",
        "timestamp",
        "payment_method",
        "device_id",
        "ip_id",
        "address_id",
        "account_age_days",
        "location",
        "refund_count",
        "refund_amount",
        "chargeback_count",
        "chargeback_amount",
        "is_refund",
        "is_chargeback",
        "ml_probability",
        "ml_risk_score",
        "evidence_risk",
        "abuse_risk",
        "final_risk",
        "risk_level",
        "recommended_action",
        "exposure_at_risk",
        "relationship_points",
        "evidence_signals",
        "escalated",
        "incident_id",
        "incident_severity",
        "incident_type",
        "incident_reason",
    ]


def persist_transaction_risk_report(
    transaction,
    prediction,
    decision,
    escalated=False,
    incident_id="",
    incident_severity="",
    incident_type="",
    incident_reason="",
):
    """
    Persist a complete audit/report row for every newly scored
    transaction. The report contains the transaction properties,
    model output, deterministic evidence, abuse score, final score,
    recommendation and escalation status.
    """

    TRANSACTION_RISK_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk = prediction.get(
        "risk",
        {},
    ) if isinstance(prediction, dict) else {}

    evidence_signals = decision.get(
        "evidence_reasons",
        [],
    )

    report = {
        "report_id": (
            "RISK-"
            + str(
                transaction.get(
                    "transaction_id",
                    "UNKNOWN",
                )
            )
            + "-"
            + datetime.now().strftime(
                "%Y%m%d%H%M%S"
            )
        ),
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "transaction_id": transaction.get(
            "transaction_id",
            "",
        ),
        "customer_id": transaction.get(
            "customer_id",
            "",
        ),
        "merchant_id": transaction.get(
            "merchant_id",
            "",
        ),
        "amount": _safe_float(
            transaction.get(
                "amount",
                0,
            )
        ),
        "timestamp": transaction.get(
            "timestamp",
            "",
        ),
        "payment_method": transaction.get(
            "payment_method",
            "",
        ),
        "device_id": transaction.get(
            "device_id",
            "",
        ),
        "ip_id": transaction.get(
            "ip_id",
            "",
        ),
        "address_id": transaction.get(
            "address_id",
            "",
        ),
        "account_age_days": transaction.get(
            "account_age_days",
            "",
        ),
        "location": transaction.get(
            "location",
            "",
        ),
        "refund_count": transaction.get(
            "refund_count",
            0,
        ),
        "refund_amount": transaction.get(
            "refund_amount",
            0,
        ),
        "chargeback_count": transaction.get(
            "chargeback_count",
            0,
        ),
        "chargeback_amount": transaction.get(
            "chargeback_amount",
            0,
        ),
        "is_refund": transaction.get(
            "is_refund",
            0,
        ),
        "is_chargeback": transaction.get(
            "is_chargeback",
            0,
        ),
        "ml_probability": _safe_float(
            decision.get(
                "ml_probability",
                risk.get(
                    "fraud_probability",
                    0,
                ),
            )
        ),
        "ml_risk_score": _safe_float(
            decision.get(
                "ml_score",
                0,
            )
        ),
        "evidence_risk": _safe_float(
            decision.get(
                "evidence_score",
                0,
            )
        ),
        "abuse_risk": _safe_float(
            decision.get(
                "abuse_risk",
                decision.get(
                    "relationship_points",
                    0,
                ),
            )
        ),
        "final_risk": _safe_float(
            decision.get(
                "final_score",
                0,
            )
        ),
        "risk_level": decision.get(
            "risk_level",
            risk.get(
                "risk_level",
                "",
            ),
        ),
        "recommended_action": decision.get(
            "recommended_action",
            risk.get(
                "recommended_action",
                "",
            ),
        ),
        "exposure_at_risk": (
            _safe_float(
                decision.get(
                    "ml_probability",
                    0,
                )
            )
            * _safe_float(
                transaction.get(
                    "amount",
                    0,
                )
            )
        ),
        "relationship_points": _safe_float(
            decision.get(
                "relationship_points",
                0,
            )
        ),
        "evidence_signals": json.dumps(
            evidence_signals,
            default=str,
        ),
        "escalated": int(bool(escalated)),
        "incident_id": incident_id,
        "incident_severity": incident_severity,
        "incident_type": incident_type,
        "incident_reason": incident_reason,
    }

    columns = _risk_report_columns()

    report_exists = TRANSACTION_RISK_REPORT_FILE.exists()

    with open(
        TRANSACTION_RISK_REPORT_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )

        if not report_exists:
            writer.writeheader()

        writer.writerow(
            {
                column: report.get(
                    column,
                    "",
                )
                for column in columns
            }
        )


def _load_incident_json_for_update():
    if not INCIDENT_FILE.exists():
        return []

    try:
        with open(
            INCIDENT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            incidents = data.get(
                "incidents",
                [],
            )
            return incidents if isinstance(
                incidents,
                list,
            ) else []

        return data if isinstance(
            data,
            list,
        ) else []

    except Exception:
        return []


def create_incident_from_new_transaction(
    transaction,
    prediction,
    decision,
):
    """
    Escalate a newly scored transaction into the existing Incident
    Centre only when the final risk reaches the review threshold.

    The incident is persisted to incident_summary.json so it survives
    Streamlit reruns/restarts and appears in the existing Incident Centre.
    """

    final_score = _safe_float(
        decision.get(
            "final_score",
            0,
        )
    )

    if final_score < 40:
        return {
            "escalated": False,
            "incident_id": "",
            "severity": "",
            "incident_type": "",
            "reason": "",
        }

    if final_score >= 80:
        severity = "CRITICAL"
    elif final_score >= 60:
        severity = "HIGH"
    else:
        severity = "MEDIUM"

    evidence_reasons = decision.get(
        "evidence_reasons",
        [],
    )

    strongest_signal = "Risk signal combination"

    if evidence_reasons:
        strongest = max(
            evidence_reasons,
            key=lambda item: _safe_float(
                item.get(
                    "score",
                    0,
                )
            ),
        )
        strongest_signal = str(
            strongest.get(
                "factor",
                strongest_signal,
            )
        )

    abuse_risk = _safe_float(
        decision.get(
            "abuse_risk",
            0,
        )
    )

    if abuse_risk >= 75:
        incident_type = "Abuse Ring Risk"
    elif strongest_signal:
        incident_type = strongest_signal
    else:
        incident_type = "Fraud Risk Escalation"

    reason_parts = [
        f"Final risk {final_score:.2f}/100",
        f"ML risk {decision.get('ml_score', 0):.2f}/100",
        f"evidence risk {decision.get('evidence_score', 0):.2f}/100",
        f"abuse risk {abuse_risk:.2f}/100",
    ]

    reason = (
        "New transaction escalated after live risk assessment: "
        + "; ".join(reason_parts)
        + "."
    )

    incidents = _load_incident_json_for_update()

    # Do not create duplicate incidents for the same transaction.
    transaction_id = str(
        transaction.get(
            "transaction_id",
            "",
        )
    )

    for existing in incidents:
        if str(
            existing.get(
                "source_transaction_id",
                "",
            )
        ) == transaction_id:
            return {
                "escalated": True,
                "incident_id": existing.get(
                    "incident_id",
                    "",
                ),
                "severity": existing.get(
                    "severity",
                    severity,
                ),
                "incident_type": existing.get(
                    "incident_type",
                    incident_type,
                ),
                "reason": existing.get(
                    "description",
                    reason,
                ),
            }

    max_number = 0

    for existing in incidents:
        raw_id = str(
            existing.get(
                "incident_id",
                "",
            )
        )

        try:
            digits = raw_id.replace(
                "INC-",
                "",
            ).replace(
                "INC_",
                "",
            )
            max_number = max(
                max_number,
                int(digits),
            )
        except Exception:
            continue

    incident_id = f"INC-{max_number + 1:04d}"

    amount = _safe_float(
        transaction.get(
            "amount",
            0,
        )
    )

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    incident = {
        "incident_id": incident_id,
        "incident_type": incident_type,
        "severity": severity,
        "risk_score": round(
            final_score
        ),
        "transaction_count": 1,
        "customer_count": 1,
        "estimated_exposure": amount,
        "total_transaction_amount": amount,
        "first_seen": transaction.get(
            "timestamp",
            now,
        ),
        "last_seen": transaction.get(
            "timestamp",
            now,
        ),
        "detected_at": now,
        "status": "OPEN",
        "source": "live_transaction_risk_engine",
        "source_transaction_id": transaction_id,
        "description": reason,
        "escalation_reason": reason,
        "recommended_action": decision.get(
            "recommended_action",
            "REVIEW",
        ),
    }

    incidents.append(
        incident
    )

    INCIDENT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        INCIDENT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            incidents,
            file,
            indent=2,
            default=str,
        )

    load_incidents.clear()
    load_json.clear()

    return {
        "escalated": True,
        "incident_id": incident_id,
        "severity": severity,
        "incident_type": incident_type,
        "reason": reason,
    }



def predict_new_transaction(transaction):
    """Send the new transaction through the existing API/model."""
    response = requests.post(
        f"{API_URL}/predict",
        json={
            "customer_id": transaction["customer_id"],
            "merchant_id": transaction["merchant_id"],
            "amount": float(transaction["amount"]),
            "timestamp": transaction["timestamp"],
            "payment_method": transaction["payment_method"],
            "device_id": transaction["device_id"],
            "ip_id": transaction["ip_id"],
            "address_id": transaction["address_id"],
            "account_age_days": int(
                transaction["account_age_days"]
            ),
            "location": transaction["location"],
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def _safe_transaction_value(transaction, key, default=""):
    value = transaction.get(key, default)
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def get_active_model_version():
    """Return the model version currently active in the risk API."""
    try:
        response = requests.get(
            f"{API_URL}/model-info",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("model_version", "")).strip()
    except Exception:
        return ""


def _build_existing_transaction_payload(row):
    """Convert an existing CSV row into the same /predict payload used for new transactions."""
    return {
        "transaction_id": str(
            _safe_transaction_value(row, "transaction_id", "")
        ),
        "customer_id": str(
            _safe_transaction_value(row, "customer_id", "")
        ),
        "merchant_id": str(
            _safe_transaction_value(row, "merchant_id", "")
        ),
        "amount": float(
            _safe_transaction_value(row, "amount", 0) or 0
        ),
        "timestamp": str(
            _safe_transaction_value(row, "timestamp", "")
        ),
        "payment_method": str(
            _safe_transaction_value(row, "payment_method", "")
        ),
        "device_id": str(
            _safe_transaction_value(row, "device_id", "")
        ),
        "ip_id": str(
            _safe_transaction_value(row, "ip_id", "")
        ),
        "address_id": str(
            _safe_transaction_value(row, "address_id", "")
        ),
        "account_age_days": int(
            float(
                _safe_transaction_value(
                    row,
                    "account_age_days",
                    0,
                )
                or 0
            )
        ),
        "location": str(
            _safe_transaction_value(row, "location", "")
        ),
    }


def rescore_existing_incidents_with_active_model(incidents, transactions_df):
    """
    Re-score existing incidents through the currently active model.

    Existing incident JSON records contain historical risk values. This
    function maps each incident to its source transaction, sends that
    transaction through the same /predict endpoint used by new
    transactions, applies the existing deterministic risk-fusion engine,
    and persists the refreshed risk values with the active model version.

    This does not create new incidents and does not create learning labels.
    Human-confirmed feedback remains the only source of continual-learning
    labels.
    """
    active_version = get_active_model_version()

    if not active_version:
        return {
            "status": "error",
            "message": "Could not determine the active model version.",
            "updated": 0,
            "failed": 0,
            "skipped": len(incidents or []),
        }

    if transactions_df is None or transactions_df.empty:
        return {
            "status": "error",
            "message": "Transaction dataset is unavailable.",
            "updated": 0,
            "failed": 0,
            "skipped": len(incidents or []),
        }

    transaction_lookup = {}
    if "transaction_id" in transactions_df.columns:
        for _, row in transactions_df.iterrows():
            transaction_id = str(
                _safe_transaction_value(row, "transaction_id", "")
            ).strip()
            if transaction_id:
                transaction_lookup[transaction_id] = row.to_dict()

    updated = 0
    failed = 0
    skipped = 0

    refreshed_incidents = []

    for incident in incidents or []:
        incident_copy = dict(incident)
        current_version = str(
            incident_copy.get("model_version", "")
        ).strip()

        if current_version == active_version:
            skipped += 1
            refreshed_incidents.append(incident_copy)
            continue

        source_transaction_id = str(
            incident_copy.get("source_transaction_id", "")
        ).strip()

        row = transaction_lookup.get(source_transaction_id)

        # Legacy incidents may not have source_transaction_id. Fall back to
        # the first transaction belonging to the incident.
        if row is None and "incident_id" in transactions_df.columns:
            incident_id = str(
                incident_copy.get("incident_id", "")
            ).strip()
            if incident_id:
                matches = transactions_df[
                    transactions_df["incident_id"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .isin(incident_id_variants(incident_id))
                ]
                if not matches.empty:
                    row = matches.iloc[0].to_dict()
                    source_transaction_id = str(
                        _safe_transaction_value(
                            row,
                            "transaction_id",
                            "",
                        )
                    ).strip()

        if row is None:
            failed += 1
            refreshed_incidents.append(incident_copy)
            continue

        try:
            transaction = _build_existing_transaction_payload(row)
            prediction = predict_new_transaction(transaction)
            decision = calculate_live_risk_decision(
                prediction,
                transaction,
            )

            final_score = _safe_float(
                decision.get("final_score", 0)
            )
            final_level = str(
                decision.get("risk_level", "LOW")
            ).upper()

            incident_copy["risk_score"] = round(final_score)
            incident_copy["severity"] = final_level
            incident_copy["recommended_action"] = decision.get(
                "recommended_action",
                incident_copy.get("recommended_action", "REVIEW"),
            )
            incident_copy["model_version"] = active_version
            incident_copy["ml_probability"] = round(
                _safe_float(decision.get("ml_probability", 0)),
                6,
            )
            incident_copy["ml_score"] = round(
                _safe_float(decision.get("ml_score", 0)),
                2,
            )
            incident_copy["evidence_score"] = round(
                _safe_float(decision.get("evidence_score", 0)),
                2,
            )
            incident_copy["final_risk_score"] = round(
                final_score,
                2,
            )
            incident_copy["last_model_scored_at"] = (
                datetime.now().astimezone().isoformat()
            )

            if source_transaction_id and not incident_copy.get(
                "source_transaction_id"
            ):
                incident_copy["source_transaction_id"] = (
                    source_transaction_id
                )

            updated += 1

        except Exception:
            failed += 1

        refreshed_incidents.append(incident_copy)

    try:
        INCIDENT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with open(
            INCIDENT_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                refreshed_incidents,
                file,
                indent=2,
                default=str,
            )
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Could not persist refreshed incidents: {exc}",
            "updated": updated,
            "failed": failed,
            "skipped": skipped,
        }

    load_incidents.clear()
    load_json.clear()

    return {
        "status": "success",
        "model_version": active_version,
        "updated": updated,
        "failed": failed,
        "skipped": skipped,
        "total": len(refreshed_incidents),
    }


def enrich_new_transaction_with_prediction(
    transaction,
    prediction,
):
    """Build a CSV row using the model's real engineered features."""
    row = dict(transaction)

    behavioral_features = prediction.get(
        "behavioral_features",
        {},
    )

    for key, value in behavioral_features.items():
        row[key] = value

    risk = prediction.get("risk", {})

    row["risk_score"] = risk.get(
        "risk_score",
        "",
    )

    row["fraud_probability"] = risk.get(
        "fraud_probability",
        "",
    )

    row["fraud_probability_percent"] = risk.get(
        "fraud_probability_percent",
        "",
    )

    row["risk_level"] = risk.get(
        "risk_level",
        "",
    )

    row["recommended_action"] = risk.get(
        "recommended_action",
        "",
    )

    # These represent unknown real-world outcomes at insertion time.
    row["is_fraud"] = 0
    row["fraud_type"] = ""

    # The prediction endpoint does not create an incident.
    row["incident_id"] = ""
    row["incident_type"] = ""
    row["incident_severity"] = ""
    row["detected_incident_id"] = ""

    return row


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_incidents():

    data = load_json(
        INCIDENT_FILE
    )

    if isinstance(data, dict):

        if "incidents" in data:
            return data["incidents"]

        return [data]

    if isinstance(data, list):
        return data

    return []


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_response_actions():

    if not RESPONSE_FILE.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            RESPONSE_FILE
        )

    except Exception:
        return pd.DataFrame()


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def load_metrics():

    data = load_json(
        METRICS_FILE
    )

    if isinstance(data, dict):
        return data

    return {}


@st.cache_data(
    ttl=10,
    show_spinner=False,
)
def load_prediction_audit():
    """Load prediction audit records written by the API."""
    if not AUDIT_FILE.exists():
        return pd.DataFrame()

    records = []

    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(record, dict):
                    records.append(record)

    except Exception:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ============================================================
# HELPERS
# ============================================================

def parse_json(value):

    if isinstance(
        value,
        (list, dict),
    ):
        return value

    if value is None:
        return []

    if isinstance(
        value,
        float,
    ) and pd.isna(value):
        return []

    try:

        result = json.loads(
            str(value)
        )

        if isinstance(
            result,
            (list, dict),
        ):
            return result

    except Exception:
        pass

    return []


def pretty_type(value):

    if value is None:
        return ""

    return (
        str(value)
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
        .title()
    )


def format_currency(value):

    try:

        return (
            f"₹{float(value):,.0f}"
        )

    except Exception:

        return "₹0"


def format_number(value):

    try:

        return f"{int(value):,}"

    except Exception:

        return "0"


def incident_id_variants(value):

    raw = str(
        value or ""
    ).strip()

    if not raw:
        return []

    variants = [
        raw
    ]

    try:

        if raw.upper().startswith(
            "INC-"
        ):

            number = int(
                raw.split(
                    "-",
                    1,
                )[1]
            )

            variants.extend(
                [
                    f"INC_{number:03d}",
                    f"INC_{number:04d}",
                ]
            )

        elif raw.upper().startswith(
            "INC_"
        ):

            number = int(
                raw.split(
                    "_",
                    1,
                )[1]
            )

            variants.extend(
                [
                    f"INC-{number:04d}",
                    f"INC_{number:03d}",
                ]
            )

    except Exception:
        pass

    return list(
        dict.fromkeys(
            variants
        )
    )


# ============================================================
# TRANSACTION NORMALIZATION
# ============================================================

def prepare_transaction_data(
    df,
    incidents,
):
    """
    Normalize transaction-level risk information.

    Existing transaction risk fields are preferred.
    Otherwise risk is mapped from the transaction's incident.

    All calculated vectors are explicitly converted to pandas
    Series before fillna() is called.
    """

    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    result = df.copy()

    # --------------------------------------------------------
    # INCIDENT ID
    # --------------------------------------------------------

    if "incident_id" not in result.columns:

        result["incident_id"] = ""

    result["incident_id"] = (
        result["incident_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # INCIDENT LOOKUP
    # --------------------------------------------------------

    incident_lookup = {}

    for incident in (
        incidents or []
    ):

        incident_id = str(
            incident.get(
                "incident_id",
                "",
            )
        ).strip()

        if not incident_id:
            continue

        raw_score = pd.to_numeric(
            incident.get(
                "risk_score",
                0,
            ),
            errors="coerce",
        )

        try:

            incident_score = float(
                raw_score
            )

            if pd.isna(
                incident_score
            ):

                incident_score = 0.0

        except Exception:

            incident_score = 0.0

        incident_severity = str(
            incident.get(
                "severity",
                "",
            )
        ).upper().strip()

        incident_type = str(
            incident.get(
                "incident_type",
                "",
            )
        ).strip()

        incident_information = {
            "risk_score":
                incident_score,

            "risk_level":
                incident_severity,

            "incident_type":
                incident_type,
        }

        for variant in (
            incident_id_variants(
                incident_id
            )
        ):

            incident_lookup[
                variant
            ] = incident_information

    # --------------------------------------------------------
    # CALCULATE TRANSACTION RISK
    # --------------------------------------------------------

    scores = []
    levels = []
    types = []

    for _, row in result.iterrows():

        transaction_incident_id = str(
            row.get(
                "incident_id",
                "",
            )
        ).strip()

        incident_information = None

        for variant in (
            incident_id_variants(
                transaction_incident_id
            )
        ):

            if (
                variant
                in incident_lookup
            ):

                incident_information = (
                    incident_lookup[
                        variant
                    ]
                )

                break

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        existing_score = pd.to_numeric(
            row.get(
                "risk_score",
                None,
            ),
            errors="coerce",
        )

        if pd.notna(
            existing_score
        ):

            score = float(
                existing_score
            )

        elif (
            incident_information
            is not None
        ):

            score = float(
                incident_information[
                    "risk_score"
                ]
            )

        else:

            fraud_value = str(
                row.get(
                    "is_fraud",
                    "",
                )
            ).strip().lower()

            if fraud_value in {
                "1",
                "true",
                "yes",
            }:

                score = 75.0

            else:

                score = 0.0

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        # ----------------------------------------------------
        # LEVEL
        # ----------------------------------------------------

        existing_level = str(
            row.get(
                "risk_level",
                "",
            )
        ).upper().strip()

        if existing_level in {
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
        }:

            level = existing_level

        elif (
            incident_information
            is not None
            and incident_information[
                "risk_level"
            ]
            in {
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            }
        ):

            level = incident_information[
                "risk_level"
            ]

        elif score >= 90:

            level = "CRITICAL"

        elif score >= 75:

            level = "HIGH"

        elif score >= 50:

            level = "MEDIUM"

        else:

            level = "LOW"

        # ----------------------------------------------------
        # INCIDENT TYPE
        # ----------------------------------------------------

        if (
            incident_information
            is not None
        ):

            incident_type = str(
                incident_information.get(
                    "incident_type",
                    "",
                )
            ).strip()

        else:

            incident_type = str(
                row.get(
                    "incident_type",
                    "",
                )
            ).strip()

        scores.append(
            score
        )

        levels.append(
            level
        )

        types.append(
            incident_type
        )

    # --------------------------------------------------------
    # CRITICAL FIX
    #
    # NEVER:
    #
    # pd.to_numeric(scores).fillna(...)
    #
    # because that can return numpy.ndarray.
    #
    # ALWAYS construct a pandas Series first.
    # --------------------------------------------------------

    score_series = pd.Series(
        scores,
        index=result.index,
        dtype="float64",
    )

    result["risk_score"] = (
        score_series
        .fillna(0.0)
        .clip(
            lower=0,
            upper=100,
        )
        .round(0)
        .astype(int)
    )

    level_series = pd.Series(
        levels,
        index=result.index,
        dtype="string",
    )

    result["risk_level"] = (
        level_series
        .fillna("LOW")
        .astype(str)
        .str.upper()
    )

    type_series = pd.Series(
        types,
        index=result.index,
        dtype="string",
    )

    result[
        "incident_type_display"
    ] = (
        type_series
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # FRAUD TYPE
    # --------------------------------------------------------

    if "fraud_type" not in result.columns:

        result["fraud_type"] = ""

    result["fraud_type"] = (
        result["fraud_type"]
        .fillna("")
        .astype(str)
        .replace(
            {
                "nan": "",
                "None": "",
            }
        )
    )

    return result
def format_transaction_table(df):

    if df is None or df.empty:
        return pd.DataFrame()

    display = df.copy()

    columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "amount",
        "timestamp",
        "payment_method",
        "risk_score",
        "risk_level",
        "fraud_type",
        "incident_id",
    ]

    columns = [
        c
        for c in columns
        if c in display.columns
    ]

    display = display[
        columns
    ].rename(
        columns={
            "transaction_id":
                "Transaction ID",

            "customer_id":
                "Customer ID",

            "merchant_id":
                "Merchant ID",

            "amount":
                "Amount",

            "timestamp":
                "Timestamp",

            "payment_method":
                "Payment Method",

            "risk_score":
                "Risk Score",

            "risk_level":
                "Risk Level",

            "fraud_type":
                "Fraud Type",

            "incident_id":
                "Incident ID",
        }
    )

    if "Amount" in display.columns:

        amount_series = pd.to_numeric(
            display["Amount"],
            errors="coerce",
        )

        display["Amount"] = (
            amount_series
            .fillna(0)
            .map(
                lambda x:
                    f"₹{x:,.2f}"
            )
        )

    return display


def get_selected_incident():

    selected_id = (
        st.session_state.get(
            "selected_incident_id"
        )
    )

    if not selected_id:
        return None

    incidents = load_incidents()

    for incident in incidents:

        if str(
            incident.get(
                "incident_id",
                "",
            )
        ) == str(
            selected_id
        ):

            return incident

    return None


def get_incident_transactions(
    incident,
    df,
):

    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    if "incident_id" not in df.columns:
        return pd.DataFrame()

    incident_id = str(
        incident.get(
            "incident_id",
            "",
        )
    ).strip()

    possible_ids = (
        incident_id_variants(
            incident_id
        )
    )

    if not possible_ids:
        return pd.DataFrame()

    normalized_ids = set(
        str(x).strip()
        for x in possible_ids
    )

    result = df[
        df["incident_id"]
        .astype(str)
        .str.strip()
        .isin(
            normalized_ids
        )
    ]

    return result.copy()


# ============================================================
# AI FRAUD ASSISTANT
# ============================================================

def render_prediction_evidence(prediction):
    """Display deterministic evidence returned by the fraud-risk API."""
    evidence = (
        prediction.get("evidence", {})
        if isinstance(prediction, dict)
        else {}
    )

    if not isinstance(evidence, dict):
        return

    factors = evidence.get("risk_factors", [])
    relationship = evidence.get(
        "relationship_evidence",
        {},
    )

    # If the API does not provide relationship counts, derive them
    # directly from the transaction dataset for this transaction.
    if not isinstance(relationship, dict):
        relationship = {}

    current_transaction = st.session_state.get(
        "last_added_transaction",
        {},
    )

    try:
        transactions_df = load_transactions()
    except Exception:
        transactions_df = pd.DataFrame()

    def _live_customer_count(column_name):
        if (
            not isinstance(transactions_df, pd.DataFrame)
            or transactions_df.empty
            or column_name not in transactions_df.columns
            or "customer_id" not in transactions_df.columns
        ):
            return relationship.get(
                {
                    "device_id": "device_customer_count",
                    "ip_id": "ip_customer_count",
                    "address_id": "address_customer_count",
                }.get(
                    column_name,
                    "",
                ),
                0,
            )

        target = str(
            current_transaction.get(
                column_name,
                "",
            )
        ).strip()

        if not target:
            return relationship.get(
                {
                    "device_id": "device_customer_count",
                    "ip_id": "ip_customer_count",
                    "address_id": "address_customer_count",
                }.get(
                    column_name,
                    "",
                ),
                0,
            )

        # Count UNIQUE customers across every transaction sharing the
        # same device/IP/address attribute. The dashboard intentionally
        # displays only the number, not the underlying entity details.
        matched = transactions_df[
            transactions_df[column_name]
            .fillna("")
            .astype(str)
            .str.strip()
            == target
        ]

        customer_values = (
            matched["customer_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        customer_values = customer_values[
            customer_values != ""
        ]

        return int(
            customer_values.nunique()
        )

    relationship = {
        "device_customer_count": _live_customer_count(
            "device_id"
        ),
        "ip_customer_count": _live_customer_count(
            "ip_id"
        ),
        "address_customer_count": _live_customer_count(
            "address_id"
        ),
    }
    explanation = evidence.get(
        "explanation",
        {},
    )

    st.subheader("Deterministic Risk Evidence")

    rows = []

    for factor in factors:
        if not isinstance(factor, dict):
            continue

        rows.append(
            {
                "Risk Factor": factor.get(
                    "factor",
                    "Unknown",
                ),
                "Value": factor.get(
                    "value",
                    "",
                ),
                "Severity": factor.get(
                    "severity",
                    "",
                ),
                "Evidence": factor.get(
                    "explanation",
                    "",
                ),
            }
        )

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

    if isinstance(relationship, dict) and relationship:
        e1, e2, e3 = st.columns(
            3,
            gap="medium",
        )

        with e1:
            visible_metric(
                "Shared Device Customers",
                relationship.get(
                    "device_customer_count",
                    0,
                ),
            )

        with e2:
            visible_metric(
                "Shared IP Customers",
                relationship.get(
                    "ip_customer_count",
                    0,
                ),
            )

        with e3:
            visible_metric(
                "Shared Address Customers",
                relationship.get(
                    "address_customer_count",
                    0,
                ),
            )

    if isinstance(explanation, dict):
        source = explanation.get("source")
        llm_role = explanation.get("llm_role")
        decision_source = explanation.get(
            "decision_source"
        )

        if source or llm_role or decision_source:
            st.caption(
                f"Evidence source: {source or 'N/A'} · "
                f"LLM role: {llm_role or 'N/A'} · "
                f"Decision source: {decision_source or 'N/A'}"
            )


def render_ai_fraud_assistant(
    incidents,
    transactions,
    selected_incident=None,
):
    st.title("🤖 AI Fraud Assistant")
    st.caption(
        "Ask questions about incidents, transactions, customers, "
        "fraud signals and recommended actions."
    )

    if selected_incident:
        context = build_incident_context(
            selected_incident
        )
        st.info(
            "AI context: "
            f"{selected_incident.get('incident_id', 'Current incident')}"
        )
    else:
        context = build_dashboard_context(
            incidents
        )
        st.info(
            "AI context: active fraud incidents"
        )

    st.subheader("Suggested Questions")

    q1, q2, q3 = st.columns(3, gap="medium")

    with q1:
        if st.button(
            "Why is this incident critical?",
            key="ai_suggest_critical",
            width="stretch",
        ):
            st.session_state.fraud_pending_question = (
                "Why is this incident considered high risk or critical?"
            )
            st.rerun()

    with q2:
        if st.button(
            "Explain the root cause",
            key="ai_suggest_root",
            width="stretch",
        ):
            st.session_state.fraud_pending_question = (
                "Explain the root cause using the available fraud evidence."
            )
            st.rerun()

    with q3:
        if st.button(
            "What should I do next?",
            key="ai_suggest_action",
            width="stretch",
        ):
            st.session_state.fraud_pending_question = (
                "What should the merchant do next and why?"
            )
            st.rerun()

    st.divider()

    # ------------------------------------------------------------
    # CHAT HISTORY
    # ------------------------------------------------------------
    for message in st.session_state.fraud_chat_messages:
        role = (
            "assistant"
            if message.get("role") == "assistant"
            else "user"
        )

        with st.chat_message(role):
            content = str(message.get("content", ""))
            if role == "assistant":
                safe_content = html.escape(content).replace("\n", "<br>")
                st.markdown(
                    f'<div class="ai-live-response ai-history-response">{safe_content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(content)

    question = st.chat_input(
        "Ask the fraud assistant..."
    )

    if not question:
        question = st.session_state.fraud_pending_question
        st.session_state.fraud_pending_question = None

    if not question:
        return

    # Save and immediately show the user's message.
    st.session_state.fraud_chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # ------------------------------------------------------------
    # LIVE OLLAMA RESPONSE
    # ------------------------------------------------------------
    answer_parts = []

    with st.chat_message("assistant"):
        # This placeholder is updated as Ollama sends each chunk.
        response_placeholder = st.empty()

        # Visible status line so the user knows the local model
        # is actually working while the first token is generated.
        status_placeholder = st.empty()

        try:
            status_placeholder.markdown(
                '<div class="ai-stream-status">'
                '<span class="ai-stream-dot"></span>'
                'Qwen is analyzing the fraud evidence...'
                '</div>',
                unsafe_allow_html=True,
            )

            # Use the conversation BEFORE adding the assistant answer.
            conversation_for_model = (
                st.session_state.fraud_chat_messages[-8:]
            )

            for chunk in stream_ollama(
                question=question,
                context=context,
                conversation_history=conversation_for_model,
            ):
                answer_parts.append(chunk)

                current_answer = "".join(
                    answer_parts
                )

                # The cursor makes it visually obvious that the
                # local model is still generating text.
                safe_answer = html.escape(current_answer + " ▌").replace("\n", "<br>")
                response_placeholder.markdown(
                    f'<div class="ai-live-response">{safe_answer}</div>',
                    unsafe_allow_html=True,
                )

            answer = "".join(
                answer_parts
            ).strip()

            if not answer:
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            # Remove the cursor after generation completes.
            safe_answer = html.escape(answer).replace("\n", "<br>")
            response_placeholder.markdown(
                f'<div class="ai-live-response">{safe_answer}</div>',
                unsafe_allow_html=True,
            )

            status_placeholder.empty()

        except Exception as exc:
            status_placeholder.empty()

            answer = (
                "I couldn't process the request right now. "
                "Check the AI service configuration.\n\n"
                f"Technical detail: `{exc}`"
            )

            safe_answer = html.escape(answer).replace("\n", "<br>")
            response_placeholder.markdown(
                f'<div class="ai-live-response">{safe_answer}</div>',
                unsafe_allow_html=True,
            )

    # Persist the complete answer so it remains visible after
    # Streamlit reruns.
    st.session_state.fraud_chat_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# API STATUS
# ============================================================

def check_api():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=2,
        )

        return (
            response.status_code
            == 200
        )

    except Exception:

        return False


api_online = check_api()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ Risk Sentinel"
)

st.sidebar.caption(
    "Merchant fraud intelligence platform"
)

st.sidebar.divider()


page_options = {
    "🏠  Command Center": "Command Center",
    "▣  Transaction Explorer": "Transaction Explorer",
    "♙  Customer Intelligence": "Customer Intelligence",
    "◈  Model Performance": "Model Performance",
    "◫  Incident Centre": "Incident Centre",
    "◉  Live Risk Simulator": "Live Risk Simulator",
    "✦  AI Fraud Assistant": "AI Fraud Assistant",
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(page_options.keys()),
    key="main_navigation_v2",
)

page = page_options[selected_page]

# ============================================================
# INVESTIGATION / NAVIGATION SCROLL RESET
# ============================================================
# Force the Streamlit content area to the top after a navigation
# rerun, including Investigate clicks where the page name remains
# Command Center but selected_incident_id changes.
st.components.v1.html(
    """
    <script>
    (() => {
        const resetScroll = () => {
            try {
                const doc = window.parent.document;
                const win = window.parent;

                win.scrollTo(0, 0);

                [
                    'section.main',
                    'div[data-testid="stAppViewContainer"]',
                    'div[data-testid="stAppViewBlockContainer"]',
                    'div[data-testid="stMainBlockContainer"]',
                    'div[data-testid="stVerticalBlock"]'
                ].forEach((selector) => {
                    doc.querySelectorAll(selector).forEach((el) => {
                        el.scrollTop = 0;
                        el.scrollLeft = 0;
                        if (typeof el.scrollTo === "function") {
                            el.scrollTo(0, 0);
                        }
                    });
                });

                doc.documentElement.scrollTop = 0;
                doc.body.scrollTop = 0;
            } catch (e) {}
        };

        resetScroll();
        setTimeout(resetScroll, 100);
        setTimeout(resetScroll, 300);
        setTimeout(resetScroll, 600);
        setTimeout(resetScroll, 1000);

        try {
            const observer = new MutationObserver(() => resetScroll());
            observer.observe(doc.body, {
                childList: true,
                subtree: true
            });
            setTimeout(() => observer.disconnect(), 1500);
        } catch (e) {}
    })();
    </script>
    """,
    height=0,
    width=0,
)

# A selected incident always opens in Command Center.
# We change the local routing variable only; we never mutate
# st.session_state["main_navigation"] after the radio is created.
if st.session_state.get("selected_incident_id"):
    page = "Command Center"


# ============================================================
# PAGE NAVIGATION SCROLL RESET
# ============================================================
# Reset the main page to the top ONLY when the selected dashboard
# page changes. This prevents Streamlit from reopening a new page
# at the previous scroll position (for example at the bottom of
# the AI Fraud Assistant) while leaving normal interactions alone.
st.components.v1.html(
    f"""
    <script>
    (() => {{
        const currentPage = {json.dumps(page)} + "::" + {json.dumps(str(st.session_state.get("selected_incident_id") or ""))};
        const storageKey = "merchantRiskSentinelLastPage";

        const resetMainScroll = () => {{
            try {{
                const win = window.parent;
                const doc = win.document;

                win.scrollTo({{ top: 0, left: 0, behavior: "instant" }});

                const selectors = [
                    "section.main",
                    '[data-testid="stAppViewContainer"]',
                    '[data-testid="stAppViewContainer"] > .main',
                    ".main"
                ];

                selectors.forEach((selector) => {{
                    const element = doc.querySelector(selector);
                    if (element) {{
                        element.scrollTo({{
                            top: 0,
                            left: 0,
                            behavior: "instant"
                        }});
                    }}
                }});

                if (doc.documentElement) {{
                    doc.documentElement.scrollTop = 0;
                }}

                if (doc.body) {{
                    doc.body.scrollTop = 0;
                }}
            }} catch (error) {{
                // Ignore browser sandbox restrictions.
            }}
        }};

        let previousPage = null;
        try {{
            previousPage = window.parent.localStorage.getItem(storageKey);
        }} catch (error) {{}}

        if (previousPage !== currentPage) {{
            resetMainScroll();

            // Streamlit finishes replacing the page DOM shortly after
            // the rerun, so repeat the reset after the new page mounts.
            setTimeout(resetMainScroll, 50);
            setTimeout(resetMainScroll, 150);
            setTimeout(resetMainScroll, 300);

            try {{
                window.parent.localStorage.setItem(storageKey, currentPage);
            }} catch (error) {{}}
        }}
    }})();
    </script>
    """,
    height=0,
    width=0,
)

st.sidebar.divider()


st.sidebar.markdown(
    '<div style="font-size:.76rem;font-weight:800;letter-spacing:.06em;'
    'color:#94a3b8 !important;-webkit-text-fill-color:#94a3b8 !important;'
    'margin:1.1rem 0 .65rem;">SYSTEM STATUS</div>',
    unsafe_allow_html=True,
)

def sidebar_status(label, online=True):
    dot = "#22c55e" if online else "#ef4444"
    state = "Online" if online else "Offline"
    st.sidebar.markdown(
        f'''
        <div style="display:flex;align-items:center;justify-content:space-between;
                    gap:.5rem;padding:.28rem 0;">
            <div style="display:flex;align-items:center;gap:.45rem;">
                <span style="width:9px;height:9px;border-radius:50%;
                             background:{dot};display:inline-block;
                             box-shadow:0 0 0 2px rgba(255,255,255,.04);"></span>
                <span style="font-size:.76rem;font-weight:650;
                             color:#f8fafc !important;
                             -webkit-text-fill-color:#f8fafc !important;">
                    {label}
                </span>
            </div>
            <span style="font-size:.68rem;font-weight:800;
                         color:{dot} !important;
                         -webkit-text-fill-color:{dot} !important;">
                {state}
            </span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


sidebar_status("Risk API", api_online)
sidebar_status("Root-Cause Engine", True)
sidebar_status("Incident Engine", True)
sidebar_status("Response Engine", True)
sidebar_status("Detection Engine", True)
sidebar_status("Live Prediction Engine", True)



def check_ollama():
    """Return True when Ollama is reachable."""
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=3,
        )
        return response.status_code == 200
    except Exception:
        return False


ollama_online = check_ollama()

sidebar_status(
    "Ollama (Local LLM)",
    ollama_online,
)



st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:.76rem;font-weight:800;letter-spacing:.06em;'
    'color:#94a3b8 !important;-webkit-text-fill-color:#94a3b8 !important;'
    'margin-bottom:.55rem;">QUICK ACTIONS</div>',
    unsafe_allow_html=True,
)

if st.sidebar.button(
    "＋  Simulate Transaction",
    key="sidebar_simulate_transaction",
    width="stretch",
):
    page = "Live Risk Simulator"

if st.sidebar.button(
    "⇩  Export Report",
    key="sidebar_export_report",
    width="stretch",
):
    st.sidebar.info("Use the export controls in the active section.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:.76rem;font-weight:800;letter-spacing:.06em;'
    'color:#94a3b8 !important;-webkit-text-fill-color:#94a3b8 !important;'
    'margin-bottom:.55rem;">DATA SUMMARY</div>',
    unsafe_allow_html=True,
)

try:
    sidebar_tx_count = len(load_transactions())
except Exception:
    sidebar_tx_count = 0

st.sidebar.markdown(
    f'<div style="font-size:.72rem;color:#cbd5e1 !important;'
    f'-webkit-text-fill-color:#cbd5e1 !important;">Total Transactions</div>'
    f'<div style="font-size:1.45rem;font-weight:800;color:#ffffff !important;'
    f'-webkit-text-fill-color:#ffffff !important;margin:.1rem 0 .55rem;">'
    f'{sidebar_tx_count:,}</div>',
    unsafe_allow_html=True,
)

try:
    sidebar_high_risk = int(
        pd.to_numeric(
            pd.Series(
                [x.get("risk_score", 0) for x in load_incidents()]
            ),
            errors="coerce",
        ).fillna(0).ge(70).sum()
    )
except Exception:
    sidebar_high_risk = 0

st.sidebar.markdown(
    f'<div style="font-size:.72rem;color:#cbd5e1 !important;'
    f'-webkit-text-fill-color:#cbd5e1 !important;">High Risk</div>'
    f'<div style="font-size:1.45rem;font-weight:800;color:#ef4444 !important;'
    f'-webkit-text-fill-color:#ef4444 !important;">{sidebar_high_risk:,}</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ABUSE RISK SENTINEL
# ============================================================

def render_abuse_risk_sentinel(incident, affected):
    """
    Lightweight abuse-ring / relationship sentinel for the
    existing incident investigation page.

    This is intentionally derived only from the transaction records
    already loaded for the selected incident. It does not replace
    the existing ML risk score or modify any existing functionality.
    """
    st.header("🔗 Abuse Risk Sentinel")
    st.caption(
        "Relationship-based abuse-ring analysis using customers, "
        "devices, IPs, addresses and linked transactions."
    )

    if affected is None or affected.empty:
        st.info(
            "No linked transaction data is available to build the "
            "abuse-risk relationship graph."
        )
        return

    # Normalize the relationship fields without changing the source data.
    graph_df = affected.copy()

    def _clean_series(column):
        if column not in graph_df.columns:
            return pd.Series(
                [""] * len(graph_df),
                index=graph_df.index,
                dtype="object",
            )
        return (
            graph_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    graph_df["_customer"] = _clean_series("customer_id")
    graph_df["_device"] = _clean_series("device_id")
    graph_df["_ip"] = _clean_series("ip_id")
    graph_df["_address"] = _clean_series("address_id")
    graph_df["_transaction"] = _clean_series("transaction_id")

    # Remove empty identifiers from relationship analysis.
    for col in [
        "_customer",
        "_device",
        "_ip",
        "_address",
        "_transaction",
    ]:
        graph_df.loc[
            graph_df[col].isin(["", "nan", "None", "NaN"]),
            col,
        ] = ""

    unique_customers = set(
        graph_df.loc[
            graph_df["_customer"] != "",
            "_customer",
        ]
    )
    unique_devices = set(
        graph_df.loc[
            graph_df["_device"] != "",
            "_device",
        ]
    )
    unique_ips = set(
        graph_df.loc[
            graph_df["_ip"] != "",
            "_ip",
        ]
    )
    unique_addresses = set(
        graph_df.loc[
            graph_df["_address"] != "",
            "_address",
        ]
    )
    unique_transactions = set(
        graph_df.loc[
            graph_df["_transaction"] != "",
            "_transaction",
        ]
    )

    # Relationship reuse counts.
    device_reuse = (
        graph_df.loc[
            graph_df["_device"] != "",
            ["_device", "_customer"],
        ]
        .drop_duplicates()
        .groupby("_device")["_customer"]
        .nunique()
        if not graph_df.empty
        else pd.Series(dtype="int64")
    )

    ip_reuse = (
        graph_df.loc[
            graph_df["_ip"] != "",
            ["_ip", "_customer"],
        ]
        .drop_duplicates()
        .groupby("_ip")["_customer"]
        .nunique()
        if not graph_df.empty
        else pd.Series(dtype="int64")
    )

    address_reuse = (
        graph_df.loc[
            graph_df["_address"] != "",
            ["_address", "_customer"],
        ]
        .drop_duplicates()
        .groupby("_address")["_customer"]
        .nunique()
        if not graph_df.empty
        else pd.Series(dtype="int64")
    )

    max_device_reuse = int(
        device_reuse.max()
    ) if not device_reuse.empty else 0

    max_ip_reuse = int(
        ip_reuse.max()
    ) if not ip_reuse.empty else 0

    max_address_reuse = int(
        address_reuse.max()
    ) if not address_reuse.empty else 0

    # ============================================================
    # INCIDENT-SPECIFIC ABUSE-RING SCORE
    # ============================================================
    #
    # Calculate this score only from the currently selected incident.
    # The score is therefore different when the incident's relationship
    # network is different. The visible breakdown below is the same
    # source of truth used for the headline score.

    shared_device_count = int(
        (device_reuse >= 2).sum()
    ) if not device_reuse.empty else 0

    shared_ip_count = int(
        (ip_reuse >= 2).sum()
    ) if not ip_reuse.empty else 0

    shared_address_count = int(
        (address_reuse >= 2).sum()
    ) if not address_reuse.empty else 0

    customer_count = len(unique_customers)
    transaction_count = len(unique_transactions)

    suspicious_count = 0
    if "is_fraud" in graph_df.columns:
        fraud_series = pd.to_numeric(
            graph_df["is_fraud"],
            errors="coerce",
        ).fillna(0)
        suspicious_count = int(
            (fraud_series > 0).sum()
        )

    fraud_ratio = (
        suspicious_count / transaction_count
        if transaction_count > 0
        else 0.0
    )

    # Component scores. Each component has a bounded maximum so that
    # a large incident cannot automatically become 100/100.
    customer_component = min(
        15.0,
        max(
            0,
            customer_count - 1,
        ) * 1.0,
    )

    device_excess = 0.0
    if not device_reuse.empty:
        device_excess = float(
            (
                device_reuse[
                    device_reuse >= 2
                ] - 1
            ).sum()
        )

    device_component = min(
        25.0,
        device_excess * 2.0,
    )

    ip_excess = 0.0
    if not ip_reuse.empty:
        ip_excess = float(
            (
                ip_reuse[
                    ip_reuse >= 2
                ] - 1
            ).sum()
        )

    ip_component = min(
        15.0,
        ip_excess * 2.0,
    )

    address_excess = 0.0
    if not address_reuse.empty:
        address_excess = float(
            (
                address_reuse[
                    address_reuse >= 2
                ] - 1
            ).sum()
        )

    address_component = min(
        10.0,
        address_excess * 1.5,
    )

    import math

    transaction_component = min(
        10.0,
        math.log1p(
            transaction_count
        ) * 2.0,
    )

    fraud_component = min(
        15.0,
        fraud_ratio * 15.0,
    )

    active_relationship_types = sum(
        [
            int(shared_device_count > 0),
            int(shared_ip_count > 0),
            int(shared_address_count > 0),
        ]
    )

    multi_signal_component = min(
        10.0,
        max(
            0,
            active_relationship_types - 1,
        ) * 5.0,
    )

    relationship_score = int(
        round(
            min(
                100.0,
                customer_component
                + device_component
                + ip_component
                + address_component
                + transaction_component
                + fraud_component
                + multi_signal_component,
            )
        )
    )

    # ============================================================
    # RISK BREAKDOWN
    # ============================================================

    score_breakdown = pd.DataFrame(
        [
            {
                "Risk Signal": "Customer connectivity",
                "Evidence": (
                    f"{customer_count:,} connected customers"
                ),
                "Contribution": round(
                    customer_component,
                    1,
                ),
            },
            {
                "Risk Signal": "Device reuse",
                "Evidence": (
                    f"{shared_device_count:,} devices shared by "
                    "multiple customers"
                ),
                "Contribution": round(
                    device_component,
                    1,
                ),
            },
            {
                "Risk Signal": "IP reuse",
                "Evidence": (
                    f"{shared_ip_count:,} IPs shared by "
                    "multiple customers"
                ),
                "Contribution": round(
                    ip_component,
                    1,
                ),
            },
            {
                "Risk Signal": "Address reuse",
                "Evidence": (
                    f"{shared_address_count:,} addresses shared by "
                    "multiple customers"
                ),
                "Contribution": round(
                    address_component,
                    1,
                ),
            },
            {
                "Risk Signal": "Linked transactions",
                "Evidence": (
                    f"{transaction_count:,} linked transactions"
                ),
                "Contribution": round(
                    transaction_component,
                    1,
                ),
            },
            {
                "Risk Signal": "Fraud concentration",
                "Evidence": (
                    f"{suspicious_count:,} suspicious of "
                    f"{transaction_count:,} linked transactions"
                ),
                "Contribution": round(
                    fraud_component,
                    1,
                ),
            },
            {
                "Risk Signal": "Multi-signal overlap",
                "Evidence": (
                    f"{active_relationship_types}/3 relationship "
                    "types reused"
                ),
                "Contribution": round(
                    multi_signal_component,
                    1,
                ),
            },
        ]
    )

    # Reconcile displayed contributions with the headline score after
    # one-decimal rounding.
    displayed_total = float(
        score_breakdown["Contribution"].sum()
    )
    rounding_difference = round(
        relationship_score - displayed_total,
        1,
    )

    if abs(rounding_difference) > 0:
        score_breakdown.loc[
            score_breakdown.index[-1],
            "Contribution",
        ] = round(
            score_breakdown.loc[
                score_breakdown.index[-1],
                "Contribution",
            ] + rounding_difference,
            1,
        )

    if relationship_score >= 75:
        ring_level = "CRITICAL"
        ring_action = "HOLD_AND_INVESTIGATE"
    elif relationship_score >= 50:
        ring_level = "HIGH"
        ring_action = "REVIEW"
    elif relationship_score >= 25:
        ring_level = "MEDIUM"
        ring_action = "MONITOR"
    else:
        ring_level = "LOW"
        ring_action = "ALLOW"

    # ============================================================
    # ABUSE RISK SENTINEL SUMMARY
    # ============================================================

    # These cards were accidentally omitted from the previous version.
    # Restore them before the relationship graph so the Sentinel is
    # visibly complete.
    exposure = 0.0
    if "amount" in graph_df.columns:
        exposure = float(
            pd.to_numeric(
                graph_df["amount"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

    s1, s2, s3, s4, s5, s6 = st.columns(
        6,
        gap="small",
    )

    with s1:
        visible_metric(
            "Abuse Risk",
            f"{relationship_score}/100",
        )

    with s2:
        visible_metric(
            "Risk Level",
            ring_level,
        )

    with s3:
        visible_metric(
            "Recommended Action",
            ring_action,
        )

    with s4:
        visible_metric(
            "Customers",
            f"{len(unique_customers):,}",
        )

    with s5:
        visible_metric(
            "Shared Devices",
            f"{shared_device_count:,}",
        )

    with s6:
        visible_metric(
            "Connected Exposure",
            format_currency(exposure),
        )

    st.markdown("#### Abuse-Ring Risk Breakdown")

    st.dataframe(
        score_breakdown,
        width="stretch",
        hide_index=True,
    )

    if ring_level == "CRITICAL":
        st.error(
            f"**{ring_level} abuse-ring signal** — "
            "strong relationship reuse detected across the "
            "incident-linked transaction network."
        )
    elif ring_level == "HIGH":
        st.warning(
            f"**{ring_level} abuse-ring signal** — "
            "multiple shared entities connect the incident activity."
        )
    elif ring_level == "MEDIUM":
        st.info(
            f"**{ring_level} abuse-ring signal** — "
            "some relationship reuse is present and should be reviewed."
        )
    else:
        st.success(
            f"**{ring_level} abuse-ring signal** — "
            "limited relationship reuse was detected."
        )

    # ------------------------------------------------------------
    # Relationship graph
    # ------------------------------------------------------------
    st.subheader("Abuse-Ring Relationship Graph")
    st.caption(
        "Select a relationship category to inspect the linked entities "
        "and transactions behind the abuse-risk signal."
    )

    # Compact relationship graph. The existing abuse-risk calculations
    # above are unchanged; this section only changes the visualization
    # and adds drill-down details for each relationship category.
    dot_lines = [
        "graph AbuseRiskSentinel {",
        'graph [rankdir=LR, bgcolor="transparent", '
        'pad="0.2", nodesep="0.5", ranksep="0.8", splines=ortho];',
        'node [shape=box, style="rounded,filled", '
        'fontname="Arial", fontsize=10, margin="0.16,0.10"];',
        'edge [color="#94a3b8", penwidth=1.5];',
        f'incident [label="INCIDENT\\n{str(incident.get("incident_id", "N/A"))}", '
        'shape=doublecircle, fillcolor="#fee2e2", color="#dc2626", '
        'fontcolor="#991b1b", penwidth=2];',
        f'customers [label="CUSTOMERS\\n{len(unique_customers):,}", '
        'fillcolor="#dbeafe", color="#2563eb", fontcolor="#1e3a8a"];',
        f'devices [label="SHARED DEVICES\\n{len(unique_devices):,}", '
        'fillcolor="#ede9fe", color="#7c3aed", fontcolor="#4c1d95"];',
        f'ips [label="SHARED IPs\\n{len(unique_ips):,}", '
        'fillcolor="#dcfce7", color="#16a34a", fontcolor="#166534"];',
        f'addresses [label="SHARED ADDRESSES\\n{len(unique_addresses):,}", '
        'fillcolor="#fef3c7", color="#d97706", fontcolor="#92400e"];',
        f'transactions [label="TRANSACTIONS\\n{len(unique_transactions):,}", '
        'fillcolor="#f1f5f9", color="#64748b", fontcolor="#334155"];',
        'incident -- customers [label="linked", color="#dc2626"];',
        'customers -- devices [label="shared", color="#7c3aed"];',
        'customers -- ips [label="shared", color="#16a34a"];',
        'customers -- addresses [label="shared", color="#d97706"];',
        'customers -- transactions [label="activity", color="#64748b"];',
        "}",
    ]

    st.graphviz_chart(
        "\n".join(dot_lines),
        use_container_width=True,
    )

    # ------------------------------------------------------------
    # Interactive relationship drill-down
    # ------------------------------------------------------------
    if "abuse_graph_selection" not in st.session_state:
        st.session_state.abuse_graph_selection = None

    st.markdown("#### Inspect Relationship Details")

    g1, g2, g3, g4 = st.columns(4, gap="small")

    with g1:
        if st.button(
            f"👥 Customers · {len(unique_customers):,}",
            key="abuse_graph_customers",
            use_container_width=True,
        ):
            st.session_state.abuse_graph_selection = "customers"

    with g2:
        if st.button(
            f"📱 Devices · {len(unique_devices):,}",
            key="abuse_graph_devices",
            use_container_width=True,
        ):
            st.session_state.abuse_graph_selection = "devices"

    with g3:
        if st.button(
            f"🌐 IPs · {len(unique_ips):,}",
            key="abuse_graph_ips",
            use_container_width=True,
        ):
            st.session_state.abuse_graph_selection = "ips"

    with g4:
        if st.button(
            f"🏠 Addresses · {len(unique_addresses):,}",
            key="abuse_graph_addresses",
            use_container_width=True,
        ):
            st.session_state.abuse_graph_selection = "addresses"

    selection = st.session_state.abuse_graph_selection

    if selection == "customers":
        customer_detail = (
            graph_df.loc[
                graph_df["_customer"] != "",
                [
                    "_customer",
                    "_device",
                    "_ip",
                    "_address",
                    "_transaction",
                    "amount",
                ],
            ]
            .copy()
        )

        customer_summary = (
            customer_detail.groupby("_customer", dropna=False)
            .agg(
                Transactions=("_transaction", "nunique"),
                Devices=("_device", lambda s: s[s != ""].nunique()),
                IPs=("_ip", lambda s: s[s != ""].nunique()),
                Addresses=("_address", lambda s: s[s != ""].nunique()),
                Exposure=("amount", lambda s: pd.to_numeric(
                    s, errors="coerce"
                ).fillna(0).sum()),
            )
            .sort_values(
                ["Transactions", "Exposure"],
                ascending=False,
            )
            .reset_index()
            .rename(columns={"_customer": "Customer"})
        )

        st.markdown("**Customer relationships linked to this incident**")
        st.dataframe(
            customer_summary,
            use_container_width=True,
            hide_index=True,
        )

    elif selection == "devices":
        device_detail = (
            graph_df.loc[
                graph_df["_device"] != "",
                [
                    "_device",
                    "_customer",
                    "_transaction",
                    "amount",
                ],
            ]
            .copy()
        )

        device_summary = (
            device_detail.groupby("_device", dropna=False)
            .agg(
                Customers=("_customer", lambda s: s[s != ""].nunique()),
                Transactions=("_transaction", lambda s: s[s != ""].nunique()),
                Exposure=("amount", lambda s: pd.to_numeric(
                    s, errors="coerce"
                ).fillna(0).sum()),
            )
            .sort_values(
                ["Customers", "Transactions"],
                ascending=False,
            )
            .reset_index()
            .rename(columns={"_device": "Device"})
        )

        st.markdown("**Shared devices and the customers using them**")
        st.dataframe(
            device_summary,
            use_container_width=True,
            hide_index=True,
        )

    elif selection == "ips":
        ip_detail = (
            graph_df.loc[
                graph_df["_ip"] != "",
                [
                    "_ip",
                    "_customer",
                    "_transaction",
                    "amount",
                ],
            ]
            .copy()
        )

        ip_summary = (
            ip_detail.groupby("_ip", dropna=False)
            .agg(
                Customers=("_customer", lambda s: s[s != ""].nunique()),
                Transactions=("_transaction", lambda s: s[s != ""].nunique()),
                Exposure=("amount", lambda s: pd.to_numeric(
                    s, errors="coerce"
                ).fillna(0).sum()),
            )
            .sort_values(
                ["Customers", "Transactions"],
                ascending=False,
            )
            .reset_index()
            .rename(columns={"_ip": "IP Address"})
        )

        st.markdown("**Shared IP addresses and the customers using them**")
        st.dataframe(
            ip_summary,
            use_container_width=True,
            hide_index=True,
        )

    elif selection == "addresses":
        address_detail = (
            graph_df.loc[
                graph_df["_address"] != "",
                [
                    "_address",
                    "_customer",
                    "_transaction",
                    "amount",
                ],
            ]
            .copy()
        )

        address_summary = (
            address_detail.groupby("_address", dropna=False)
            .agg(
                Customers=("_customer", lambda s: s[s != ""].nunique()),
                Transactions=("_transaction", lambda s: s[s != ""].nunique()),
                Exposure=("amount", lambda s: pd.to_numeric(
                    s, errors="coerce"
                ).fillna(0).sum()),
            )
            .sort_values(
                ["Customers", "Transactions"],
                ascending=False,
            )
            .reset_index()
            .rename(columns={"_address": "Address"})
        )

        st.markdown("**Shared addresses and the customers using them**")
        st.dataframe(
            address_summary,
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------------
    # Evidence summary
    # ------------------------------------------------------------
    evidence_rows = []

    if max_device_reuse > 1:
        evidence_rows.append(
            {
                "Signal": "Device reuse",
                "Value": f"{max_device_reuse} customers on one device",
                "Risk": "High" if max_device_reuse >= 5 else "Medium",
            }
        )

    if max_ip_reuse > 1:
        evidence_rows.append(
            {
                "Signal": "IP reuse",
                "Value": f"{max_ip_reuse} customers on one IP",
                "Risk": "High" if max_ip_reuse >= 5 else "Medium",
            }
        )

    if max_address_reuse > 1:
        evidence_rows.append(
            {
                "Signal": "Address reuse",
                "Value": f"{max_address_reuse} customers on one address",
                "Risk": "High" if max_address_reuse >= 5 else "Medium",
            }
        )

    if suspicious_count > 0:
        evidence_rows.append(
            {
                "Signal": "Suspicious transactions",
                "Value": f"{suspicious_count:,} linked transactions",
                "Risk": "High" if suspicious_count >= 5 else "Medium",
            }
        )

    if evidence_rows:
        st.dataframe(
            pd.DataFrame(evidence_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No strong shared-entity abuse indicators were found "
            "in the available incident-linked records."
        )

    with st.expander("View relationship details"):
        detail_rows = []

        for device, count in device_reuse.sort_values(
            ascending=False
        ).head(10).items():
            detail_rows.append(
                {
                    "Entity Type": "Device",
                    "Entity": str(device),
                    "Connected Customers": int(count),
                }
            )

        for ip, count in ip_reuse.sort_values(
            ascending=False
        ).head(10).items():
            detail_rows.append(
                {
                    "Entity Type": "IP",
                    "Entity": str(ip),
                    "Connected Customers": int(count),
                }
            )

        for address, count in address_reuse.sort_values(
            ascending=False
        ).head(10).items():
            detail_rows.append(
                {
                    "Entity Type": "Address",
                    "Entity": str(address),
                    "Connected Customers": int(count),
                }
            )

        if detail_rows:
            st.dataframe(
                pd.DataFrame(detail_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No relationship reuse details are available.")

# ============================================================
# COMMAND CENTER
# ============================================================


# ============================================================
# LIVE RISK DECISION / HUMAN-IN-THE-LOOP ENGINE
# ============================================================

def calculate_live_risk_decision(prediction, transaction):
    """
    Combine the existing ML probability with deterministic evidence.

    The ML model remains unchanged. This is a transparent policy/risk
    fusion layer used for live decision support:
      - 60% ML probability
      - 40% deterministic evidence

    The merchant/analyst retains final decision authority.
    """

    risk = (
        prediction.get("risk", {})
        if isinstance(prediction, dict)
        else {}
    )

    evidence = (
        prediction.get("evidence", {})
        if isinstance(prediction, dict)
        else {}
    )

    if not isinstance(risk, dict):
        risk = {}

    if not isinstance(evidence, dict):
        evidence = {}

    try:
        ml_probability = float(
            risk.get(
                "fraud_probability",
                0.0,
            )
        )
    except (TypeError, ValueError):
        ml_probability = 0.0

    ml_probability = max(
        0.0,
        min(
            1.0,
            ml_probability,
        ),
    )

    factors = evidence.get(
        "risk_factors",
        [],
    )

    if not isinstance(factors, list):
        factors = []

    # Relationship counts are derived from the same transaction CSV
    # used by the dashboard. This prevents the UI from displaying 0
    # merely because the prediction API omitted a nested relationship
    # payload.
    relationship = {}

    try:
        transactions_df = load_transactions()
    except Exception:
        transactions_df = pd.DataFrame()

    def _count_shared_customers(column_name):
        if (
            not isinstance(transactions_df, pd.DataFrame)
            or transactions_df.empty
            or column_name not in transactions_df.columns
            or "customer_id" not in transactions_df.columns
        ):
            return 0

        target = str(
            transaction.get(
                column_name,
                "",
            )
        ).strip()

        if not target:
            return 0

        work = transactions_df[
            transactions_df[column_name]
            .fillna("")
            .astype(str)
            .str.strip()
            == target
        ]

        return int(
            work["customer_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )

    relationship[
        "device_customer_count"
    ] = _count_shared_customers(
        "device_id"
    )

    relationship[
        "ip_customer_count"
    ] = _count_shared_customers(
        "ip_id"
    )

    relationship[
        "address_customer_count"
    ] = _count_shared_customers(
        "address_id"
    )

    evidence_score = 0.0
    evidence_reasons = []

    # High-confidence deterministic signals.
    ffactor_scores = {
    # Weak behavioral signals
    "Device change": 3,
    "IP change": 4,
    "Location change": 4,

    # Stronger transaction anomalies
    "Amount anomaly": 8,
    "Merchant amount anomaly": 4,

    # Velocity is more indicative of automated/abusive activity
    "High velocity": 8,
    "Very high velocity": 12,

    # Relationship-based signals
    "Shared device": 8,
    "Shared IP address": 8,
    "Shared address": 6,
}
        # ------------------------------------------------------------
    # DETERMINISTIC EVIDENCE WEIGHTS
    # ------------------------------------------------------------
    # A new device is a weak signal by itself.
    # Stronger weights are reserved for behavioral anomalies
    # and suspicious relationships.

    factor_scores = {
        "Device change": 3,
        "IP change": 4,
        "Location change": 4,

        "Amount anomaly": 8,
        "Merchant amount anomaly": 4,

        "High velocity": 8,
        "Very high velocity": 12,

        "Shared device": 8,
        "Shared IP address": 8,
        "Shared address": 6,
    }

    for factor in factors:

        if not isinstance(factor, dict):
            continue

        factor_name = str(
            factor.get(
                "factor",
                "",
            )
        )

        matched_score = factor_scores.get(
            factor_name,
            0,
        )

        if matched_score <= 0:
            continue

        evidence_score += matched_score

        evidence_reasons.append(
            {
                "factor": factor_name,
                "score": matched_score,
                "severity": factor.get(
                    "severity",
                    "",
                ),
                "value": factor.get(
                    "value",
                    "",
                ),
            }
        )

    # Relationship reuse is evaluated from the live relationship counts,
    # rather than from a static incident score.
    relationship_points = 0

    relationship_limits = {
        "device_customer_count": 20,
        "ip_customer_count": 20,
        "address_customer_count": 15,
    }

    for key, maximum_points in relationship_limits.items():

        try:
            count = int(
                relationship.get(
                    key,
                    0,
                ) or 0
            )
        except (TypeError, ValueError):
            count = 0

        if count > 1:
            points = min(
                maximum_points,
                5 + (
                    min(
                        count,
                        20,
                    ) - 1
                ) * 1.5,
            )

            relationship_points += points

    evidence_score += relationship_points

    # Keep deterministic evidence on a 0-100 scale.
    evidence_score = max(
        0.0,
        min(
            100.0,
            evidence_score,
        ),
    )

    # Transparent fusion: preserve the ML model as the primary signal
    # while allowing independent deterministic evidence to increase risk.
    
    ml_score = max(
        0.0,
        min(
            100.0,
            ml_probability * 100.0,
        ),
    )

    evidence_adjustment = min(
        evidence_score * 0.15,
        15.0,
    )

    final_score = min(
        100.0,
        ml_score + evidence_adjustment,
    )

    final_score = max(
        0.0,
        min(
            100.0,
            final_score,
        ),
    )

    if final_score >= 80:
        final_level = "CRITICAL"
        final_action = "HOLD"
    elif final_score >= 60:
        final_level = "HIGH"
        final_action = "HOLD"
    elif final_score >= 40:
        final_level = "MEDIUM"
        final_action = "REVIEW"
    elif final_score >= 20:
        final_level = "LOW"
        final_action = "MONITOR"
    else:
        final_level = "LOW"
        final_action = "ALLOW"

    return {
        "ml_probability": ml_probability,
        "ml_score": round(
            ml_probability * 100.0,
            2,
        ),
        "evidence_score": round(
            evidence_score,
            2,
        ),
        "final_score": round(
            final_score,
            2,
        ),
        "risk_level": final_level,
        "recommended_action": final_action,
        "evidence_reasons": evidence_reasons,
        "relationship_points": round(
            relationship_points,
            2,
        ),
    }


def render_human_in_loop_decision(
    prediction,
    transaction,
):
    """
    Human-in-the-Loop decision layer.

    Separates:
    1. AI/model recommendation
    2. Human operational decision
    3. Confirmed ground truth

    Only confirmed fraud / confirmed legitimate outcomes
    are sent to the continual-learning system.

    INCONCLUSIVE outcomes are audit-only and are NOT
    converted into training labels.
    """

    transaction_id = transaction.get(
        "transaction_id",
        prediction.get("transaction", {}).get(
            "transaction_id",
            "UNKNOWN",
        ),
    )

    risk = prediction.get("risk", {})

    fraud_probability = float(
        risk.get("fraud_probability", 0.0)
    )

    fraud_probability_percent = float(
        risk.get(
            "fraud_probability_percent",
            fraud_probability * 100,
        )
    )

    risk_score = float(
        risk.get("risk_score", 0)
    )

    risk_level = risk.get(
        "risk_level",
        "UNKNOWN",
    )

    recommended_action = risk.get(
        "recommended_action",
        "ALLOW",
    )

    behavioral_features = prediction.get(
        "behavioral_features",
        {},
    )

    evidence = prediction.get(
        "evidence",
        {},
    )

    # ============================================================
    # EXISTING RISK DECISION ENGINE
    # ============================================================

    try:
        decision = calculate_live_risk_decision(
            prediction,
            transaction,
        )
    except Exception:
        decision = {
            "recommended_action": recommended_action,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "fraud_probability": fraud_probability,
            "fraud_probability_percent": fraud_probability_percent,
        }

    if isinstance(decision, dict):
        recommended_action = decision.get(
            "recommended_action",
            recommended_action,
        )

    # ============================================================
    # EXPLAINABLE RISK DECISION SUMMARY
    # ============================================================
    # Present the final decision in the same terms used by the
    # underlying ML + deterministic evidence engine. This gives an
    # analyst an immediate answer to: "Why was this flagged?"
    decision_score = float(
        decision.get(
            "final_score",
            risk_score,
        ) or 0.0
    )
    decision_ml_score = float(
        decision.get(
            "ml_score",
            fraud_probability_percent,
        ) or 0.0
    )
    decision_evidence_score = float(
        decision.get(
            "evidence_score",
            0.0,
        ) or 0.0
    )
    decision_evidence_reasons = decision.get(
        "evidence_reasons",
        [],
    )
    if not isinstance(decision_evidence_reasons, list):
        decision_evidence_reasons = []

    reason_items = []
    for evidence_reason in decision_evidence_reasons[:4]:
        if not isinstance(evidence_reason, dict):
            continue

        factor_name = html.escape(
            str(
                evidence_reason.get(
                    "factor",
                    "Risk signal",
                )
            )
        )
        factor_score = float(
            evidence_reason.get(
                "score",
                0.0,
            ) or 0.0
        )
        reason_items.append(
            f"<li><strong>{factor_name}</strong> "
            f"<span>+{factor_score:.1f} evidence points</span></li>"
        )

    if reason_items:
        reason_html = "".join(reason_items)
    else:
        reason_html = (
            "<li><strong>No deterministic evidence signals</strong> "
            "<span>The ML model is the primary risk signal.</span></li>"
        )

    decision_level_html = html.escape(
        str(
            decision.get(
                "risk_level",
                risk_level,
            )
        )
    )
    decision_action_html = html.escape(
        str(recommended_action)
    )

    # ============================================================
    # EXPLAINABLE RISK DECISION
    # ============================================================
    # Use native Streamlit components here instead of rendering the
    # decision card as one large HTML/Markdown block. This prevents
    # Streamlit from displaying the HTML source literally.
    st.markdown("### 🔎 Explainable Risk Decision")
    st.markdown(
        f"**{decision_level_html} RISK — {decision_action_html}**"
    )
    st.caption("Why this transaction reached the current decision")

    score_col, ml_col, evidence_col = st.columns(3)
    with score_col:
        visible_metric(
            "Final Score",
            f"{decision_score:.0f}/100",
        )
    with ml_col:
        visible_metric(
            "ML Signal",
            f"{decision_ml_score:.1f}/100",
        )
    with evidence_col:
        visible_metric(
            "Evidence Signal",
            f"+{decision_evidence_score:.1f} pts",
        )

    st.markdown("**Key reasons**")
    if decision_evidence_reasons:
        for evidence_reason in decision_evidence_reasons[:4]:
            if not isinstance(evidence_reason, dict):
                continue
            factor_name = str(
                evidence_reason.get("factor", "Risk signal")
            )
            factor_score = float(
                evidence_reason.get("score", 0.0) or 0.0
            )
            st.markdown(
                f"- **{factor_name}** — "
                f"+{factor_score:.1f} evidence points"
            )
    else:
        st.markdown(
            "- **No deterministic evidence signals** — "
            "The ML model is the primary risk signal."
        )

    st.caption(
        "Decision source: ML probability + bounded deterministic evidence. "
        "Analyst retains final operational authority."
    )

    exposure_at_risk = float(
        transaction.get(
            "amount",
            0,
        )
    )

    # ============================================================
    # HEADER
    # ============================================================

    st.markdown(
    """
    <div style="
        padding:18px;
        border-radius:12px;
        border:1px solid rgba(128,128,128,0.25);
        margin-top:20px;
        margin-bottom:15px;
        background:#ffffff;
    ">
        <h3 style="margin:0; color:#111827;">
            🎯 Risk Decision & Human Review
        </h3>
        <p style="margin-top:8px; color:#475569;">
            The ML system provides a recommendation.
            A human analyst provides the final operational decision.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

    # ============================================================
    # AI RECOMMENDATION
    # ============================================================

    # ============================================================
# AI RECOMMENDATION
# ============================================================

    # ============================================================
    # HEADER
    # ============================================================

    st.subheader("🎯 Risk Decision & Human Review")

    st.caption(
        "The ML system provides a recommendation. "
        "A human analyst provides the final operational decision."
    )

    # ============================================================
    # AI RECOMMENDATION
    # ============================================================

    st.markdown("### 🤖 Model Recommendation")

    # Read directly from prediction.
    # This avoids depending on variables outside the function scope.

    display_risk = prediction.get("risk", {}) or {}

    display_fraud_probability = float(
        display_risk.get("fraud_probability", 0.0) or 0.0
    )

    display_fraud_probability_percent = float(
        display_risk.get(
            "fraud_probability_percent",
            display_fraud_probability * 100,
        ) or 0.0
    )

    display_risk_score = float(
        display_risk.get("risk_score", 0.0) or 0.0
    )

    display_risk_level = str(
        display_risk.get("risk_level", "UNKNOWN")
    )

    display_recommended_action = str(
        display_risk.get("recommended_action", "ALLOW")
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        visible_metric(
            "Fraud Probability",
            f"{display_fraud_probability_percent:.2f}%",
        )

    with col2:
        visible_metric(
            "Risk Score",
            f"{display_risk_score:.0f}",
        )

    with col3:
        visible_metric(
            "Risk Level",
            display_risk_level,
        )

    with col4:
        visible_metric(
            "Recommended Action",
            display_recommended_action,
        )

    st.caption(
        "This is the model recommendation, not the final human decision."
    )

    # ============================================================
    # FINANCIAL EXPOSURE
    # ============================================================

    st.markdown("### 💰 Financial Exposure")

    exposure_col1, exposure_col2 = st.columns(2)

    with exposure_col1:
        visible_metric(
            "Transaction Amount",
            f"₹{exposure_at_risk:,.2f}",
        )

    with exposure_col2:
        expected_loss = (
            exposure_at_risk * display_fraud_probability
        )

        visible_metric(
            "Expected Loss",
            f"₹{expected_loss:,.2f}",
        )

    # ============================================================
    # RISK EVIDENCE
    # ============================================================

    st.markdown("### 🔎 Decision Evidence")

    risk_factors = evidence.get(
        "risk_factors",
        [],
    )

    relationship_evidence = evidence.get(
        "relationship_evidence",
        {},
    )

    if not isinstance(relationship_evidence, dict):
        relationship_evidence = {}

    if risk_factors:
        st.warning(
            f"{len(risk_factors)} risk factor(s) detected."
        )

        for factor in risk_factors:
            if isinstance(factor, dict):
                factor_name = factor.get(
                    "name",
                    factor.get(
                        "factor",
                        "Risk factor",
                    ),
                )

                factor_description = factor.get(
                    "description",
                    factor.get(
                        "reason",
                        "",
                    ),
                )

                st.markdown(
                    f"**{factor_name}**"
                )

                if factor_description:
                    st.caption(
                        factor_description
                    )

            else:
                st.markdown(
                    f"- {factor}"
                )
    else:
        st.success(
            "No deterministic risk factors were triggered."
        )

    # ============================================================
    # RELATIONSHIP EVIDENCE
    # ============================================================

    if relationship_evidence:
        st.markdown(
            "#### 🔗 Relationship Evidence"
        )

        rel_col1, rel_col2, rel_col3 = st.columns(3)

        with rel_col1:
            visible_metric(
                "Customers / Device",
                str(
                    relationship_evidence.get(
                        "device_customer_count",
                        0,
                    )
                ),
            )

        with rel_col2:
            visible_metric(
                "Customers / IP",
                str(
                    relationship_evidence.get(
                        "ip_customer_count",
                        0,
                    )
                ),
            )

        with rel_col3:
            visible_metric(
                "Customers / Address",
                str(
                    relationship_evidence.get(
                        "address_customer_count",
                        0,
                    )
                ),
            )

    # ============================================================
    # CONTINUAL LEARNING STATUS
    # ============================================================

    st.markdown("### 🧠 Continual Learning")

    try:
        current_learning_status = learning_status()

        total_feedback = int(
            current_learning_status.get(
                "total_feedback",
                0,
            )
        )

        confirmed_fraud = int(
            current_learning_status.get(
                "confirmed_fraud",
                0,
            )
        )

        confirmed_legitimate = int(
            current_learning_status.get(
                "confirmed_legitimate",
                0,
            )
        )

        minimum_feedback = int(
            current_learning_status.get(
                "minimum_feedback_for_retraining",
                10,
            )
        )

        active_model = current_learning_status.get(
            "active_model"
        )

    except Exception:
        total_feedback = 0
        confirmed_fraud = 0
        confirmed_legitimate = 0
        minimum_feedback = 10
        active_model = None

            # ============================================================
    # CONTINUAL LEARNING METRICS
    # ============================================================

    learning_cols = st.columns(4)

    with learning_cols[0]:
        visible_metric(
            "Confirmed Labels",
            str(total_feedback),
        )

    with learning_cols[1]:
        visible_metric(
            "Confirmed Fraud",
            str(confirmed_fraud),
        )

    with learning_cols[2]:
        visible_metric(
            "Confirmed Legitimate",
            str(confirmed_legitimate),
        )

    with learning_cols[3]:
        visible_metric(
            "Active Model",
            str(active_model) if active_model else "Current",
        )

    remaining_labels = max(
        minimum_feedback - total_feedback,
        0,
    )

    if remaining_labels > 0:
        st.info(
            f"Continual learning needs "
            f"{remaining_labels} more confirmed label(s) "
            f"before the next retraining evaluation."
        )
    else:
        st.success(
            "Retraining threshold reached. "
            "The maintenance worker can evaluate a candidate model."
        )

    # ============================================================
    # HUMAN DECISION
    # ============================================================

    st.markdown("### 👤 Analyst Decision")

    st.info(
        "Choose the operational action independently from the model recommendation."
    )

    form_key = (
        f"human_decision_form_{transaction_id}"
    )

    action_options = [
        "ALLOW",
        "REVIEW",
        "HOLD",
    ]

    default_index = (
        action_options.index(recommended_action)
        if recommended_action in action_options
        else 1
    )

    with st.form(form_key):

        human_decision = st.radio(
            "Final operational action",
            action_options,
            index=default_index,
            horizontal=True,
        )

        decision_reason = st.text_area(
            "Analyst decision reason",
            placeholder=(
                "Explain why you accepted, reviewed, "
                "or held this transaction."
            ),
        )

        # ========================================================
        # GROUND TRUTH
        # ========================================================

        st.markdown("### 🏷️ Ground Truth")

        st.caption(
            "Ground truth is separate from the operational decision. "
            "Only confirmed outcomes are used for model learning."
        )

        ground_truth = st.radio(
            "What is the final outcome of this transaction?",
            [
                "INCONCLUSIVE",
                "CONFIRMED_FRAUD",
                "CONFIRMED_LEGITIMATE",
            ],
            index=0,
        )

        ground_truth_reason = st.text_area(
            "Ground-truth evidence / investigation notes",
            placeholder=(
                "Example: confirmed by customer, "
                "chargeback received, manual investigation, etc."
            ),
        )

        submitted = st.form_submit_button(
            "Submit Decision",
            use_container_width=True,
        )

    # ============================================================
    # SUBMISSION
    # ============================================================

    if submitted:

        timestamp = datetime.utcnow().isoformat()

        # ========================================================
        # OPERATIONAL AUDIT RECORD
        # ========================================================

        audit_record = {
            "timestamp": timestamp,
            "transaction_id": transaction_id,
            "model_recommendation": recommended_action,
            "model_risk_level": risk_level,
            "model_risk_score": risk_score,
            "fraud_probability": fraud_probability,
            "fraud_probability_percent": (
                fraud_probability_percent
            ),
            "human_decision": human_decision,
            "decision_reason": decision_reason,
            "ground_truth": ground_truth,
            "ground_truth_reason": ground_truth_reason,
        }

        if "risk_decision_audit" not in st.session_state:
            st.session_state[
                "risk_decision_audit"
            ] = []

        st.session_state[
            "risk_decision_audit"
        ].append(
            audit_record
        )

        # ========================================================
        # CONTINUAL LEARNING FEEDBACK
        # ========================================================
        #
        # Operational action is NOT ground truth.
        #
        # HOLD does not automatically mean fraud.
        # REVIEW does not automatically mean fraud.
        # ALLOW does not automatically mean legitimate.
        #
        # Only confirmed ground truth enters training.
        #
        # INCONCLUSIVE is intentionally excluded.
        # ========================================================

        feedback_submitted = False

        if ground_truth in [
            "CONFIRMED_FRAUD",
            "CONFIRMED_LEGITIMATE",
        ]:

            label = (
                1
                if ground_truth == "CONFIRMED_FRAUD"
                else 0
            )

            feedback_payload = {
    "transaction_id": transaction_id,
    "label": label,
    "ground_truth": ground_truth,
    "ai_recommendation": recommended_action,
    "human_decision": human_decision,
    "final_decision": human_decision,
    "reason": (
        ground_truth_reason
        or decision_reason
        or "Analyst-confirmed outcome"
    ),
    "transaction": transaction,
    "features": behavioral_features,
}

            try:
                feedback_response = requests.post(
                    f"{API_URL}/feedback",
                    json=feedback_payload,
                    timeout=30,
                )

                feedback_response.raise_for_status()

                feedback_submitted = True

            except Exception as exc:
                st.error(
                    "Could not submit feedback to the "
                    f"continual-learning service: {exc}"
                )

        # ========================================================
        # RESULT
        # ========================================================

        if feedback_submitted:

            if ground_truth == "CONFIRMED_FRAUD":
                st.success(
                    "Confirmed fraud label recorded. "
                    "The transaction is now available for continual learning."
                )
            else:
                st.success(
                    "Confirmed legitimate label recorded. "
                    "The transaction is now available for continual learning."
                )

        elif ground_truth == "INCONCLUSIVE":

            st.info(
                "Operational decision recorded. "
                "Because the outcome is inconclusive, "
                "this transaction will NOT be used for model training."
            )

        # ========================================================
        # AI VS HUMAN VS GROUND TRUTH
        # ========================================================

        st.markdown(
            "### 📊 AI vs Human vs Ground Truth"
        )

        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

        with comparison_col1:
            st.markdown(
                "**🤖 AI Recommendation**"
            )

            st.write(
                recommended_action
            )

        with comparison_col2:
            st.markdown(
                "**👤 Human Decision**"
            )

            st.write(
                human_decision
            )

        with comparison_col3:
            st.markdown(
                "**🏷️ Ground Truth**"
            )

            st.write(
                ground_truth
            )

        # ========================================================
        # REFRESH LEARNING STATUS
        # ========================================================

        try:

            updated_learning_status = learning_status()

            updated_feedback_count = int(
                updated_learning_status.get(
                    "total_feedback",
                    0,
                )
            )

            updated_minimum = int(
                updated_learning_status.get(
                    "minimum_feedback_for_retraining",
                    10,
                )
            )

            if updated_feedback_count >= updated_minimum:

                st.warning(
                    "Retraining threshold reached. "
                    "The maintenance worker will evaluate "
                    "a candidate model."
                )

            else:

                st.caption(
                    f"Continual-learning progress: "
                    f"{updated_feedback_count}/"
                    f"{updated_minimum} confirmed labels."
                )

        except Exception:
            pass

        # ========================================================
        # STORE LAST DECISION
        # ========================================================

        st.session_state[
            "last_human_decision"
        ] = audit_record

    # ============================================================
    # SESSION AUDIT TRAIL
    # ============================================================

    audit_history = st.session_state.get(
        "risk_decision_audit",
        [],
    )

    if audit_history:

        st.markdown(
            "### 📋 Decision Audit Trail"
        )

        audit_df = pd.DataFrame(
            audit_history
        )

        display_columns = [
            "timestamp",
            "transaction_id",
            "model_recommendation",
            "human_decision",
            "ground_truth",
            "model_risk_level",
            "fraud_probability",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in audit_df.columns
        ]

        if available_columns:

            st.dataframe(
                audit_df[available_columns],
                use_container_width=True,
                hide_index=True,
            )



def find_related_transactions(
    transaction,
    transactions_df=None,
):
    """
    Return existing dataset rows connected to the current transaction
    through the same customer, merchant, device, IP, or address.

    The current transaction is excluded. Each row receives a Match Reason
    so the relationship is visible directly in the transaction dataset.
    """

    if transactions_df is None:
        transactions_df = load_transactions()

    if transactions_df is None or transactions_df.empty:
        return pd.DataFrame()

    df = transactions_df.copy()

    relationship_columns = [
        ("customer_id", "Customer"),
        ("merchant_id", "Merchant"),
        ("device_id", "Device"),
        ("ip_id", "IP"),
        ("address_id", "Address"),
    ]

    current_id = str(
        transaction.get(
            "transaction_id",
            "",
        )
        or ""
    ).strip().lower()

    related_mask = pd.Series(
        False,
        index=df.index,
    )

    match_masks = []

    for column, label in relationship_columns:
        if column not in df.columns:
            continue

        current_value = str(
            transaction.get(
                column,
                "",
            )
            or ""
        ).strip().lower()

        if not current_value:
            continue

        values = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = values == current_value
        related_mask = related_mask | mask
        match_masks.append(
            (
                label,
                mask,
            )
        )

    related = df.loc[
        related_mask
    ].copy()

    if related.empty:
        return pd.DataFrame()

    if "transaction_id" in related.columns:
        related = related[
            related[
                "transaction_id"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            != current_id
        ].copy()

    if related.empty:
        return pd.DataFrame()

    def build_match_reason(row):
        reasons = []

        for label, mask in match_masks:
            if bool(
                mask.get(
                    row.name,
                    False,
                )
            ):
                reasons.append(
                    f"Same {label}"
                )

        return ", ".join(reasons)

    related[
        "Match Reason"
    ] = related.apply(
        build_match_reason,
        axis=1,
    )

    preferred_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "address_id",
        "amount",
        "timestamp",
        "risk_score",
        "risk_level",
        "recommended_action",
        "Match Reason",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in related.columns
    ]

    return related[
        display_columns
    ].reset_index(
        drop=True
    )


def render_related_transactions(
    transaction,
):
    current_id = str(
    transaction.get("transaction_id", "")
).strip().lower()
    """
    Display all dataset transactions connected to the current transaction.
    """

    st.markdown(
        "### 🔗 Related Transactions in Dataset"
    )

    st.caption(
        "Existing transactions that share the same customer, merchant, "
        "device, IP address or address are shown here."
    )

    current_transactions = load_transactions()

    related = find_related_transactions(
        transaction,
        current_transactions,
    )

    if related.empty:
        st.success(
            "No other transactions share the customer, merchant, "
            "device, IP or address."
        )
        return

    st.info(
        f"{len(related):,} related transaction(s) found in "
        "transactions.csv."
    )

    st.dataframe(
        related,
        width="stretch",
        hide_index=True,
    )

    # Relationship counts exclude the current transaction.
    count_definitions = [
        ("customer_id", "Same Customer"),
        ("merchant_id", "Same Merchant"),
        ("device_id", "Same Device"),
        ("ip_id", "Same IP"),
        ("address_id", "Same Address"),
    ]

    count_columns = st.columns(
        len(count_definitions),
        gap="small",
    )

    for column, (field, label) in zip(
        count_columns,
        count_definitions,
    ):
        current_value = str(
            transaction.get(
                field,
                "",
            )
            or ""
        ).strip().lower()

        count = 0

        if (
            current_value
            and field in current_transactions.columns
        ):
            values = (
                current_transactions[field]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            count = int(
                (
                    values == current_value
                ).sum()
            )

            # Exclude the transaction being assessed.
            if "transaction_id" in current_transactions.columns:
                current_ids = (
                    current_transactions[
                        "transaction_id"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                if current_id in set(
                    current_ids[
                        values == current_value
                    ]
                ):
                    count = max(
                        0,
                        count - 1,
                    )

        with column:
            visible_metric(
                label,
                f"{count:,}",
            )


if page == "Command Center":

    # ========================================================
    # ACTIVE MODEL -> EXISTING INCIDENTS
    # ========================================================
    # Incident records can pre-date the currently active model. Surface
    # that state and let the analyst refresh all existing incidents through
    # the same prediction path used by newly entered transactions.
    incidents_for_sync = load_incidents()
    active_model_version = get_active_model_version()
    stale_incident_count = sum(
        1
        for item in incidents_for_sync
        if active_model_version
        and str(item.get("model_version", "")).strip()
        != active_model_version
    )

    if stale_incident_count > 0:
        st.info(
            f"{stale_incident_count} existing incident(s) are using historical "
            "risk scores. Re-score them with the active continual-learning model "
            "to bring the Incident Centre up to date."
        )

        if st.button(
            "🔄 Re-score Existing Incidents",
            key="rescore_existing_incidents",
            width="stretch",
        ):
            with st.spinner(
                f"Re-scoring {stale_incident_count} incident(s) with "
                f"model {active_model_version}..."
            ):
                sync_result = rescore_existing_incidents_with_active_model(
                    incidents_for_sync,
                    load_transactions(),
                )

            if sync_result.get("status") == "success":
                st.success(
                    f"Updated {sync_result['updated']} incident(s) with active "
                    f"model {sync_result['model_version']}. "
                    f"Skipped {sync_result['skipped']} already-current incident(s)."
                )
                if sync_result.get("failed", 0):
                    st.warning(
                        f"{sync_result['failed']} incident(s) could not be re-scored "
                        "because their source transaction was unavailable or invalid."
                    )
                st.rerun()
            else:
                st.error(
                    sync_result.get(
                        "message",
                        "Existing incidents could not be re-scored.",
                    )
                )

    selected_incident = (
        get_selected_incident()
    )

    # ========================================================
    # INCIDENT INVESTIGATION VIEW
    # ========================================================

    if selected_incident is not None:

        incident = selected_incident

        incident_id = str(
            incident.get(
                "incident_id",
                "N/A",
            )
        )

        incident_type = pretty_type(
            incident.get(
                "incident_type",
                "UNKNOWN",
            )
        )

        severity = str(
            incident.get(
                "severity",
                "MEDIUM",
            )
        ).upper()

        try:

            risk_score = int(
                float(
                    incident.get(
                        "risk_score",
                        0,
                    )
                )
            )

        except Exception:

            risk_score = 0

        transaction_count = int(
            float(
                incident.get(
                    "transaction_count",
                    0,
                )
            )
        )

        customer_count = int(
            float(
                incident.get(
                    "customer_count",
                    0,
                )
            )
        )

        estimated_exposure = float(
            incident.get(
                "estimated_exposure",
                0,
            )
        )

        # ====================================================
        # BACK BUTTON
        # ====================================================

        if st.button(
            "← Back to Command Center",
            key="back_to_command_center",
        ):

            st.session_state[
                "selected_incident_id"
            ] = None

            st.rerun()

        # ====================================================
        # HEADER
        # ====================================================

        severity_class = (
            "critical-badge"
            if severity == "CRITICAL"
            else
            "high-badge"
            if severity == "HIGH"
            else
            "medium-badge"
            if severity == "MEDIUM"
            else
            "low-badge"
        )

        severity_icon = (
            "🔴"
            if severity == "CRITICAL"
            else
            "🟠"
            if severity == "HIGH"
            else
            "🟡"
            if severity == "MEDIUM"
            else
            "🟢"
        )

        # Use native Streamlit elements here instead of raw HTML.
        # This prevents Streamlit from ever displaying the HTML source
        # as literal text and keeps the investigation header readable.
        severity_color = {
            "CRITICAL": "red",
            "HIGH": "orange",
            "MEDIUM": "orange",
            "LOW": "green",
        }.get(severity, "gray")

        st.markdown(
            f"## {severity_icon} {severity}"
        )
        st.title(incident_id)
        st.subheader(incident_type)
        st.caption(
            "Complete incident investigation, transaction evidence "
            "and response analysis."
        )

        # ====================================================
        # INCIDENT METRICS
        # ====================================================

        m1, m2, m3, m4 = st.columns(
            4,
            gap="medium",
        )

        with m1:

            visible_metric(
                "Risk Score",
                f"{risk_score}/100",
            )

        with m2:

            visible_metric(
                "Transactions",
                f"{transaction_count:,}",
            )

        with m3:

            visible_metric(
                "Customers",
                f"{customer_count:,}",
            )

        with m4:

            visible_metric(
                "Potential Exposure",
                format_currency(
                    estimated_exposure
                ),
            )

        st.divider()

        # ====================================================
        # LOAD TRANSACTIONS FOR INVESTIGATION
        # ====================================================

        transactions = (
            prepare_transaction_data(
                load_transactions(),
                load_incidents(),
            )
        )

        affected = (
            get_incident_transactions(
                incident,
                transactions,
            )
        )

        # ====================================================
        # AFFECTED TRANSACTIONS
        # ====================================================

        st.header(
            "Affected Transactions"
        )

        if affected.empty:

            st.info(
                "No transaction records were found "
                "for this incident."
            )

        else:

            st.caption(
                f"{len(affected):,} transactions "
                "linked to this incident."
            )

            st.dataframe(
                format_transaction_table(
                    affected.head(500)
                ),
                width="stretch",
                hide_index=True,
            )

        st.divider()

        # ====================================================
        # FRAUD SIGNALS
        # ====================================================

        st.header(
            "Fraud Signals"
        )

        raw_fraud_types = incident.get(
            "fraud_types"
        )

        fraud_types = parse_json(
            raw_fraud_types
        )

        fraud_counts = {}

        # ----------------------------------------------------
        # CASE 1:
        # fraud_types is a dictionary
        # ----------------------------------------------------

        if isinstance(
            fraud_types,
            dict,
        ):

            for key, value in (
                fraud_types.items()
            ):

                try:

                    count = int(
                        float(value)
                    )

                except Exception:

                    count = 0

                if count > 0:

                    fraud_counts[
                        str(key)
                    ] = count

        # ----------------------------------------------------
        # CASE 2:
        # fraud_types is a list
        # ----------------------------------------------------

        elif isinstance(
            fraud_types,
            list,
        ):

            for item in fraud_types:

                if isinstance(
                    item,
                    dict,
                ):

                    fraud_name = str(
                        item.get(
                            "fraud_type",
                            item.get(
                                "type",
                                "UNKNOWN",
                            ),
                        )
                    )

                    try:

                        count = int(
                            float(
                                item.get(
                                    "count",
                                    item.get(
                                        "transactions",
                                        1,
                                    ),
                                )
                            )
                        )

                    except Exception:

                        count = 1

                    fraud_counts[
                        fraud_name
                    ] = (
                        fraud_counts.get(
                            fraud_name,
                            0,
                        )
                        + count
                    )

                else:

                    fraud_name = str(
                        item
                    ).strip()

                    if fraud_name:

                        fraud_counts[
                            fraud_name
                        ] = (
                            fraud_counts.get(
                                fraud_name,
                                0,
                            )
                            + 1
                        )

        # ----------------------------------------------------
        # FALLBACK:
        # derive directly from transactions
        # ----------------------------------------------------

        if (
            not fraud_counts
            and not affected.empty
            and "fraud_type"
            in affected.columns
        ):

            transaction_fraud_types = (
                affected[
                    "fraud_type"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            transaction_fraud_types = (
                transaction_fraud_types[
                    ~transaction_fraud_types
                    .isin(
                        [
                            "",
                            "nan",
                            "None",
                            "normal",
                        ]
                    )
                ]
            )

            counts = (
                transaction_fraud_types
                .value_counts()
                .to_dict()
            )

            fraud_counts = {
                str(key): int(value)
                for key, value
                in counts.items()
                if int(value) > 0
            }

        if fraud_counts:

            fraud_df = pd.DataFrame(
                [
                    {
                        "Fraud Type":
                            pretty_type(
                                fraud_type
                            ),
                        "Transactions":
                            count,
                    }
                    for fraud_type, count
                    in sorted(
                        fraud_counts.items(),
                        key=lambda item:
                            item[1],
                        reverse=True,
                    )
                ]
            )

            st.dataframe(
                fraud_df,
                width="stretch",
                hide_index=True,
            )

            total_signal_transactions = int(
                fraud_df[
                    "Transactions"
                ].sum()
            )

            st.caption(
                f"{len(fraud_df)} fraud signal type(s) "
                f"· {total_signal_transactions:,} "
                "signal transactions"
            )

        else:

            primary_type = str(
                incident.get(
                    "incident_type",
                    "",
                )
            ).strip()

            if primary_type:

                st.info(
                    "Primary fraud signal: "
                    f"{pretty_type(primary_type)}"
                )

            else:

                st.warning(
                    "No fraud signal data is available "
                    "for this incident."
                )

        st.divider()

        # ====================================================
        # INCIDENT ANALYSIS
        # ====================================================

        st.header(
            "Incident Analysis"
        )

        analysis_col1, analysis_col2 = (
            st.columns(
                2,
                gap="large",
            )
        )

        with analysis_col1:

            st.subheader(
                "Incident Profile"
            )

            first_seen = incident.get(
                "first_seen",
                "N/A",
            )

            last_seen = incident.get(
                "last_seen",
                "N/A",
            )

            duration = incident.get(
                "duration_minutes",
                0,
            )

            st.markdown(
                f"""
                <div class="section-card">

                <div class="incident-stat">
                    <b>Incident ID:</b>
                    {incident_id}
                </div>

                <div class="incident-stat">
                    <b>Incident Type:</b>
                    {incident_type}
                </div>

                <div class="incident-stat">
                    <b>Severity:</b>
                    {severity}
                </div>

                <div class="incident-stat">
                    <b>First Seen:</b>
                    {first_seen}
                </div>

                <div class="incident-stat">
                    <b>Last Seen:</b>
                    {last_seen}
                </div>

                <div class="incident-stat">
                    <b>Duration:</b>
                    {duration} minutes
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with analysis_col2:

            st.subheader(
                "Risk Metrics"
            )

            average_risk = incident.get(
                "average_risk_score",
                0,
            )

            fraud_transactions = incident.get(
                "fraud_transactions",
                0,
            )

            fraud_rate = incident.get(
                "fraud_rate",
                0,
            )

            if isinstance(
                fraud_rate,
                (int, float),
            ):

                fraud_rate_display = (
                    fraud_rate * 100
                    if fraud_rate <= 1
                    else fraud_rate
                )

            else:

                fraud_rate_display = 0

            st.markdown(
                f"""
                <div class="section-card">

                <div class="incident-stat">
                    <b>Maximum Risk:</b>
                    {risk_score}/100
                </div>

                <div class="incident-stat">
                    <b>Average Risk:</b>
                    {average_risk}
                </div>

                <div class="incident-stat">
                    <b>Fraud Transactions:</b>
                    {format_number(fraud_transactions)}
                </div>

                <div class="incident-stat">
                    <b>Fraud Rate:</b>
                    {fraud_rate_display:.2f}%
                </div>

                <div class="incident-stat">
                    <b>Estimated Exposure:</b>
                    {format_currency(estimated_exposure)}
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ====================================================
        # ROOT CAUSES
        # ====================================================

        st.header(
            "Root Cause Analysis"
        )

        root_causes = parse_json(
            incident.get(
                "root_causes"
            )
        )

        if root_causes:

            if isinstance(
                root_causes,
                dict,
            ):

                for key, value in (
                    root_causes.items()
                ):

                    st.markdown(
                        f"**{pretty_type(key)}**\n\n{value}"
                    )
                    st.divider()

            else:

                for cause in root_causes:

                    st.markdown(
                        f"• {cause}"
                    )

        else:

            st.info(
                "No detailed root-cause records "
                "were generated for this incident."
            )

        st.divider()

        # ====================================================
        # ABUSE RISK SENTINEL
        # ====================================================
        render_abuse_risk_sentinel(
            incident,
            affected,
        )

        # ====================================================
        # CONTINUAL LEARNING / HUMAN REVIEW
        # ====================================================
        # Existing incidents must have the same analyst feedback loop as
        # newly entered transactions. The analyst reviews a representative
        # transaction from this incident, confirms the ground truth, and
        # the confirmed label is sent to the same /feedback endpoint.
        # INCONCLUSIVE remains audit-only.

        st.header(
            "🧠 Continual Learning & Human Review"
        )

        st.caption(
            "Review this incident's representative transaction, compare the AI recommendation "
            "with your investigation, and record the confirmed outcome. Confirmed fraud or "
            "legitimate outcomes become continual-learning labels; inconclusive outcomes do not."
        )

        learning_transaction = None
        learning_prediction = None

        if affected is not None and not affected.empty:
            source_transaction_id = str(
                incident.get(
                    "source_transaction_id",
                    "",
                )
            ).strip()

            candidate_rows = affected

            if source_transaction_id and "transaction_id" in affected.columns:
                source_match = affected[
                    affected["transaction_id"]
                    .astype(str)
                    .str.strip()
                    == source_transaction_id
                ]

                if not source_match.empty:
                    candidate_rows = source_match

            learning_row = candidate_rows.iloc[0]
            learning_transaction = _build_existing_transaction_payload(
                learning_row
            )

            active_learning_model = get_active_model_version()
            learning_cache_key = (
                incident_id,
                learning_transaction.get("transaction_id", ""),
                active_learning_model,
            )

            cached_learning = st.session_state.get(
                "incident_learning_predictions",
                {},
            )

            if learning_cache_key in cached_learning:
                learning_prediction = cached_learning[learning_cache_key]
            else:
                try:
                    with st.spinner(
                        "Running the active continual-learning model on this incident..."
                    ):
                        learning_prediction = predict_new_transaction(
                            learning_transaction
                        )

                    cached_learning[learning_cache_key] = learning_prediction
                    st.session_state[
                        "incident_learning_predictions"
                    ] = cached_learning

                except Exception as exc:
                    st.error(
                        "Could not run the active model for incident feedback: "
                        f"{exc}"
                    )
                    learning_prediction = None

        if learning_prediction is not None and learning_transaction is not None:
            st.info(
                f"Continual-learning review is attached to transaction "
                f"**{learning_transaction.get('transaction_id', 'N/A')}** "
                f"from incident **{incident_id}**. The active model is evaluated "
                "before the analyst records the final outcome."
            )

            render_human_in_loop_decision(
                learning_prediction,
                learning_transaction,
            )

        elif affected is None or affected.empty:
            st.warning(
                "Continual-learning feedback cannot be opened because this incident "
                "has no linked transaction record."
            )

        st.divider()

        # ====================================================
        # RESPONSE ACTIONS
        # ====================================================

        st.header(
            "Recommended Response"
        )

        response_df = (
            load_response_actions()
        )

        response_for_incident = (
            pd.DataFrame()
        )

        if (
            not response_df.empty
            and "incident_id"
            in response_df.columns
        ):

            response_for_incident = (
                response_df[
                    response_df[
                        "incident_id"
                    ]
                    .astype(str)
                    .isin(
                        incident_id_variants(
                            incident_id
                        )
                    )
                ]
            )

        if not response_for_incident.empty:

            for _, action in (
                response_for_incident.iterrows()
            ):

                action_text = str(
                    action.get(
                        "recommended_action",
                        action.get(
                            "action",
                            "Review incident",
                        ),
                    )
                )

                st.info(
                    f"**Recommended Action**\n\n{action_text}"
                )

        else:

            # Sensible incident-specific fallback.
            if severity == "CRITICAL":

                actions = [
                    "Temporarily restrict high-risk transactions.",
                    "Review all linked customer accounts.",
                    "Review shared device and IP indicators.",
                    "Escalate the incident to fraud operations.",
                ]

            elif severity == "HIGH":

                actions = [
                    "Increase monitoring on linked accounts.",
                    "Review suspicious transactions.",
                    "Apply additional verification where appropriate.",
                ]

            else:

                actions = [
                    "Continue monitoring linked transactions.",
                    "Review suspicious activity for escalation.",
                ]

            for action in actions:

                st.info(
                    f"**Recommended Action**\n\n{action}"
                )

        st.divider()

        # ====================================================
        # INVESTIGATION ACTIONS
        # ====================================================

        st.header(
            "Investigation Actions"
        )

        action_col1, action_col2, action_col3 = (
            st.columns(
                3,
                gap="medium",
            )
        )

        with action_col1:

            if st.button(
                "🔒 Restrict Risk",
                key=f"restrict_{incident_id}",
                width="stretch",
            ):

                st.success(
                    f"Risk restriction initiated for {incident_id}."
                )

        with action_col2:

            if st.button(
                "👤 Review Customers",
                key=f"customers_{incident_id}",
                width="stretch",
            ):

                st.info(
                    "Customer review queue created."
                )

        with action_col3:

            if st.button(
                "🚨 Escalate Incident",
                key=f"escalate_{incident_id}",
                width="stretch",
            ):

                st.success(
                    f"{incident_id} escalated to fraud operations."
                )

        st.divider()

        # ====================================================
        # AI ASSISTANT FOR THIS INCIDENT
        # ====================================================

        render_ai_fraud_assistant(
            load_incidents(),
            transactions,
            selected_incident=incident,
        )

        # IMPORTANT:
        # Do NOT render Command Center again here.
        # The investigation is intentionally the entire page.
    # ========================================================

    # ========================================================
    # NORMAL COMMAND CENTER VIEW
    # ========================================================

    else:

        incidents = load_incidents()

        st.title("Merchant Risk Sentinel")
        st.caption(
            "AI-powered fraud early warning and incident response"
        )

        if not incidents:

            st.warning(
                "No fraud incidents are currently available."
            )

        else:

            incident_df = pd.DataFrame(incidents)

            numeric_columns = [
                "risk_score",
                "average_risk_score",
                "transaction_count",
                "fraud_transactions",
                "customer_count",
                "device_count",
                "ip_count",
                "address_count",
                "total_transaction_amount",
                "estimated_exposure",
                "duration_minutes",
            ]

            for column in numeric_columns:
                if column in incident_df.columns:
                    incident_df[column] = (
                        pd.to_numeric(
                            incident_df[column],
                            errors="coerce",
                        )
                        .fillna(0)
                    )

            total_incidents = len(incident_df)

            if "severity" in incident_df.columns:
                severity_series = (
                    incident_df["severity"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )
                critical_incidents = int(
                    (severity_series == "CRITICAL").sum()
                )
                high_incidents = int(
                    (severity_series == "HIGH").sum()
                )
                medium_incidents = int(
                    (severity_series == "MEDIUM").sum()
                )
                low_incidents = int(
                    (severity_series == "LOW").sum()
                )
            else:
                critical_incidents = high_incidents = 0
                medium_incidents = low_incidents = 0

            if "estimated_exposure" in incident_df.columns:
                total_exposure = float(
                    incident_df["estimated_exposure"].sum()
                )
            elif "total_transaction_amount" in incident_df.columns:
                total_exposure = float(
                    incident_df["total_transaction_amount"].sum()
                )
            else:
                total_exposure = 0.0

            # ------------------------------------------------
            # KPI SECTION
            # ------------------------------------------------

            st.markdown(
                '<div class="command-section-title">'
                'Total Potential Exposure'
                '</div>',
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4, k5 = st.columns(
                [1.8, 0.95, 0.95, 0.95, 0.95],
                gap="small",
            )

            def metric_card(column, label, value, footer="", tone="blue"):
                with column:
                    st.markdown(
                        f'''
                        <div class="dashboard-kpi-card">
                            <div class="dashboard-kpi-label">{label}</div>
                            <div class="dashboard-kpi-value">{value}</div>
                            <div class="dashboard-kpi-footer {tone}">{footer}</div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

            metric_card(
                k1,
                "Potential Exposure",
                format_currency(total_exposure),
                "",
                "neutral",
            )
            metric_card(
                k2,
                "Active Incidents",
                f"{total_incidents:,}",
                "Total Active",
                "blue",
            )
            metric_card(
                k3,
                "Critical",
                f"{critical_incidents:,}",
                "High Priority",
                "red",
            )
            metric_card(
                k4,
                "Medium",
                f"{medium_incidents:,}",
                "Monitor",
                "amber",
            )
            metric_card(
                k5,
                "Low",
                f"{low_incidents:,}",
                "Low Priority",
                "green",
            )

            st.markdown(
                '<div style="height:14px"></div>',
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # ACTIVE FRAUD INCIDENTS
            # ------------------------------------------------

            st.markdown(
                '<div class="incident-section-heading">'
                '<span class="incident-heading-icon">🚨</span>'
                '<span>Active Fraud Incidents</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Select an incident to open the complete investigation."
            )

            filter_col1, filter_col2 = st.columns(
                [1, 2],
                gap="small",
            )

            with filter_col1:
                severity_filter = st.selectbox(
                    "Severity",
                    ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                    key="command_center_severity_filter",
                )

            with filter_col2:
                incident_search = st.text_input(
                    "Search incidents",
                    placeholder="Incident ID or fraud type",
                    key="command_center_incident_search",
                )

            filtered_incidents = list(incidents)

            if severity_filter != "ALL":
                filtered_incidents = [
                    incident
                    for incident in filtered_incidents
                    if str(
                        incident.get("severity", "")
                    ).upper().strip() == severity_filter
                ]

            if incident_search:
                search_value = incident_search.strip().lower()
                filtered_incidents = [
                    incident
                    for incident in filtered_incidents
                    if (
                        search_value
                        in str(incident.get("incident_id", "")).lower()
                        or search_value
                        in str(incident.get("incident_type", "")).lower()
                    )
                ]

            severity_order = {
                "CRITICAL": 0,
                "HIGH": 1,
                "MEDIUM": 2,
                "LOW": 3,
            }

            def safe_float(value):
                try:
                    return float(value or 0)
                except Exception:
                    return 0.0

            filtered_incidents = sorted(
                filtered_incidents,
                key=lambda item: (
                    severity_order.get(
                        str(item.get("severity", "")).upper().strip(),
                        99,
                    ),
                    -safe_float(item.get("risk_score", 0)),
                    -safe_float(
                        item.get(
                            "estimated_exposure",
                            item.get("total_transaction_amount", 0),
                        )
                    ),
                ),
            )

            st.caption(
                f"Showing {len(filtered_incidents):,} incidents"
            )

            header_widths = [
                0.9, 1.35, 0.9, 0.8, 1.0,
                0.85, 1.2, 1.25, 1.25,
            ]

            header_cols = st.columns(
                header_widths,
                gap="small",
            )

            headers = [
                "Incident ID",
                "Type",
                "Severity",
                "Risk Score",
                "Affected Customers",
                "Transactions",
                "Potential Exposure",
                "Time Detected",
                "Actions",
            ]

            for column, header in zip(header_cols, headers):
                with column:
                    st.markdown(
                        f'<div class="incident-table-header">{header}</div>',
                        unsafe_allow_html=True,
                    )

            if not filtered_incidents:

                st.info(
                    "No incidents match the selected filters."
                )

            else:

                show_all_key = "show_all_command_incidents"

                if show_all_key not in st.session_state:
                    st.session_state[show_all_key] = False

                visible_incidents = (
                    filtered_incidents
                    if st.session_state[show_all_key]
                    else filtered_incidents[:5]
                )

                for incident in visible_incidents:

                    incident_id = str(
                        incident.get("incident_id", "N/A")
                    )

                    incident_type = pretty_type(
                        incident.get("incident_type", "UNKNOWN")
                    )

                    severity = str(
                        incident.get("severity", "MEDIUM")
                    ).upper().strip()

                    try:
                        risk_score = int(
                            float(incident.get("risk_score", 0))
                        )
                    except Exception:
                        risk_score = 0

                    try:
                        transaction_count = int(
                            float(
                                incident.get(
                                    "transaction_count",
                                    0,
                                )
                            )
                        )
                    except Exception:
                        transaction_count = 0

                    try:
                        customer_count = int(
                            float(
                                incident.get(
                                    "customer_count",
                                    0,
                                )
                            )
                        )
                    except Exception:
                        customer_count = 0

                    try:
                        exposure = float(
                            incident.get(
                                "estimated_exposure",
                                incident.get(
                                    "total_transaction_amount",
                                    0,
                                ),
                            )
                        )
                    except Exception:
                        exposure = 0.0

                    detected_at = incident.get(
                        "first_seen",
                        incident.get(
                            "detected_at",
                            incident.get(
                                "created_at",
                                "N/A",
                            ),
                        ),
                    )

                    detected_text = str(
                        detected_at if detected_at is not None else "N/A"
                    )

                    if len(detected_text) > 19:
                        detected_text = detected_text[:19]

                    row_cols = st.columns(
                        header_widths,
                        gap="small",
                    )

                    severity_class = {
                        "CRITICAL": "severity-critical",
                        "HIGH": "severity-high",
                        "MEDIUM": "severity-medium",
                        "LOW": "severity-low",
                    }.get(
                        severity,
                        "severity-low",
                    )

                    with row_cols[0]:
                        st.markdown(
                            f'<div class="incident-table-cell incident-link">'
                            f'{incident_id}</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[1]:
                        st.markdown(
                            f'<div class="incident-table-cell incident-type-cell">'
                            f'{incident_type}</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[2]:
                        st.markdown(
                            f'<div class="incident-table-cell">'
                            f'<span class="{severity_class}">{severity}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[3]:
                        risk_color = (
                            "#dc2626"
                            if severity == "CRITICAL"
                            else "#ea580c"
                            if severity == "HIGH"
                            else "#d97706"
                            if severity == "MEDIUM"
                            else "#16a34a"
                        )
                        st.markdown(
                            f'<div class="incident-table-cell risk-number" '
                            f'style="color:{risk_color} !important;'
                            f'-webkit-text-fill-color:{risk_color} !important;">'
                            f'{risk_score}/100</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[4]:
                        st.markdown(
                            f'<div class="incident-table-cell">'
                            f'{customer_count:,}</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[5]:
                        st.markdown(
                            f'<div class="incident-table-cell">'
                            f'{transaction_count:,}</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[6]:
                        st.markdown(
                            f'<div class="incident-table-cell">'
                            f'{format_currency(exposure)}</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[7]:
                        st.markdown(
                            f'<div class="incident-table-cell time-cell">'
                            f'{detected_text}</div>',
                            unsafe_allow_html=True,
                        )

                    with row_cols[8]:
                        if st.button(
                            "🔍 Investigate",
                            key=f"table_investigate_{incident_id}",
                            width="stretch",
                        ):
                            st.session_state[
                                "selected_incident_id"
                            ] = incident_id
                            st.rerun()

                    st.markdown(
                        '<div class="incident-row-divider"></div>',
                        unsafe_allow_html=True,
                    )

                if len(filtered_incidents) > 5:

                    if st.session_state[show_all_key]:

                        if st.button(
                            "Show Top 5 Incidents",
                            key="show_top5_command_incidents",
                        ):
                            st.session_state[
                                show_all_key
                            ] = False
                            st.rerun()

                    else:

                        if st.button(
                            "View All Incidents  →",
                            key="show_all_command_incidents_button",
                        ):
                            st.session_state[
                                show_all_key
                            ] = True
                            st.rerun()

            # ------------------------------------------------
            # INCIDENT DISTRIBUTION
            # ------------------------------------------------

            st.markdown(
                '<div class="command-subsection-title">'
                'Incident Distribution'
                '</div>',
                unsafe_allow_html=True,
            )

            distribution_col1, distribution_col2 = st.columns(
                2,
                gap="medium",
            )

            with distribution_col1:

                st.markdown(
                    '<div class="distribution-title">'
                    'By Incident Type'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if "incident_type" in incident_df.columns:

                    type_distribution = (
                        incident_df["incident_type"]
                        .fillna("UNKNOWN")
                        .astype(str)
                        .map(pretty_type)
                        .value_counts()
                        .rename_axis("Incident Type")
                        .reset_index(name="Incidents")
                    )

                    st.dataframe(
                        type_distribution,
                        width="stretch",
                        hide_index=True,
                    )

                else:

                    st.info(
                        "Incident type data is unavailable."
                    )

            with distribution_col2:

                st.markdown(
                    '<div class="distribution-title">'
                    'By Severity'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if "severity" in incident_df.columns:

                    severity_distribution = (
                        incident_df["severity"]
                        .fillna("UNKNOWN")
                        .astype(str)
                        .str.upper()
                        .value_counts()
                        .rename_axis("Severity")
                        .reset_index(name="Incidents")
                    )

                    st.dataframe(
                        severity_distribution,
                        width="stretch",
                        hide_index=True,
                    )

                else:

                    st.info(
                        "Severity information is unavailable."
                    )

elif page == "Transaction Explorer":

    st.title(
        "Transaction Explorer"
    )

    st.caption(
        "Search merchant transactions and "
        "filter them by fraud risk."
    )

    df = prepare_transaction_data(
        load_transactions(),
        load_incidents(),
    )

    # Include a transaction created in the current session immediately in
    # Transaction Explorer. This avoids a stale cached dataset hiding the
    # newly appended CSV row until the cache is refreshed.
    latest_saved_transaction = st.session_state.get(
        "last_added_transaction",
        None,
    )

    if isinstance(
        latest_saved_transaction,
        dict,
    ) and latest_saved_transaction:

        latest_df = pd.DataFrame(
            [latest_saved_transaction]
        )

        try:
            latest_prepared = prepare_transaction_data(
                latest_df,
                load_incidents(),
            )

            if not latest_prepared.empty:

                if "transaction_id" in df.columns:
                    existing_ids = set(
                        df[
                            "transaction_id"
                        ]
                        .fillna("")
                        .astype(str)
                    )
                else:
                    existing_ids = set()

                latest_id = str(
                    latest_saved_transaction.get(
                        "transaction_id",
                        "",
                    )
                )

                if latest_id not in existing_ids:
                    df = pd.concat(
                        [
                            df,
                            latest_prepared,
                        ],
                        ignore_index=True,
                    )

        except Exception:
            # The persisted CSV remains the source of truth. If the
            # optional immediate-refresh preparation fails, do not break
            # the existing Transaction Explorer.
            pass

    if df.empty:

        st.error(
            "Transaction dataset unavailable."
        )

    else:

        # ====================================================
        # SEARCH AND FILTER
        # ====================================================

        search_col, risk_col = (
            st.columns(
                [3, 1],
                gap="medium",
            )
        )

        with search_col:

            st.markdown(
                '<div class="transaction-filter-label">'
                'Search transactions'
                '</div>',
                unsafe_allow_html=True,
            )

            search = st.text_input(
                "Search transactions",
                placeholder=(
                    "Transaction ID / "
                    "Customer ID / "
                    "Merchant ID"
                ),
                label_visibility="collapsed",
                key=(
                    "transaction_explorer_"
                    "search"
                ),
            )

        with risk_col:

            st.markdown(
                '<div class="transaction-filter-label">'
                'Risk Level'
                '</div>',
                unsafe_allow_html=True,
            )

            risk_filter = (
                st.selectbox(
                    "Risk Level",
                    [
                        "ALL",
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                        "LOW",
                    ],
                    label_visibility="collapsed",
                    key=(
                        "transaction_explorer_"
                        "risk"
                    ),
                )
            )

        result = df.copy()

        # ====================================================
        # TEXT SEARCH
        # ====================================================

        if search:

            search_value = (
                search
                .strip()
                .lower()
            )

            mask = pd.Series(
                False,
                index=result.index,
                dtype=bool,
            )

            searchable_columns = [
                "transaction_id",
                "customer_id",
                "merchant_id",
            ]

            for column in (
                searchable_columns
            ):

                if column in result.columns:

                    column_mask = (
                        result[
                            column
                        ]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.contains(
                            search_value,
                            na=False,
                            regex=False,
                        )
                    )

                    mask = (
                        mask
                        | column_mask
                    )

            result = result[
                mask
            ]

        # ====================================================
        # RISK FILTER
        # ====================================================

        if risk_filter != "ALL":

            result = result[
                result[
                    "risk_level"
                ]
                .fillna("")
                .astype(str)
                .str.upper()
                == risk_filter
            ]

        # ====================================================
        # SUMMARY
        # ====================================================

        summary1, summary2, summary3, summary4 = (
            st.columns(
                4,
                gap="medium",
            )
        )

        critical_count = int(
            (
                result[
                    "risk_level"
                ]
                == "CRITICAL"
            ).sum()
        )

        high_count = int(
            (
                result[
                    "risk_level"
                ]
                == "HIGH"
            ).sum()
        )

        medium_count = int(
            (
                result[
                    "risk_level"
                ]
                == "MEDIUM"
            ).sum()
        )

        low_count = int(
            (
                result[
                    "risk_level"
                ]
                == "LOW"
            ).sum()
        )

        summary_cards = [
            ("Critical", critical_count),
            ("High", high_count),
            ("Medium", medium_count),
            ("Low", low_count),
        ]

        for card_col, (label, value) in zip(
            [summary1, summary2, summary3, summary4],
            summary_cards,
        ):
            with card_col:
                st.markdown(
                    f'''
                    <div class="transaction-summary-card">
                        <div class="transaction-summary-label">
                            {label}
                        </div>
                        <div class="transaction-summary-value">
                            {value:,}
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ====================================================
        # FILTERED SEARCH RESULTS
        # ====================================================

        if (
            search
            or risk_filter != "ALL"
        ):

            st.subheader(
                "Filtered Transactions"
            )

            st.caption(
                f"{len(result):,} matching "
                "transactions."
            )

            if result.empty:

                st.info(
                    "No transactions match "
                    "the current search/filter."
                )

            else:

                if (
                    "timestamp"
                    in result.columns
                ):

                    result = (
                        result.copy()
                    )

                    result[
                        "_sort_timestamp"
                    ] = pd.to_datetime(
                        result[
                            "timestamp"
                        ],
                        errors="coerce",
                    )

                    result = (
                        result
                        .sort_values(
                            "_sort_timestamp",
                            ascending=False,
                        )
                        .drop(
                            columns=[
                                "_sort_timestamp"
                            ]
                        )
                    )

                st.dataframe(
                    format_transaction_table(
                        result.head(500)
                    ),
                    width="stretch",
                    hide_index=True,
                )

                if len(result) > 500:

                    st.caption(
                        "Showing the first 500 "
                        "matching transactions."
                    )

        # ====================================================
        # RECENT TRANSACTIONS
        # ====================================================

        else:

            st.subheader(
                "Recent Transactions"
            )

            st.caption(
                "Latest transactions across "
                "the merchant dataset."
            )

            recent = df.copy()

            if (
                "timestamp"
                in recent.columns
            ):

                recent[
                    "_sort_timestamp"
                ] = pd.to_datetime(
                    recent[
                        "timestamp"
                    ],
                    errors="coerce",
                )

                recent = (
                    recent
                    .sort_values(
                        "_sort_timestamp",
                        ascending=False,
                    )
                    .drop(
                        columns=[
                            "_sort_timestamp"
                        ]
                    )
                )

            st.dataframe(
                format_transaction_table(
                    recent.head(100)
                ),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Showing the 100 most recent "
                "transactions."
            )

        st.divider()

        # ====================================================
        # FRAUD TRANSACTION BREAKDOWN
        # ====================================================

        st.subheader(
            "Fraud Type Breakdown"
        )

        if "fraud_type" in df.columns:

            fraud_series = (
                df[
                    "fraud_type"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            fraud_series = (
                fraud_series[
                    ~fraud_series.isin(
                        [
                            "",
                            "nan",
                            "None",
                            "normal",
                        ]
                    )
                ]
            )

            if fraud_series.empty:

                st.info(
                    "No fraud transactions "
                    "were found."
                )

            else:

                fraud_breakdown = (
                    fraud_series
                    .value_counts()
                    .rename_axis(
                        "Fraud Type"
                    )
                    .reset_index(
                        name="Transactions"
                    )
                )

                fraud_breakdown[
                    "Fraud Type"
                ] = (
                    fraud_breakdown[
                        "Fraud Type"
                    ]
                    .map(
                        pretty_type
                    )
                )

                st.dataframe(
                    fraud_breakdown,
                    width="stretch",
                    hide_index=True,
                )

        else:

            st.info(
                "Fraud type information "
                "is unavailable."
            )
            # ============================================================
# CUSTOMER INTELLIGENCE
# ============================================================

elif page == "Customer Intelligence":

    st.title(
        "Customer Intelligence"
    )

    st.caption(
        "Investigate customer-level transaction "
        "history, risk exposure and fraud activity."
    )

    df = prepare_transaction_data(
        load_transactions(),
        load_incidents(),
    )

    if df.empty:

        st.error(
            "Transaction dataset unavailable."
        )

    else:

        # ====================================================
        # CUSTOMER SEARCH
        # ====================================================

        search_col1, search_col2 = (
            st.columns(
                [3, 1],
                gap="medium",
            )
        )

        with search_col1:

            customer_search = (
                st.text_input(
                    "Search Customer",
                    placeholder=(
                        "Enter Customer ID"
                    ),
                    key=(
                        "customer_intelligence_"
                        "search"
                    ),
                )
            )

        with search_col2:

            customer_risk_filter = (
                st.selectbox(
                    "Risk Level",
                    [
                        "ALL",
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                        "LOW",
                    ],
                    key=(
                        "customer_intelligence_"
                        "risk"
                    ),
                )
            )

        # ====================================================
        # CUSTOMER LIST
        # ====================================================

        if customer_search:

            search_value = (
                customer_search
                .strip()
                .lower()
            )

            if "customer_id" in df.columns:

                customer_df = df[
                    df[
                        "customer_id"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(
                        search_value,
                        na=False,
                        regex=False,
                    )
                ].copy()

            else:

                customer_df = (
                    pd.DataFrame()
                )

        else:

            customer_df = df.copy()

        # ====================================================
        # RISK FILTER
        # ====================================================

        if (
            customer_risk_filter
            != "ALL"
            and not customer_df.empty
        ):

            customer_df = (
                customer_df[
                    customer_df[
                        "risk_level"
                    ]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    == customer_risk_filter
                ]
                .copy()
            )

        if customer_df.empty:

            st.info(
                "No customers found matching "
                "the current search and filters."
            )

        else:

            # =================================================
            # CUSTOMER SUMMARY
            # =================================================

            if "customer_id" in customer_df.columns:

                unique_customers = (
                    customer_df[
                        "customer_id"
                    ]
                    .dropna()
                    .astype(str)
                    .nunique()
                )

            else:

                unique_customers = 0

            total_transactions = len(
                customer_df
            )

            total_exposure = 0.0

            if "amount" in customer_df.columns:

                total_exposure = float(
                    pd.to_numeric(
                        customer_df[
                            "amount"
                        ],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )

            critical_transactions = int(
                (
                    customer_df[
                        "risk_level"
                    ]
                    .astype(str)
                    .str.upper()
                    == "CRITICAL"
                ).sum()
            )

            m1, m2, m3, m4 = (
                st.columns(
                    4,
                    gap="medium",
                )
            )

            with m1:

                visible_metric(
                    "Customers",
                    f"{unique_customers:,}",
                )

            with m2:

                visible_metric(
                    "Transactions",
                    f"{total_transactions:,}",
                )

            with m3:

                visible_metric(
                    "Potential Exposure",
                    format_currency(
                        total_exposure
                    ),
                )

            with m4:

                visible_metric(
                    "Critical Transactions",
                    f"{critical_transactions:,}",
                )

            st.divider()

            # =================================================
            # SINGLE CUSTOMER INVESTIGATION
            # =================================================

            if (
                customer_search
                and "customer_id"
                in customer_df.columns
            ):

                matching_customers = (
                    customer_df[
                        "customer_id"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(
                    matching_customers
                ) == 1:

                    selected_customer = (
                        matching_customers[0]
                    )

                    st.header(
                        f"Customer: {selected_customer}"
                    )

                    customer_transactions = (
                        customer_df[
                            customer_df[
                                "customer_id"
                            ]
                            .astype(str)
                            == selected_customer
                        ]
                        .copy()
                    )

                    # -----------------------------------------
                    # CUSTOMER RISK
                    # -----------------------------------------

                    if (
                        not customer_transactions.empty
                        and "risk_score"
                        in customer_transactions.columns
                    ):

                        customer_risk = int(
                            pd.to_numeric(
                                customer_transactions[
                                    "risk_score"
                                ],
                                errors="coerce",
                            )
                            .fillna(0)
                            .max()
                        )

                    else:

                        customer_risk = 0

                    if customer_risk >= 90:

                        customer_risk_level = (
                            "CRITICAL"
                        )

                    elif customer_risk >= 75:

                        customer_risk_level = (
                            "HIGH"
                        )

                    elif customer_risk >= 50:

                        customer_risk_level = (
                            "MEDIUM"
                        )

                    else:

                        customer_risk_level = (
                            "LOW"
                        )

                    c1, c2, c3, c4 = (
                        st.columns(
                            4,
                            gap="medium",
                        )
                    )

                    with c1:

                        visible_metric(
                            "Customer Risk",
                            f"{customer_risk}/100",
                        )

                    with c2:

                        visible_metric(
                            "Risk Level",
                            customer_risk_level,
                        )

                    with c3:

                        visible_metric(
                            "Transactions",
                            f"{len(customer_transactions):,}",
                        )

                    with c4:

                        customer_amount = (
                            pd.to_numeric(
                                customer_transactions[
                                    "amount"
                                ],
                                errors="coerce",
                            )
                            .fillna(0)
                            .sum()
                        )

                        visible_metric(
                            "Transaction Value",
                            format_currency(
                                customer_amount
                            ),
                        )

                    st.divider()

                    # -----------------------------------------
                    # CUSTOMER TRANSACTION HISTORY
                    # -----------------------------------------

                    st.subheader(
                        "Transaction History"
                    )

                    customer_display = (
                        customer_transactions
                        .copy()
                    )

                    if (
                        "timestamp"
                        in customer_display.columns
                    ):

                        customer_display[
                            "_timestamp"
                        ] = pd.to_datetime(
                            customer_display[
                                "timestamp"
                            ],
                            errors="coerce",
                        )

                        customer_display = (
                            customer_display
                            .sort_values(
                                "_timestamp",
                                ascending=False,
                            )
                            .drop(
                                columns=[
                                    "_timestamp"
                                ]
                            )
                        )

                    st.dataframe(
                        format_transaction_table(
                            customer_display
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    st.divider()

                    # -----------------------------------------
                    # CUSTOMER FRAUD SIGNALS
                    # -----------------------------------------

                    st.subheader(
                        "Customer Fraud Signals"
                    )

                    if (
                        "fraud_type"
                        in customer_transactions.columns
                    ):

                        customer_fraud = (
                            customer_transactions[
                                "fraud_type"
                            ]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                        )

                        customer_fraud = (
                            customer_fraud[
                                ~customer_fraud
                                .isin(
                                    [
                                        "",
                                        "normal",
                                        "nan",
                                        "None",
                                    ]
                                )
                            ]
                        )

                        if customer_fraud.empty:

                            st.info(
                                "No explicit fraud-type "
                                "signals were found."
                            )

                        else:

                            customer_fraud_df = (
                                customer_fraud
                                .value_counts()
                                .rename_axis(
                                    "Fraud Type"
                                )
                                .reset_index(
                                    name="Transactions"
                                )
                            )

                            customer_fraud_df[
                                "Fraud Type"
                            ] = (
                                customer_fraud_df[
                                    "Fraud Type"
                                ]
                                .map(
                                    pretty_type
                                )
                            )

                            st.dataframe(
                                customer_fraud_df,
                                width="stretch",
                                hide_index=True,
                            )

                    # -----------------------------------------
                    # INCIDENTS INVOLVING CUSTOMER
                    # -----------------------------------------

                    st.subheader(
                        "Linked Incidents"
                    )

                    if (
                        "incident_id"
                        in customer_transactions.columns
                    ):

                        linked_incident_ids = (
                            customer_transactions[
                                "incident_id"
                            ]
                            .dropna()
                            .astype(str)
                            .str.strip()
                        )

                        linked_incident_ids = [
                            value
                            for value
                            in linked_incident_ids.unique()
                            if value
                            and value.lower()
                            not in {
                                "nan",
                                "none",
                            }
                        ]

                        if linked_incident_ids:

                            incident_rows = []

                            all_incidents = (
                                load_incidents()
                            )

                            for incident in (
                                all_incidents
                            ):

                                incident_id = str(
                                    incident.get(
                                        "incident_id",
                                        "",
                                    )
                                ).strip()

                                incident_variants = (
                                    set(
                                        incident_id_variants(
                                            incident_id
                                        )
                                    )
                                )

                                if (
                                    incident_variants
                                    & set(
                                        linked_incident_ids
                                    )
                                ):

                                    incident_rows.append(
                                        {
                                            "Incident ID":
                                                incident_id,

                                            "Type":
                                                pretty_type(
                                                    incident.get(
                                                        "incident_type",
                                                        "",
                                                    )
                                                ),

                                            "Severity":
                                                str(
                                                    incident.get(
                                                        "severity",
                                                        "",
                                                    )
                                                ).upper(),

                                            "Risk":
                                                incident.get(
                                                    "risk_score",
                                                    0,
                                                ),

                                            "Transactions":
                                                incident.get(
                                                    "transaction_count",
                                                    0,
                                                ),

                                            "Exposure":
                                                format_currency(
                                                    incident.get(
                                                        "estimated_exposure",
                                                        0,
                                                    )
                                                ),
                                        }
                                    )

                            if incident_rows:

                                st.dataframe(
                                    pd.DataFrame(
                                        incident_rows
                                    ),
                                    width="stretch",
                                    hide_index=True,
                                )

                            else:

                                st.info(
                                    "No incident details "
                                    "were found."
                                )

                        else:

                            st.info(
                                "This customer is not "
                                "linked to a detected incident."
                            )

                else:

                    # -----------------------------------------
                    # MULTIPLE CUSTOMER MATCHES
                    # -----------------------------------------

                    st.subheader(
                        "Matching Customers"
                    )

                    customer_summary = (
                        customer_df
                        .groupby(
                            "customer_id",
                            dropna=False,
                        )
                        .agg(
                            Transactions=(
                                "customer_id",
                                "size",
                            ),
                            Max_Risk=(
                                "risk_score",
                                "max",
                            ),
                        )
                        .reset_index()
                    )

                    customer_summary[
                        "Risk Level"
                    ] = (
                        customer_summary[
                            "Max_Risk"
                        ]
                        .apply(
                            lambda value:
                                (
                                    "CRITICAL"
                                    if value >= 90
                                    else
                                    "HIGH"
                                    if value >= 75
                                    else
                                    "MEDIUM"
                                    if value >= 50
                                    else
                                    "LOW"
                                )
                        )
                    )

                    customer_summary = (
                        customer_summary
                        .rename(
                            columns={
                                "customer_id":
                                    "Customer ID",
                                "Max_Risk":
                                    "Max Risk",
                            }
                        )
                    )

                    st.dataframe(
                        customer_summary,
                        width="stretch",
                        hide_index=True,
                    )

            # =================================================
            # CUSTOMER RISK DISTRIBUTION
            # =================================================

            st.divider()

            st.subheader(
                "Customer Risk Distribution"
            )

            if (
                "customer_id"
                in customer_df.columns
            ):

                customer_risk_summary = (
                    customer_df
                    .groupby(
                        "customer_id",
                        dropna=False,
                    )[
                        "risk_score"
                    ]
                    .max()
                    .reset_index()
                )

                customer_risk_summary[
                    "Risk Level"
                ] = (
                    customer_risk_summary[
                        "risk_score"
                    ]
                    .apply(
                        lambda value:
                            (
                                "CRITICAL"
                                if value >= 90
                                else
                                "HIGH"
                                if value >= 75
                                else
                                "MEDIUM"
                                if value >= 50
                                else
                                "LOW"
                            )
                    )
                )

                distribution = (
                    customer_risk_summary[
                        "Risk Level"
                    ]
                    .value_counts()
                    .reindex(
                        [
                            "CRITICAL",
                            "HIGH",
                            "MEDIUM",
                            "LOW",
                        ],
                        fill_value=0,
                    )
                    .rename_axis(
                        "Risk Level"
                    )
                    .reset_index(
                        name="Customers"
                    )
                )

                st.dataframe(
                    distribution,
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":
    def rupees(value):
        try:
            return f"₹{float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)


    st.title("Model Performance")

    st.caption(
        "Held-out evaluation of the fraud risk model and its business-cost operating threshold."
    )

    metrics = load_metrics()

    if not isinstance(metrics, dict):
        metrics = {}

    test_metrics = metrics.get(
        "held_out_test",
        {},
    )

    if not isinstance(test_metrics, dict):
        test_metrics = {}

    def metric_value(key, default=None):
        for container in (
            test_metrics,
            metrics,
        ):
            if not isinstance(container, dict):
                continue

            value = container.get(key)

            if value is not None and value != "":
                return value

        return default

    def pct_metric(key):
        value = metric_value(key)

        if value is None:
            return "N/A"

        try:
            return f"{float(value) * 100:.2f}%"
        except (
            TypeError,
            ValueError,
        ):
            return "N/A"

    precision = pct_metric("precision")
    recall = pct_metric("recall")
    f1 = pct_metric("f1")
    pr_auc = pct_metric("pr_auc")

    st.subheader("Held-Out Test Set")

    c1, c2, c3, c4 = st.columns(
        4,
        gap="medium",
    )

    with c1:
        visible_metric(
            "Precision",
            precision,
        )

    with c2:
        visible_metric(
            "Recall",
            recall,
        )

    with c3:
        visible_metric(
            "F1 Score",
            f1,
        )

    with c4:
        visible_metric(
            "PR-AUC",
            pr_auc,
        )

    # Local currency formatter used by the AI Risk Manager summary.
    def _risk_manager_rupees(value):
        try:
            return f"₹{float(value):,.2f}"
        except (
            TypeError,
            ValueError,
        ):
            return str(value)

    # ============================================================
    # AI RISK MANAGER EVALUATION SUMMARY
    # ============================================================
    # Keep this tied to the existing held-out evaluation artifact.
    # No metrics are recomputed from training data here.

    st.subheader("AI Risk Manager Evaluation")

    evaluation_col1, evaluation_col2, evaluation_col3 = st.columns(
        3,
        gap="medium",
    )

    heldout_rows = (
        test_metrics.get("test_rows")
        or test_metrics.get("held_out_rows")
        or metrics.get("held_out_test_rows")
    )

    with evaluation_col1:
        visible_metric(
            "Evaluation Set",
            (
                f"{int(float(heldout_rows)):,} rows"
                if heldout_rows is not None
                else "Held-out test"
            ),
        )

    with evaluation_col2:
        visible_metric(
            "Evaluation Method",
            "Temporal Holdout",
        )

    with evaluation_col3:
        visible_metric(
            "Threshold Selection",
            "Validation Only",
        )

    st.caption(
        "The operating threshold is selected using validation data. "
        "The final precision, recall, F1 and PR-AUC are reported on "
        "the untouched held-out temporal test set."
    )

    # Make the business-loss trade-off immediately visible.
    cost_summary = pd.DataFrame(
        [
            {
                "Metric": "False-Positive Cost",
                "Value": rupees(
                    metric_value(
                        "false_positive_cost",
                        0,
                    )
                ),
            },
            {
                "Metric": "False-Negative Cost",
                "Value": rupees(
                    metric_value(
                        "false_negative_cost",
                        0,
                    )
                ),
            },
            {
                "Metric": "Expected Loss",
                "Value": rupees(
                    metric_value(
                        "expected_loss",
                        0,
                    )
                ),
            },
        ]
    )

    st.markdown("#### Business Loss Summary")

    st.dataframe(
        cost_summary,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        st.subheader("Operating Threshold")

        threshold = metrics.get(
            "selected_threshold",
            test_metrics.get(
                "threshold",
                0.30,
            ),
        )

        try:
            threshold_display = (
                f"{float(threshold):.4f}"
            )
        except (
            TypeError,
            ValueError,
        ):
            threshold_display = str(
                threshold
            )

        visible_metric(
            "Selected Threshold",
            threshold_display,
        )

        st.caption(
            "Selected using validation data only to minimize estimated financial loss."
        )

    with right:
        st.subheader("Financial Cost")

        fp_cost = metric_value(
            "false_positive_cost",
            0,
        )

        fn_cost = metric_value(
            "false_negative_cost",
            0,
        )

        expected_loss = metric_value(
            "expected_loss",
            0,
        )

        def rupees(value):
            try:
                return f"₹{float(value):,.2f}"
            except (
                TypeError,
                ValueError,
            ):
                return str(value)

        cost1, cost2, cost3 = st.columns(3)

        with cost1:
            visible_metric(
                "False-Positive Cost",
                rupees(fp_cost),
            )

        with cost2:
            visible_metric(
                "False-Negative Cost",
                rupees(fn_cost),
            )

        with cost3:
            visible_metric(
                "Expected Loss",
                rupees(expected_loss),
            )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        {
            "Predicted Fraud": [
                metric_value(
                    "true_positives",
                    0,
                ),
                metric_value(
                    "false_positives",
                    0,
                ),
            ],
            "Predicted Legitimate": [
                metric_value(
                    "false_negatives",
                    0,
                ),
                metric_value(
                    "true_negatives",
                    0,
                ),
            ],
        },
        index=[
            "Actual Fraud",
            "Actual Legitimate",
        ],
    )

    st.dataframe(
        cm,
        width="stretch",
    )

    st.divider()

    st.subheader("Threshold Optimization")

    # ------------------------------------------------------------
    # LOAD THE REAL VALIDATION THRESHOLD SWEEP
    # ------------------------------------------------------------
    # The metrics artifact can store validation results as either a
    # list or a nested dictionary. Extract the actual threshold rows
    # instead of assuming one fixed JSON shape.
    def _extract_threshold_rows(value):
        if isinstance(value, list):
            rows = [
                row
                for row in value
                if isinstance(row, dict)
                and "threshold" in row
            ]
            if rows:
                return rows

            for item in value:
                rows = _extract_threshold_rows(item)
                if rows:
                    return rows

        elif isinstance(value, dict):
            # Prefer known validation/threshold containers.
            for key in (
                "validation",
                "threshold_comparison",
                "thresholds",
                "threshold_sweep",
                "threshold_results",
                "results",
            ):
                if key in value:
                    rows = _extract_threshold_rows(value[key])
                    if rows:
                        return rows

            # Finally search nested values without depending on the
            # exact artifact schema.
            for nested in value.values():
                rows = _extract_threshold_rows(nested)
                if rows:
                    return rows

        return []


    threshold_rows = _extract_threshold_rows(metrics)

    if (
        isinstance(
            threshold_rows,
            list,
        )
        and threshold_rows
    ):
        threshold_table = pd.DataFrame(
            threshold_rows
        ).copy()

        # Present the real validation sweep in a judge-friendly order.
        preferred_columns = [
            "threshold",
            "precision",
            "recall",
            "f1",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "estimated_cost",
        ]

        available_columns = [
            column
            for column in preferred_columns
            if column in threshold_table.columns
        ]

        if available_columns:
            threshold_table = threshold_table[
                available_columns
            ]

        st.dataframe(
            threshold_table,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "Threshold comparison is available in the evaluation artifact when recorded."
        )

    # ------------------------------------------------------------
    # COST-AWARE THRESHOLD TRADE-OFF
    # ------------------------------------------------------------

    if (
        isinstance(threshold_rows, list)
        and threshold_rows
    ):
        threshold_df = pd.DataFrame(
            threshold_rows
        ).copy()

        # Normalize common artifact column names.
        rename_candidates = {
            "false_positive_cost": "False Positive Cost",
            "false_negative_cost": "False Negative Cost",
            "estimated_cost": "Estimated Cost",
            "expected_loss": "Expected Loss",
            "threshold": "Threshold",
        }

        threshold_df = threshold_df.rename(
            columns={
                key: value
                for key, value in rename_candidates.items()
                if key in threshold_df.columns
            }
        )

        cost_column = (
            "Estimated Cost"
            if "Estimated Cost" in threshold_df.columns
            else "Expected Loss"
            if "Expected Loss" in threshold_df.columns
            else None
        )

        if (
            "Threshold" in threshold_df.columns
            and cost_column is not None
        ):
            threshold_chart = threshold_df[
                [
                    "Threshold",
                    cost_column,
                ]
            ].copy()

            threshold_chart["Threshold"] = pd.to_numeric(
                threshold_chart["Threshold"],
                errors="coerce",
            )

            threshold_chart[cost_column] = pd.to_numeric(
                threshold_chart[cost_column],
                errors="coerce",
            )

            threshold_chart = threshold_chart.dropna()

            if not threshold_chart.empty:
                threshold_chart = threshold_chart.set_index(
                    "Threshold"
                )

                st.markdown(
                    "#### Expected Loss by Operating Threshold"
                )

                st.line_chart(
                    threshold_chart[
                        [cost_column]
                    ],
                    height=280,
                    use_container_width=True,
                )

                st.caption(
                    "Lower estimated cost indicates a better business "
                    "trade-off between false positives and false negatives. "
                    "The threshold sweep comes from validation data only."
                )

    st.divider()

    st.subheader("Model Configuration")

    model_name = metrics.get(
        "model",
        metrics.get(
            "model_name",
            "HistGradientBoostingClassifier",
        ),
    )

    config_df = pd.DataFrame(
        [
            {
                "Metric": "Model",
                "Value": model_name,
            },
            {
                "Metric": "Selected Threshold",
                "Value": threshold_display,
            },
            {
                "Metric": "False Positive Cost",
                "Value": rupees(fp_cost),
            },
            {
                "Metric": "False Negative Cost",
                "Value": rupees(fn_cost),
            },
            {
                "Metric": "Expected Loss",
                "Value": rupees(expected_loss),
            },
        ]
    )

    st.dataframe(
        config_df,
        width="stretch",
        hide_index=True,
    )

    # ------------------------------------------------------------
    # SELECTION-CRITERIA CHECK
    # ------------------------------------------------------------


    st.divider()

    st.subheader("Prediction Audit Trail")

    audit_df = load_prediction_audit()

    if audit_df.empty:
        st.info(
            "No prediction audit records found yet. "
            "The API can populate reports/prediction_audit.jsonl for new decisions."
        )
    else:
        preferred_columns = [
            "timestamp",
            "transaction_id",
            "fraud_probability",
            "risk_score",
            "model_version",
            "threshold",
            "risk_factors",
            "incident_id",
            "decision",
            "action",
            "recommended_action",
        ]

        visible_columns = [
            column
            for column in preferred_columns
            if column in audit_df.columns
        ]

        remaining_columns = [
            column
            for column in audit_df.columns
            if column not in visible_columns
        ]

        audit_view = audit_df[
            visible_columns + remaining_columns
        ].tail(100)

        st.dataframe(
            audit_view,
            width="stretch",
            hide_index=True,
        )

    if not test_metrics:
        st.warning(
            "Held-out model metrics were not found in strong_optimized_metrics.json. "
            "Run the model evaluation/optimization step to populate the report."
        )



    # ============================================================
    # MODEL MONITORING
    # ============================================================
    # Added monitoring only. Existing Model Performance sections,
    # incident handling, AI assistant, and other dashboard features
    # remain unchanged.
    st.divider()

    st.subheader("Model Monitoring")

    st.caption(
        "Production-style monitoring of fraud-rate behavior, "
        "risk distribution and current model performance."
    )

    monitoring_df = load_transactions()

    if monitoring_df.empty:
        st.info(
            "Transaction data is not available for model monitoring."
        )

    else:
        monitoring_df = monitoring_df.copy()

        if "timestamp" in monitoring_df.columns:
            monitoring_df["timestamp"] = pd.to_datetime(
                monitoring_df["timestamp"],
                errors="coerce",
            )

        # --------------------------------------------------------
        # OBSERVED FRAUD RATE
        # --------------------------------------------------------

        if "is_fraud" in monitoring_df.columns:

            monitoring_df["is_fraud"] = pd.to_numeric(
                monitoring_df["is_fraud"],
                errors="coerce",
            ).fillna(0).astype(int)

            monitoring_total = len(monitoring_df)

            monitoring_fraud_count = int(
                monitoring_df["is_fraud"].sum()
            )

            monitoring_fraud_rate = (
                monitoring_fraud_count / monitoring_total
                if monitoring_total
                else 0
            )

        else:

            monitoring_total = len(monitoring_df)
            monitoring_fraud_count = 0
            monitoring_fraud_rate = 0

        # --------------------------------------------------------
        # RECENT FRAUD-RATE COMPARISON
        # --------------------------------------------------------

        recent_rate = None
        previous_rate = None
        rate_change = None

        if (
            "timestamp" in monitoring_df.columns
            and "is_fraud" in monitoring_df.columns
            and monitoring_df["timestamp"].notna().any()
        ):

            valid_monitoring = (
                monitoring_df[
                    monitoring_df["timestamp"].notna()
                ]
                .sort_values("timestamp")
            )

            latest_timestamp = (
                valid_monitoring["timestamp"].max()
            )

            seven_days = pd.Timedelta(days=7)

            recent_start = latest_timestamp - seven_days
            previous_start = latest_timestamp - seven_days * 2

            recent_df = valid_monitoring[
                valid_monitoring["timestamp"] >= recent_start
            ]

            previous_df = valid_monitoring[
                (valid_monitoring["timestamp"] >= previous_start)
                & (valid_monitoring["timestamp"] < recent_start)
            ]

            if not recent_df.empty:
                recent_rate = float(recent_df["is_fraud"].mean())

            if not previous_df.empty:
                previous_rate = float(previous_df["is_fraud"].mean())

            if (
                recent_rate is not None
                and previous_rate is not None
                and previous_rate > 0
            ):
                rate_change = (
                    (recent_rate - previous_rate)
                    / previous_rate
                )

        # --------------------------------------------------------
        # OPERATING THRESHOLD
        # --------------------------------------------------------

        monitoring_threshold = metrics.get(
            "selected_threshold",
            test_metrics.get("threshold", 0.30),
        )

        try:
            monitoring_threshold = float(monitoring_threshold)
        except (TypeError, ValueError):
            monitoring_threshold = 0.30

        # --------------------------------------------------------
        # MONITORING KPIs
        # --------------------------------------------------------

        m1, m2, m3, m4 = st.columns(4, gap="medium")

        with m1:
            visible_metric(
                "Observed Fraud Rate",
                f"{monitoring_fraud_rate * 100:.2f}%",
                delta=f"{monitoring_fraud_count:,} fraud transactions",
                delta_color="normal",
            )

        with m2:
            visible_metric(
                "Last 7-Day Fraud Rate",
                (
                    f"{recent_rate * 100:.2f}%"
                    if recent_rate is not None
                    else "N/A"
                ),
            )

        with m3:
            visible_metric(
                "Fraud-Rate Change",
                (
                    f"{rate_change * 100:+.1f}%"
                    if rate_change is not None
                    else "N/A"
                ),
                delta=(
                    "vs previous 7 days"
                    if rate_change is not None
                    else None
                ),
                delta_color=(
                    "inverse"
                    if rate_change is not None and rate_change > 0
                    else "normal"
                ),
            )

        with m4:
            # Do not pass HTML through visible_metric's optional help
            # field here. Streamlit can display that generated HTML as
            # literal text under some theme/browser combinations.
            visible_metric(
                "Operating Threshold",
                f"{monitoring_threshold:.2f}",
            )

        st.caption("Selected using validation data only.")

        # --------------------------------------------------------
        # FRAUD SPIKE STATUS
        # --------------------------------------------------------

        if rate_change is not None and rate_change >= 0.50:
            st.warning(
                "Fraud-rate spike detected: the latest 7-day "
                "fraud rate is "
                f"{rate_change * 100:.1f}% above the previous "
                "7-day period."
            )
        elif rate_change is not None and rate_change <= -0.25:
            st.success(
                "Observed fraud rate has decreased compared "
                "with the previous 7-day period."
            )
        else:
            st.info(
                "No significant 7-day fraud-rate spike detected."
            )

        # --------------------------------------------------------
        # RISK DISTRIBUTION
        # --------------------------------------------------------

        audit_monitoring_df = load_prediction_audit()

        if (
            not audit_monitoring_df.empty
            and "risk_score" in audit_monitoring_df.columns
        ):

            risk_values = pd.to_numeric(
                audit_monitoring_df["risk_score"],
                errors="coerce",
            ).dropna()

            if not risk_values.empty:
                risk_labels = pd.cut(
                    risk_values,
                    bins=[
                        -float("inf"),
                        49,
                        74,
                        89,
                        float("inf"),
                    ],
                    labels=[
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL",
                    ],
                )

                risk_counts = (
                    risk_labels.value_counts()
                    .reindex(
                        [
                            "LOW",
                            "MEDIUM",
                            "HIGH",
                            "CRITICAL",
                        ],
                        fill_value=0,
                    )
                    .astype(int)
                )

                # Use a normal DataFrame with explicit string columns.
                # Passing the categorical Series directly to st.bar_chart
                # can produce the Streamlit "object object" rendering bug.
                risk_chart_df = pd.DataFrame({
                    "Risk Level": risk_counts.index.astype(str),
                    "Transactions": risk_counts.to_numpy(dtype=int),
                }).set_index("Risk Level")

                st.markdown("#### Recent Risk Distribution")

                st.bar_chart(
                    risk_chart_df,
                    y="Transactions",
                    height=280,
                    use_container_width=True,
                )

        else:
            st.caption(
                "Risk distribution will appear after prediction "
                "audit records are generated."
            )

        # --------------------------------------------------------
        # FRAUD RATE OVER TIME
        # --------------------------------------------------------

        if (
            "timestamp" in monitoring_df.columns
            and "is_fraud" in monitoring_df.columns
        ):

            timeline_df = monitoring_df[
                monitoring_df["timestamp"].notna()
            ].copy()

            if not timeline_df.empty:
                timeline_df["Date"] = timeline_df["timestamp"].dt.floor("D")

                fraud_timeline = (
                    timeline_df
                    .groupby("Date", as_index=True)["is_fraud"]
                    .mean()
                    .mul(100)
                    .rename("Fraud Rate (%)")
                    .to_frame()
                    .sort_index()
                )

                if len(fraud_timeline) >= 2:
                    st.markdown("#### Fraud Rate Over Time")
                    st.line_chart(
                        fraud_timeline,
                        y="Fraud Rate (%)",
                        height=300,
                        use_container_width=True,
                    )

        # --------------------------------------------------------
        # HELD-OUT PERFORMANCE
        # --------------------------------------------------------

        st.markdown("#### Held-Out Performance Baseline")

        hm1, hm2, hm3, hm4 = st.columns(
            4,
            gap="medium",
        )

        with hm1:
            visible_metric("Precision", precision)

        with hm2:
            visible_metric("Recall", recall)

        with hm3:
            visible_metric("F1 Score", f1)

        with hm4:
            visible_metric("PR-AUC", pr_auc)

        st.caption(
            "Monitoring is read-only. It does not retrain, replace, "
            "or automatically modify the production model."
        )


# ============================================================
# INCIDENT CENTRE
# ============================================================

elif page == "Incident Centre":

    st.title(
        "Incident Centre"
    )

    st.caption(
        "Prioritize incidents requiring immediate "
        "fraud-operations attention."
    )

    incidents = load_incidents()

    if not incidents:

        st.info(
            "No incidents are currently available."
        )

    else:

        incident_df = pd.DataFrame(
            incidents
        )

        # ----------------------------------------------------
        # NORMALIZE NUMERIC VALUES
        # ----------------------------------------------------

        for column in [
            "risk_score",
            "transaction_count",
            "customer_count",
            "estimated_exposure",
        ]:

            if column in incident_df.columns:

                incident_df[
                    column
                ] = pd.to_numeric(
                    incident_df[
                        column
                    ],
                    errors="coerce",
                ).fillna(0)

        # ----------------------------------------------------
        # SEVERITY NORMALIZATION
        # ----------------------------------------------------

        if "severity" in incident_df.columns:

            incident_df[
                "severity"
            ] = (
                incident_df[
                    "severity"
                ]
                .fillna("LOW")
                .astype(str)
                .str.upper()
            )

        # ----------------------------------------------------
        # PRIORITY SCORE
        # ----------------------------------------------------

        severity_weight = {
            "CRITICAL": 4,
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
        }

        incident_df[
            "_severity_weight"
        ] = (
            incident_df[
                "severity"
            ]
            .map(
                severity_weight
            )
            .fillna(0)
        )

        incident_df[
            "Priority Score"
        ] = (
            incident_df[
                "_severity_weight"
            ] * 100
            + incident_df[
                "risk_score"
            ]
        )

        incident_df = (
            incident_df
            .sort_values(
                [
                    "Priority Score",
                    "estimated_exposure",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        critical_count = int(
            (
                incident_df[
                    "severity"
                ]
                == "CRITICAL"
            ).sum()
        )

        high_count = int(
            (
                incident_df[
                    "severity"
                ]
                == "HIGH"
            ).sum()
        )

        total_exposure = float(
            incident_df[
                "estimated_exposure"
            ].sum()
        )

        m1, m2, m3 = (
            st.columns(
                3,
                gap="medium",
            )
        )

        with m1:

            visible_metric(
                "Critical Incidents",
                f"{critical_count:,}",
            )

        with m2:

            visible_metric(
                "High Incidents",
                f"{high_count:,}",
            )

        with m3:

            visible_metric(
                "Exposure at Risk",
                format_currency(
                    total_exposure
                ),
            )

        st.divider()

        # ====================================================
        # PRIORITY QUEUE
        # ====================================================

        st.header(
            "Priority Queue"
        )

        for _, incident in (
            incident_df.iterrows()
        ):

            incident_id = str(
                incident.get(
                    "incident_id",
                    "N/A",
                )
            )

            severity = str(
                incident.get(
                    "severity",
                    "LOW",
                )
            ).upper()

            incident_type = pretty_type(
                incident.get(
                    "incident_type",
                    "UNKNOWN",
                )
            )

            risk_score = int(
                float(
                    incident.get(
                        "risk_score",
                        0,
                    )
                )
            )

            exposure = float(
                incident.get(
                    "estimated_exposure",
                    0,
                )
            )

            if severity == "CRITICAL":

                icon = "🔴"

            elif severity == "HIGH":

                icon = "🟠"

            elif severity == "MEDIUM":

                icon = "🟡"

            else:

                icon = "🟢"

            with st.container(
                border=True
            ):

                col1, col2, col3, col4, col5 = (
                    st.columns(
                        [
                            1.2,
                            2.4,
                            1.2,
                            1.5,
                            1.5,
                        ],
                        gap="medium",
                    )
                )

                with col1:

                    st.markdown(
                        f"""
                        {icon}
                        **{severity}**
                        """
                    )

                with col2:

                    st.markdown(
                        f"""
                        **{incident_id}**

                        {incident_type}
                        """
                    )

                with col3:

                    visible_metric(
                        "Risk",
                        f"{risk_score}/100",
                    )

                with col4:

                    visible_metric(
                        "Exposure",
                        format_currency(
                            exposure
                        ),
                    )

                with col5:

                    if st.button(
                        "Investigate",
                        key=(
                            "escalation_"
                            f"{incident_id}"
                        ),
                        width="stretch",
                    ):

                        st.session_state[
                            "selected_incident_id"
                        ] = incident_id

                        st.session_state[
                            "escalation_selected"
                        ] = True

                        # Return to Command Center
                        # where the complete investigation
                        # is rendered at the top.
                        st.rerun()

        st.divider()

        # ====================================================
        # ESCALATION GUIDANCE
        # ====================================================

        st.header(
            "Incident Response Guidance"
        )

        guidance_col1, guidance_col2 = (
            st.columns(
                2,
                gap="large",
            )
        )

        with guidance_col1:

            st.markdown(
                """
                <div class="section-card">

                <h4>🔴 Critical</h4>

                <p>
                Immediate fraud-operations review.
                Consider transaction restrictions,
                account review and incident escalation.
                </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="section-card">

                <h4>🟠 High</h4>

                <p>
                Prioritized investigation with enhanced
                monitoring and customer/account review.
                </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with guidance_col2:

            st.markdown(
                """
                <div class="section-card">

                <h4>🟡 Medium</h4>

                <p>
                Continue investigation and monitor for
                additional linked suspicious activity.
                </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="section-card">

                <h4>🟢 Low</h4>

                <p>
                Maintain monitoring and allow the
                detection engine to collect additional evidence.
                </p>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# AI FRAUD ASSISTANT
# ============================================================

elif page == "AI Fraud Assistant":

    render_ai_fraud_assistant(
        load_incidents(),
        prepare_transaction_data(
            load_transactions(),
            load_incidents(),
        ),
        selected_incident=get_selected_incident(),
    )


# ============================================================
# LIVE RISK SIMULATOR
# ============================================================

elif page == "Live Risk Simulator":

    st.title(
        "⚡ Live Risk Simulator"
    )

    st.caption(
        "Search an existing transaction or customer "
        "and simulate its fraud-risk assessment."
    )

    transactions = prepare_transaction_data(
        load_transactions(),
        load_incidents(),
    )

    if transactions.empty:

        st.error(
            "Transaction dataset unavailable."
        )

    else:

        # ====================================================
        # ADD NEW TRANSACTION
        # ====================================================

        st.subheader(
            "➕ Add New Transaction"
        )

        st.caption(
            "Add a transaction directly to the existing CSV. "
            "The same transaction is then sent through the "
            "existing fraud-risk prediction API."
        )

        with st.form(
            "add_new_transaction_form",
            clear_on_submit=False,
        ):

            add_col1, add_col2, add_col3 = st.columns(
                3,
                gap="medium",
            )

            with add_col1:
                new_customer_id = st.text_input(
                    "Customer ID",
                    placeholder="CUS_012345",
                    key="new_tx_customer_id",
                )

            with add_col2:
                new_merchant_id = st.text_input(
                    "Merchant ID",
                    placeholder="MER_000001",
                    key="new_tx_merchant_id",
                )

            with add_col3:
                new_amount = st.number_input(
                    "Amount (₹)",
                    min_value=0.01,
                    value=1000.0,
                    step=100.0,
                    key="new_tx_amount",
                )

            add_col4, add_col5, add_col6 = st.columns(
                3,
                gap="medium",
            )

            with add_col4:
                new_timestamp = st.text_input(
                    "Timestamp",
                    value=pd.Timestamp.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    key="new_tx_timestamp",
                )

            with add_col5:
                new_payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "UPI",
                        "CREDIT_CARD",
                        "DEBIT_CARD",
                        "NETBANKING",
                        "WALLET",
                    ],
                    key="new_tx_payment_method",
                )

            with add_col6:
                new_account_age = st.number_input(
                    "Account Age (days)",
                    min_value=1,
                    value=365,
                    step=1,
                    key="new_tx_account_age",
                )

            add_col7, add_col8, add_col9 = st.columns(
                3,
                gap="medium",
            )

            with add_col7:
                new_device_id = st.text_input(
                    "Device ID",
                    placeholder="DEV_001234",
                    key="new_tx_device_id",
                )

            with add_col8:
                new_ip_id = st.text_input(
                    "IP ID",
                    placeholder="IP_001234",
                    key="new_tx_ip_id",
                )

            with add_col9:
                new_address_id = st.text_input(
                    "Address ID",
                    placeholder="ADDR_001234",
                    key="new_tx_address_id",
                )

            new_location = st.text_input(
                "Location",
                placeholder="Mumbai",
                key="new_tx_location",
            )

            add_transaction_clicked = st.form_submit_button(
                "➕ ADD TRANSACTION",
                width="stretch",
            )

        if add_transaction_clicked:

            validation_errors = []

            if not new_customer_id.strip():
                validation_errors.append(
                    "Customer ID is required."
                )

            if not new_merchant_id.strip():
                validation_errors.append(
                    "Merchant ID is required."
                )

            if not new_device_id.strip():
                validation_errors.append(
                    "Device ID is required."
                )

            if not new_ip_id.strip():
                validation_errors.append(
                    "IP ID is required."
                )

            if not new_address_id.strip():
                validation_errors.append(
                    "Address ID is required."
                )

            if not new_location.strip():
                validation_errors.append(
                    "Location is required."
                )

            try:
                parsed_timestamp = pd.Timestamp(
                    new_timestamp
                )

                if pd.isna(parsed_timestamp):
                    raise ValueError()

                normalized_timestamp = (
                    parsed_timestamp.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

            except Exception:
                validation_errors.append(
                    "Timestamp must be a valid date/time."
                )
                normalized_timestamp = ""

            if validation_errors:

                for validation_error in validation_errors:
                    st.error(validation_error)

            else:

                try:

                    existing_raw = pd.read_csv(
                        TRANSACTION_FILE
                    )

                    existing_ids = set(
                        existing_raw[
                            "transaction_id"
                        ]
                        .fillna("")
                        .astype(str)
                    ) if (
                        "transaction_id"
                        in existing_raw.columns
                    ) else set()

                    counter = 1

                    while (
                        f"TXN_NEW_{counter:06d}"
                        in existing_ids
                    ):
                        counter += 1

                    new_transaction_id = (
                        f"TXN_NEW_{counter:06d}"
                    )

                    raw_transaction = {
                        "transaction_id":
                            new_transaction_id,

                        "customer_id":
                            new_customer_id.strip().upper(),

                        "merchant_id":
                            new_merchant_id.strip().upper(),

                        "amount":
                            float(new_amount),

                        "timestamp":
                            normalized_timestamp,

                        "payment_method":
                            new_payment_method,

                        "device_id":
                            new_device_id.strip().upper(),

                        "ip_id":
                            new_ip_id.strip().upper(),

                        "address_id":
                            new_address_id.strip().upper(),

                        "account_age_days":
                            int(new_account_age),

                        "location":
                            new_location.strip(),

                        "refund_count": 0,
                        "refund_amount": 0.0,
                        "chargeback_count": 0,
                        "chargeback_amount": 0.0,
                        "is_refund": 0,
                        "is_chargeback": 0,
                    }

                    with st.spinner(
                        "Running the existing fraud model..."
                    ):

                        prediction = (
                            predict_new_transaction(
                                raw_transaction
                            )
                        )

                        complete_transaction = (
                            enrich_new_transaction_with_prediction(
                                raw_transaction,
                                prediction,
                            )
                        )

                        # Persist the scored transaction itself first.
                        append_transaction_to_csv(
                            complete_transaction
                        )

                        # Run the same live decision engine used by the
                        # on-screen assessment. This is the system-of-
                        # record risk decision for the new transaction.
                        live_transaction_decision = (
                            calculate_live_risk_decision(
                                prediction,
                                complete_transaction,
                            )
                        )

                        # Persist a complete audit/report row containing
                        # the transaction properties, model result,
                        # evidence, abuse risk, final score and action.
                        escalation = (
                            create_incident_from_new_transaction(
                                complete_transaction,
                                prediction,
                                live_transaction_decision,
                            )
                        )

                        persist_transaction_risk_report(
                            complete_transaction,
                            prediction,
                            live_transaction_decision,
                            escalated=escalation[
                                "escalated"
                            ],
                            incident_id=escalation[
                                "incident_id"
                            ],
                            incident_severity=escalation[
                                "severity"
                            ],
                            incident_type=escalation[
                                "incident_type"
                            ],
                            incident_reason=escalation[
                                "reason"
                            ],
                        )

                    st.session_state[
                        "last_added_transaction"
                    ] = complete_transaction

                    st.session_state[
                        "last_added_prediction"
                    ] = prediction

                    st.session_state[
                        "last_live_decision"
                    ] = live_transaction_decision

                    st.session_state[
                        "last_escalation"
                    ] = escalation

                    if escalation["escalated"]:
                        st.warning(
                            f"Transaction {new_transaction_id} "
                            f"was saved and escalated to "
                            f"{escalation['incident_id']} "
                            f"({escalation['severity']}). "
                            "It is now available in Incident Centre."
                        )
                    else:
                        st.success(
                            f"Transaction {new_transaction_id} "
                            "was saved to transactions.csv and "
                            "a persistent risk report was created. "
                            "No incident escalation was required."
                        )

                except requests.RequestException as error:

                    st.error(
                        "The transaction could not be scored "
                        "because the fraud-risk API is unavailable."
                    )

                    st.code(
                        str(error)
                    )

                except Exception as error:

                    st.error(
                        "The transaction could not be added."
                    )

                    st.exception(error)

        # ====================================================
        # NEW TRANSACTION RESULT
        # ====================================================

        if (
            "last_added_prediction"
            in st.session_state
        ):

            latest_prediction = (
                st.session_state[
                    "last_added_prediction"
                ]
            )

            latest_transaction = (
                st.session_state.get(
                    "last_added_transaction",
                    {},
                )
            )

            latest_risk = latest_prediction.get(
                "risk",
                {},
            )

            live_decision = calculate_live_risk_decision(
                latest_prediction,
                latest_transaction,
            )

            st.markdown(
                "### Latest Transaction Risk Assessment"
            )

            result_col1, result_col2, result_col3, result_col4 = (
                st.columns(
                    4,
                    gap="medium",
                )
            )

            with result_col1:
                visible_metric(
                    "Transaction ID",
                    latest_transaction.get(
                        "transaction_id",
                        "N/A",
                    ),
                )

            with result_col2:
                visible_metric(
                    "ML Risk Score",
                    f"{live_decision['ml_score']:.2f}/100",
                )

            with result_col3:
                visible_metric(
                    "Final Risk Score",
                    f"{live_decision['final_score']:.2f}/100",
                )

            with result_col4:
                visible_metric(
                    "Recommended Action",
                    live_decision[
                        "recommended_action"
                    ],
                )

            escalation_state = st.session_state.get(
                "last_escalation",
                {},
            )

            if escalation_state.get(
                "escalated",
                False,
            ):
                st.error(
                    f"🚨 Escalated to Incident Centre: "
                    f"**{escalation_state.get('incident_id', 'N/A')}** "
                    f"· {escalation_state.get('severity', 'N/A')}"
                )
            else:
                st.success(
                    "No incident escalation required for this transaction."
                )

            st.info(
                "This transaction is persisted in transactions.csv, "
                "scored using the existing fraud model and live evidence "
                "engine, and written to the persistent risk-report audit "
                "file. If the final risk reaches the escalation threshold, "
                "an incident is created and appears in Incident Centre. "
                "The transaction does not automatically retrain the model."
            )

            render_prediction_evidence(
                latest_prediction
            )

            render_human_in_loop_decision(
                latest_prediction,
                latest_transaction,
            )

            st.markdown(
                "### 📄 Persistent Risk Report"
            )

            latest_escalation = st.session_state.get(
                "last_escalation",
                {},
            )

            report_status = (
                "ESCALATED → "
                + str(
                    latest_escalation.get(
                        "incident_id",
                        "",
                    )
                )
                if latest_escalation.get(
                    "escalated",
                    False,
                )
                else "RECORDED → No escalation"
            )

            st.write(
                f"**Report status:** {report_status}"
            )

            st.caption(
                "The complete transaction properties, model output, "
                "deterministic evidence, abuse-risk score, final risk, "
                "recommendation and escalation status are persisted in "
                "reports/transaction_risk_reports.csv."
            )

            render_related_transactions(
                latest_transaction
            )

            st.divider()

        # ====================================================
        # SEARCH MODE
        # ====================================================

        st.subheader(
            "Transaction Search"
        )

        search_mode = st.radio(
            "Search by",
            [
                "Transaction ID",
                "Customer ID",
                "Merchant ID",
            ],
            horizontal=True,
            key="live_search_mode",
        )

        search_value = st.text_input(
            search_mode,
            placeholder=(
                "Enter "
                + search_mode
            ),
            key="live_search_value",
        )

        # ====================================================
        # SEARCH
        # ====================================================

        matched_transactions = (
            pd.DataFrame()
        )

        if search_value:

            value = (
                search_value
                .strip()
                .lower()
            )

            if search_mode == (
                "Transaction ID"
            ):

                search_column = (
                    "transaction_id"
                )

            elif search_mode == (
                "Customer ID"
            ):

                search_column = (
                    "customer_id"
                )

            else:

                search_column = (
                    "merchant_id"
                )

            if (
                search_column
                in transactions.columns
            ):

                matched_transactions = (
                    transactions[
                        transactions[
                            search_column
                        ]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.contains(
                            value,
                            na=False,
                            regex=False,
                        )
                    ]
                    .copy()
                )

        # ====================================================
        # SEARCH RESULTS
        # ====================================================

        if (
            search_value
            and matched_transactions.empty
        ):

            st.warning(
                f"No transactions found for "
                f"{search_mode}: {search_value}"
            )

        elif not matched_transactions.empty:

            st.success(
                f"{len(matched_transactions):,} "
                "matching transaction(s) found."
            )

            # -----------------------------------------------
            # TRANSACTION SELECTION
            # -----------------------------------------------

            if len(
                matched_transactions
            ) > 1:

                transaction_options = (
                    matched_transactions[
                        "transaction_id"
                    ]
                    .astype(str)
                    .tolist()
                )

                selected_transaction_id = (
                    st.selectbox(
                        "Select Transaction",
                        transaction_options,
                        key=(
                            "live_transaction_"
                            "selection"
                        ),
                    )
                )

                selected_transaction = (
                    matched_transactions[
                        matched_transactions[
                            "transaction_id"
                        ]
                        .astype(str)
                        == str(
                            selected_transaction_id
                        )
                    ]
                    .iloc[0]
                )

            else:

                selected_transaction = (
                    matched_transactions.iloc[0]
                )

            # -----------------------------------------------
            # TRANSACTION DETAILS
            # -----------------------------------------------

            st.divider()

            st.subheader(
                "Transaction Details"
            )

            d1, d2, d3, d4 = (
                st.columns(
                    4,
                    gap="medium",
                )
            )

            with d1:

                visible_metric(
                    "Transaction ID",
                    str(
                        selected_transaction.get(
                            "transaction_id",
                            "N/A",
                        )
                    ),
                )

            with d2:

                visible_metric(
                    "Customer ID",
                    str(
                        selected_transaction.get(
                            "customer_id",
                            "N/A",
                        )
                    ),
                )

            with d3:

                amount = pd.to_numeric(
                    selected_transaction.get(
                        "amount",
                        0,
                    ),
                    errors="coerce",
                )

                if pd.isna(amount):
                    amount = 0

                visible_metric(
                    "Amount",
                    format_currency(
                        amount
                    ),
                )

            with d4:

                visible_metric(
                    "Payment Method",
                    str(
                        selected_transaction.get(
                            "payment_method",
                            "N/A",
                        )
                    ),
                )

            # -----------------------------------------------
            # RISK RESULT
            # -----------------------------------------------

            st.subheader(
                "Risk Assessment"
            )

            try:

                transaction_risk = int(
                    float(
                        selected_transaction.get(
                            "risk_score",
                            0,
                        )
                    )
                )

            except Exception:

                transaction_risk = 0

            transaction_level = str(
                selected_transaction.get(
                    "risk_level",
                    "LOW",
                )
            ).upper()

            if transaction_risk >= 90:

                transaction_level = (
                    "CRITICAL"
                )

            elif transaction_risk >= 75:

                transaction_level = (
                    "HIGH"
                )

            elif transaction_risk >= 50:

                transaction_level = (
                    "MEDIUM"
                )

            else:

                transaction_level = (
                    "LOW"
                )

            risk_col1, risk_col2, risk_col3 = (
                st.columns(
                    3,
                    gap="medium",
                )
            )

            with risk_col1:

                visible_metric(
                    "Risk Score",
                    f"{transaction_risk}/100",
                )

            with risk_col2:

                visible_metric(
                    "Risk Level",
                    transaction_level,
                )

            with risk_col3:

                fraud_type = str(
                    selected_transaction.get(
                        "fraud_type",
                        "",
                    )
                )

                if (
                    not fraud_type
                    or fraud_type
                    in {
                        "nan",
                        "None",
                    }
                ):

                    fraud_type = "Normal"

                visible_metric(
                    "Fraud Signal",
                    pretty_type(
                        fraud_type
                    ),
                )

            # -----------------------------------------------
            # INCIDENT INFORMATION
            # -----------------------------------------------

            incident_id = str(
                selected_transaction.get(
                    "incident_id",
                    "",
                )
            ).strip()

            if incident_id and incident_id.lower() not in {
                "nan",
                "none",
            }:

                st.info(
                    f"This transaction is linked to incident "
                    f"**{incident_id}**."
                )

            # -----------------------------------------------
            # TRANSACTION SIGNALS
            # -----------------------------------------------

            st.divider()

            st.subheader(
                "Risk Signals"
            )

            signal_columns = [
                (
                    "transaction_count_last_hour",
                    "Transactions in Last Hour",
                ),
                (
                    "device_customer_count",
                    "Customers Sharing Device",
                ),
                (
                    "ip_customer_count",
                    "Customers Sharing IP",
                ),
                (
                    "address_customer_count",
                    "Customers Sharing Address",
                ),
                (
                    "payment_method_frequency",
                    "Payment Method Frequency",
                ),
                (
                    "amount_to_merchant_average",
                    "Amount / Merchant Average",
                ),
                (
                    "merchant_refund_ratio",
                    "Merchant Refund Ratio",
                ),
                (
                    "merchant_chargeback_ratio",
                    "Merchant Chargeback Ratio",
                ),
            ]

            available_signals = [
                item
                for item in signal_columns
                if item[0]
                in selected_transaction.index
            ]

            if available_signals:

                signal_columns_ui = st.columns(
                    4,
                    gap="medium",
                )

                for index, (
                    column,
                    label,
                ) in enumerate(
                    available_signals
                ):

                    value = selected_transaction.get(
                        column,
                        0,
                    )

                    try:

                        numeric_value = float(
                            value
                        )

                        if pd.isna(
                            numeric_value
                        ):

                            numeric_value = 0

                    except Exception:

                        numeric_value = 0

                    with signal_columns_ui[
                        index
                        % 4
                    ]:

                        visible_metric(
                            label,
                            (
                                f"{numeric_value:.2f}"
                                if isinstance(
                                    numeric_value,
                                    float,
                                )
                                else
                                str(
                                    numeric_value
                                )
                            ),
                        )

            else:

                st.info(
                    "No additional behavioral signals "
                    "are available for this transaction."
                )

            # -----------------------------------------------
            # FULL TRANSACTION RECORD
            # -----------------------------------------------

            with st.expander(
                "View Full Transaction Record"
            ):

                transaction_record = (
                    selected_transaction
                    .to_frame(
                        "Value"
                    )
                )

                transaction_record.index.name = (
                    "Field"
                )

                st.dataframe(
                    transaction_record,
                    width="stretch",
                )

        # ====================================================
        # SIMULATION MODE
        # ====================================================

        st.divider()

        st.subheader(
            "Manual Risk Simulation"
        )

        st.caption(
            "Enter transaction characteristics to "
            "estimate the risk level."
        )

        sim1, sim2, sim3 = (
            st.columns(
                3,
                gap="medium",
            )
        )

        with sim1:

            sim_amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=5000.0,
                step=500.0,
                key="sim_amount",
            )

            sim_velocity = st.number_input(
                "Transactions in Last Hour",
                min_value=0,
                value=1,
                step=1,
                key="sim_velocity",
            )

        with sim2:

            sim_device_customers = (
                st.number_input(
                    "Customers Sharing Device",
                    min_value=1,
                    value=1,
                    step=1,
                    key="sim_device",
                )
            )

            sim_ip_customers = (
                st.number_input(
                    "Customers Sharing IP",
                    min_value=1,
                    value=1,
                    step=1,
                    key="sim_ip",
                )
            )

        with sim3:

            sim_account_age = (
                st.number_input(
                    "Account Age (days)",
                    min_value=0,
                    value=365,
                    step=1,
                    key="sim_age",
                )
            )

            sim_payment_frequency = (
                st.number_input(
                    "Payment Method Frequency",
                    min_value=0,
                    value=1,
                    step=1,
                    key="sim_payment_frequency",
                )
            )

        # ====================================================
        # CALCULATE SIMULATED RISK
        # ====================================================

        if st.button(
            "Run Risk Simulation",
            type="primary",
            key="run_risk_simulation",
        ):

            simulated_score = 0

            # Velocity signal
            if sim_velocity >= 20:

                simulated_score += 35

            elif sim_velocity >= 10:

                simulated_score += 25

            elif sim_velocity >= 5:

                simulated_score += 15

            # Device reuse
            if sim_device_customers >= 10:

                simulated_score += 25

            elif sim_device_customers >= 5:

                simulated_score += 15

            elif sim_device_customers >= 3:

                simulated_score += 8

            # IP reuse
            if sim_ip_customers >= 10:

                simulated_score += 20

            elif sim_ip_customers >= 5:

                simulated_score += 12

            elif sim_ip_customers >= 3:

                simulated_score += 6

            # Account age
            if sim_account_age <= 7:

                simulated_score += 15

            elif sim_account_age <= 30:

                simulated_score += 8

            # Payment frequency
            if sim_payment_frequency >= 20:

                simulated_score += 15

            elif sim_payment_frequency >= 10:

                simulated_score += 8

            # Amount
            if sim_amount >= 100000:

                simulated_score += 15

            elif sim_amount >= 50000:

                simulated_score += 10

            elif sim_amount >= 25000:

                simulated_score += 5

            simulated_score = min(
                100,
                simulated_score,
            )

            if simulated_score >= 90:

                simulated_level = (
                    "CRITICAL"
                )

                simulated_action = (
                    "Immediate investigation and "
                    "risk controls recommended."
                )

            elif simulated_score >= 75:

                simulated_level = (
                    "HIGH"
                )

                simulated_action = (
                    "Enhanced verification and "
                    "priority review recommended."
                )

            elif simulated_score >= 50:

                simulated_level = (
                    "MEDIUM"
                )

                simulated_action = (
                    "Continue monitoring and "
                    "review linked signals."
                )

            else:

                simulated_level = (
                    "LOW"
                )

                simulated_action = (
                    "No immediate intervention required; "
                    "continue monitoring."
                )

            st.session_state[
                "live_prediction"
            ] = {
                "score":
                    simulated_score,

                "level":
                    simulated_level,

                "action":
                    simulated_action,
            }

        # ====================================================
        # DISPLAY SIMULATION RESULT
        # ====================================================

        prediction = (
            st.session_state.get(
                "live_prediction"
            )
        )

        if prediction is not None:

            st.divider()

            st.subheader(
                "Simulation Result"
            )

            p1, p2, p3 = (
                st.columns(
                    3,
                    gap="medium",
                )
            )

            with p1:

                visible_metric(
                    "Risk Score",
                    f"{prediction['score']}/100",
                )

            with p2:

                visible_metric(
                    "Risk Level",
                    prediction["level"],
                )

            with p3:

                if (
                    prediction["level"]
                    == "CRITICAL"
                ):

                    st.error(
                        prediction[
                            "action"
                        ]
                    )

                elif (
                    prediction["level"]
                    == "HIGH"
                ):

                    st.warning(
                        prediction[
                            "action"
                        ]
                    )

                elif (
                    prediction["level"]
                    == "MEDIUM"
                ):

                    st.info(
                        prediction[
                            "action"
                        ]
                    )

                else:

                    st.success(
                        prediction[
                            "action"
                        ]
                    )


# ============================================================
# GLOBAL FOOTER
# ============================================================

st.divider()

st.caption(
    "Merchant Risk Sentinel · AI-powered "
    "fraud detection and incident response"
)