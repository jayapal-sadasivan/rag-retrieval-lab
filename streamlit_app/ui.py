"""Reusable UI components for the RAG Retrieval Lab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import content
import rag_core as rc

ASSETS = Path(__file__).parent / "assets"


@st.cache_resource(show_spinner="Loading catalogue, embeddings, BM25 index and graph…")
def get_engine() -> rc.Engine:
    return rc.Engine()


def is_dark() -> bool:
    return st.session_state.get("theme", "Light") == "Dark"


def inject_css() -> None:
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    if is_dark():
        st.markdown(
            """
            <style>
            :root {
              --lab-bg: #0b1220;
              --lab-panel: #111a2b;
              --lab-border: #24324a;
              --lab-text: #e6edf7;
              --lab-muted: #94a3b8;
              --lab-code: #1c2740;
            }
            .stApp { background: var(--lab-bg); color: var(--lab-text); }
            .lab-hero { background: linear-gradient(135deg,#0f1b30,#111a2b); }
            .lab-card, .lab-kpi, .lab-callout {
              background: #111a2b !important; border-color: #24324a !important;
            }
            .lab-hero h1, .lab-note-head, .lab-callout-title, .lab-kpi-value { color: #e6edf7 !important; }
            .lab-hero p, .lab-kpi-label, .lab-note-title, .lab-divider { color: #94a3b8 !important; }
            .lab-note-body, .lab-callout-body { color: #cbd5e1 !important; }
            .lab-note-block { border-color: #24324a; }
            .lab-chip { background: #1c2740; color: #e2e8f0; border-color: #24324a; }
            .lab-lat { background: #111a2b; color: #e6edf7; border-color: #24324a; }
            div[data-testid="stColumn"]:has(.lab-note-title) div[data-testid="stVerticalBlockBorderWrapper"] {
              background: #0f1828 !important; border-color: #24324a !important;
            }
            div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
              background: #0f1828 !important; color: #e6edf7 !important;
              border-color: #24324a !important;
            }
            div[data-baseweb="select"] > div { background: #0f1828 !important; }
            section[data-testid="stSidebar"] { background: #0b1220; }
            section[data-testid="stSidebar"] * { color: #e6edf7; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def page_header(title: str, subtitle: str, tag: str | None = None) -> None:
    badge = f'<span class="lab-tag">{tag}</span>' if tag else ""
    st.markdown(
        f"""
        <div class="lab-hero">
          <div class="lab-hero-left">
            <div class="lab-eyebrow">RAG Retrieval Lab</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def demo_panel(ratio: tuple[float, float] = (1.65, 1.0)):
    """Two columns: experiment on the left, explanation rail on the right."""
    return st.columns(list(ratio), gap="large")


def note_card(key: str, compact: bool = False) -> None:
    entry = content.note(key)
    blocks = [
        ("🔍", "What you are seeing", entry["see"]),
        ("🧠", "Why it happens", entry["why"]),
        ("⚠️", "When this breaks", entry["breaks"]),
        ("🧪", "Try next", entry["try"]),
    ]
    with st.container(border=True):
        st.markdown('<div class="lab-note-title">Explain this demo</div>', unsafe_allow_html=True)
        for icon, title, body in blocks:
            weight = "lab-note-body" if compact else "lab-note-body"
            st.markdown(
                f'<div class="lab-note-block"><span class="lab-note-icon">{icon}</span>'
                f'<div><div class="lab-note-head">{title}</div>'
                f'<div class="{weight}">{body}</div></div></div>',
                unsafe_allow_html=True,
            )


def kpi_row(items: list[tuple[str, str, str | None]]):
    columns = st.columns(len(items), gap="small")
    for column, (label, value, accent) in zip(columns, items):
        color = rc.PALETTE.get(accent, "#334155") if accent else "#334155"
        column.markdown(
            f"""
            <div class="lab-kpi" style="border-top:3px solid {color}">
              <div class="lab-kpi-value" style="color:{color}">{value}</div>
              <div class="lab-kpi-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def method_badge(name: str) -> str:
    return f'<span class="lab-badge lab-badge-{name}">{rc.RETRIEVER_LABELS[name]}</span>'


def chips(items: list[tuple[str, str]]) -> None:
    html = "".join(
        f'<span class="lab-chip {css}">{label}</span>'
        for label, css in items
    )
    st.markdown(f'<div class="lab-chip-row">{html}</div>', unsafe_allow_html=True)


def query_input(prefix: str, presets: list[str], default: str) -> str:
    text_key = f"qtext_{prefix}"
    pills_key = f"qpills_{prefix}"
    if text_key not in st.session_state:
        st.session_state[text_key] = default

    def from_preset():
        value = st.session_state.get(pills_key)
        if value:
            st.session_state[text_key] = value

    def from_text():
        if st.session_state.get(text_key) != st.session_state.get(pills_key):
            st.session_state[pills_key] = None

    st.pills("Teaching queries", presets, key=pills_key, on_change=from_preset,
             label_visibility="collapsed")
    return st.text_input("Your query", key=text_key, on_change=from_text,
                         placeholder="Type a query…")


def results_table(frame: pd.DataFrame, score_column: str = "score",
                  score_max: float | None = None, caption: str | None = None,
                  max_rows: int = 8) -> None:
    required = {"doc_id", "title", "category", "price_inr"}
    if frame is None or len(frame) == 0 or not required.issubset(frame.columns):
        st.info("No results. That is itself a result — check the retriever routing.")
        return

    view = frame.head(max_rows).copy()
    view.insert(0, "rank", range(1, len(view) + 1))
    display = pd.DataFrame({
        "#": view["rank"],
        "Result": view["title"].astype(str).str.slice(0, 74),
        "Chunk": view.get("chunk", view["doc_id"].str.split(":").str[1]).astype(str),
        "Category": view["category"].astype(str),
        "Price": view["price_inr"].fillna(0).astype(int),
        "Score": pd.to_numeric(view[score_column], errors="coerce").fillna(0.0),
        "id": view["doc_id"],
    })

    upper = score_max if score_max is not None else float(max(1e-6, display["Score"].max()))
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Result": st.column_config.TextColumn(width="large"),
            "Chunk": st.column_config.TextColumn(width="small"),
            "Category": st.column_config.TextColumn(width="small"),
            "Price": st.column_config.NumberColumn(format="₹%d", width="small"),
            "Score": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=upper, width="medium"),
            "id": st.column_config.TextColumn("chunk id", width="small"),
        },
        key=f"tbl_{abs(hash((caption, display['id'].iloc[0], len(display))))}",
    )
    if caption:
        st.caption(caption)


def passage(frame: pd.DataFrame, index: int = 0) -> None:
    if frame is None or len(frame) == 0:
        return
    row = frame.iloc[index]
    with st.expander(f"Read the retrieved passage — {row['doc_id']}", expanded=False):
        st.markdown(f"```text\n{row['text']}\n```")


def latency_chips(values: dict[str, float]) -> None:
    parts = []
    for name, ms in values.items():
        parts.append(
            f'<span class="lab-lat lab-lat-{name}">{name} {ms:.1f} ms</span>'
        )
    st.markdown(
        '<div class="lab-lat-row">' + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def verdict(ok: bool, yes: str = "PASS", no: str = "FAIL") -> str:
    css = "lab-pass" if ok else "lab-fail"
    return f'<span class="{css}">{yes if ok else no}</span>'


def divider(label: str = "") -> None:
    st.markdown(f'<div class="lab-divider">{label}</div>', unsafe_allow_html=True)


def callout(title: str, body: str, tone: str = "info") -> None:
    st.markdown(
        f'<div class="lab-callout lab-callout-{tone}">'
        f'<div class="lab-callout-title">{title}</div>'
        f'<div class="lab-callout-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def setup_page(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} · RAG Retrieval Lab",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
