"""RAG Retrieval Lab — Streamlit entrypoint.

Run with:
    .venv\\Scripts\\python.exe -m streamlit run streamlit_app\\app.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
for noisy in ("huggingface_hub", "transformers", "sentence_transformers", "chromadb", "httpx"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import ui  # noqa: E402
from sections import (  # noqa: E402
    s01_overview,
    s02_dataset,
    s03_semantic,
    s04_sparse,
    s05_graph,
    s06_hybrid,
    s07_evaluation,
    s08_grounding,
    s09_playground,
)

st.set_page_config(
    page_title="RAG Retrieval Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("theme", "Light")
st.session_state.setdefault("presenter", False)

ui.inject_css()

engine = ui.get_engine()
stats = engine.stats()

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:.55rem;margin-bottom:.35rem">
          <div style="font-size:1.5rem">🔎</div>
          <div>
            <div style="font-weight:800;font-size:1.02rem;line-height:1.1">RAG Retrieval Lab</div>
            <div style="color:#64748b;font-size:.72rem">Semantic · Sparse · Graph</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"{stats['products']} products · {stats['chunks']} chunks · "
        f"{stats['graph_edges']:,} edges"
    )
    st.divider()
    st.radio("Theme", ["Light", "Dark"], key="theme", horizontal=True)
    st.caption("Styles the lab shell. Streamlit's own menu can theme native widgets.")
    st.toggle("Presenter mode (larger text)", key="presenter")
    st.divider()
    st.caption(
        "Companion to `RAG_Retrieval_Comparative_Tutorial.ipynb`. "
        "Every number here is reproduced by the notebook."
    )

if st.session_state["presenter"]:
    st.markdown(
        """
        <style>
        .lab-note-body { font-size: 0.95rem !important; }
        .lab-note-head { font-size: 0.92rem !important; }
        .lab-hero h1 { font-size: 2rem !important; }
        .lab-hero p { font-size: 1rem !important; }
        div[data-testid="stDataFrame"] { font-size: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

pages = [
    st.Page(s01_overview.render, title="Overview", icon="🧭", url_path="overview", default=True),
    st.Page(s02_dataset.render, title="Dataset", icon="🗂️", url_path="dataset"),
    st.Page(s03_semantic.render, title="Semantic retrieval", icon="🧠", url_path="semantic"),
    st.Page(s04_sparse.render, title="Sparse retrieval", icon="🔤", url_path="sparse"),
    st.Page(s05_graph.render, title="Graph retrieval", icon="🕸️", url_path="graph"),
    st.Page(s06_hybrid.render, title="Hybrid & routing", icon="🔀", url_path="hybrid"),
    st.Page(s07_evaluation.render, title="Evaluation", icon="📊", url_path="evaluation"),
    st.Page(s08_grounding.render, title="Grounded answers", icon="✅", url_path="grounding"),
    st.Page(s09_playground.render, title="Playground", icon="🎛️", url_path="playground"),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()
