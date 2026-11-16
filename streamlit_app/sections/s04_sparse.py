import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import rag_core as rc
import ui

PRESETS = [
    "Sony WH-1000XM4",
    "RTX 4060 laptop",
    "65W USB-C charger",
    "IP68 earbuds",
    "something cheap for the gym that will not fall out of my ears",
]

TOKEN_SAMPLES = [
    "Sony WH-1000XM5 Bluetooth Headphones",
    "65W USB-C chargers",
    "earbuds with IP68 rating",
]


@st.cache_resource(show_spinner=False)
def reference_index(texts: tuple[str, ...]):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([rc.preprocess(text) for text in texts])


def _search(index: rc.BM25Index, engine, query: str, k: int, chunk_type: str | None):
    scores = index.term_scores(query)["scores"]
    order = scores.argsort()[::-1]
    rows = []
    for idx in order:
        score = float(scores[idx])
        if score <= 0:
            break
        row = engine.chunks.iloc[int(idx)]
        if chunk_type and row["chunk_type"] != chunk_type:
            continue
        rows.append({
            "doc_id": row["doc_id"],
            "product_id": row["doc_id"].split(":")[0],
            "chunk_type": row["chunk_type"],
            "score": round(score, 3),
            "title": row["metadata"].get("sku", "-"),
            "text": row["text"],
            "category": row["category"],
            "price_inr": int(row["metadata"].get("price_inr") or 0),
        })
        if len(rows) >= k:
            break
    return rc.enrich(engine, pd.DataFrame(rows, columns=["doc_id", "product_id", "chunk_type",
                                                          "score", "title", "text",
                                                          "category", "price_inr"]))


def render() -> None:
    engine = ui.get_engine()

    ui.page_header(
        "Sparse retrieval: BM25 from scratch",
        "Exact tokens, weighted by how informative they are. Live k1 and b parameters, "
        "a term-contribution table, and a failure you can hear coming.",
        tag="Lexical",
    )

    ui.kpi_row([
        ("vocabulary", f"{len(engine.bm25.vocab):,}", "sparse"),
        ("avg doc length", f"{engine.bm25.avgdl:.1f} tokens", "sparse"),
        ("k1 default", "1.5", "graph"),
        ("b default", "0.75", "hybrid"),
    ])

    left, right = ui.demo_panel()
    with left:
        query = ui.query_input("sparse", PRESETS, PRESETS[1])
        controls = st.columns(3)
        k = controls[0].slider("Top-k", 3, 12, 6, key="sp_k")
        k1 = controls[1].slider("k1 — term saturation", 0.0, 3.0, 1.5, 0.1, key="sp_k1")
        b = controls[2].slider("b — length normalisation", 0.0, 1.0, 0.75, 0.05, key="sp_b")

        started = time.perf_counter()
        index = rc.BM25Index(engine.chunks["text"].tolist(), k1=k1, b=b)
        frame = _search(index, engine, query, k, None)
        elapsed = (time.perf_counter() - started) * 1000

        ui.latency_chips({"sparse": elapsed})
        score_max = float(frame["score"].max()) if len(frame) else 1.0
        ui.results_table(frame, score_column="score", score_max=score_max,
                         caption="BM25 scores are unbounded; the bar is relative to the top hit.")
        ui.passage(frame)

        st.markdown("#### Why did the top document win?")
        if len(frame):
            top_id = frame.iloc[0]["doc_id"]
            doc_index = int(engine.chunks.index[engine.chunks["doc_id"] == top_id][0])
            terms = index.explain(query, doc_index)
            explain_cols = st.columns([1.1, 1])
            with explain_cols[0]:
                st.dataframe(terms, hide_index=True, use_container_width=True)
            with explain_cols[1]:
                fig = charts.score_breakdown(terms)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            st.caption(
                f"Total score {terms['contribution'].sum():.3f} = sum of the term "
                "contributions above. This audit trail is impossible with a dense model."
            )
    with right:
        ui.note_card("bm25")

    ui.divider("Tokenisation — the decision that decides recall")

    token_left, token_right = ui.demo_panel()
    with token_left:
        for sample in TOKEN_SAMPLES:
            tokens = rc.tokenize(sample)
            stems = rc.preprocess(sample)
            st.markdown(
                f'<div class="lab-card" style="border:1px solid var(--lab-border);'
                f'border-radius:10px;padding:.6rem .8rem;margin-bottom:.5rem">'
                f'<div style="font-size:.86rem;font-weight:650">{sample}</div>'
                f'<div class="lab-chip-row" style="margin:.4rem 0 .1rem 0">'
                + "".join(f'<span class="lab-chip">{t}</span>' for t in tokens)
                + '</div><div style="font-size:.72rem;color:#64748b">after light stemming: '
                + " ".join(stems) + "</div></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Hyphenated identifiers emit both forms, so `WH-1000XM5` and `WH1000XM5` "
            "both match. Stopwords are removed before scoring."
        )
        with st.expander("Cross-check against the reference implementation (rank_bm25)"):
            reference = reference_index(tuple(engine.chunks["text"].tolist()))
            rows = []
            for probe in ["Sony WH-1000XM4", "65W USB-C charger", "IP68 earbuds"]:
                ours = _search(engine.bm25, engine, probe, 10, None)["doc_id"].tolist()
                ref_scores = reference.get_scores(rc.preprocess(probe))
                ref_ids = [engine.chunks.iloc[int(i)]["doc_id"]
                           for i in ref_scores.argsort()[::-1][:10]]
                rows.append({"query": probe, "top-10 overlap": f"{len(set(ours) & set(ref_ids))}/10"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with token_right:
        ui.note_card("tokenize")

    ui.divider("The failure mode: vocabulary mismatch")

    fail_left, fail_right = ui.demo_panel()
    with fail_left:
        fail_query = "something cheap for the gym that will not fall out of my ears"
        st.markdown(f"**Query:** `{fail_query}`")
        result = _search(engine.bm25, engine, fail_query, 6, None)
        ui.results_table(result, score_column="score",
                         score_max=float(result["score"].max()) if len(result) else 1.0)
        matched = [token for token in rc.preprocess(fail_query) if token in engine.bm25.vocab]
        st.markdown(f"Query terms that exist in the vocabulary: `{matched}`")
        if len(result):
            st.markdown(
                f"Top hits cost ₹{int(result['price_inr'].min()):,}–₹{int(result['price_inr'].max()):,}. "
                "The words *gym*, *fall* and *workout* are absent from the catalogue, so the "
                "scorer latched onto *cheap* and *ear* instead of intent."
            )
        suitable = engine.products[
            (engine.products["category"] == "Earbuds")
            & (engine.products["specs"].apply(lambda s: s.get("ip_rating") in {"IPX5", "IP55", "IP57", "IP68"}))
            & (engine.products["price_inr"] <= 3000)
        ]
        st.markdown(f"**What the user actually wanted** ({len(suitable)} products, none in the top-6):")
        st.dataframe(
            suitable[["product_id", "title", "price_inr"]].assign(
                ip_rating=suitable["specs"].apply(lambda s: s.get("ip_rating"))),
            hide_index=True, use_container_width=True,
            column_config={"price_inr": st.column_config.NumberColumn("price", format="₹%d")},
        )
    with fail_right:
        ui.note_card("sparsefailure")
