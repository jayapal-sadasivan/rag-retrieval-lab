import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import rag_core as rc
import ui

PRESETS = [
    "I take long flights and the engine drone gives me a headache",
    "headphones that silence aeroplane engine noise",
    "something to keep my phone alive on a long trek",
    "laptop for editing 4K video under 70000",
]

HEATMAP_SENTENCES = [
    "wireless earbuds with active noise cancellation",
    "headphones that block out engine noise on a plane",
    "a 65W USB-C charger for my laptop",
    "fast charger with USB-C output",
    "return policy for opened earbuds",
]


def _run(engine, query, k, category, max_price, chunk_type):
    started = time.perf_counter()
    frame = rc.semantic_search(
        engine, query, k=k,
        chunk_type=chunk_type or None,
        category=category or None,
        max_price=max_price if max_price else None,
    )
    return frame, (time.perf_counter() - started) * 1000


def render() -> None:
    engine = ui.get_engine()
    model_dim = engine.chunk_vecs.shape[1]

    ui.page_header(
        "Semantic retrieval: meaning as geometry",
        "Queries and chunks become 384-dimensional vectors. Similarity is proximity: "
        "no shared words required. Then watch exact identifiers break it.",
        tag="Dense",
    )

    ui.kpi_row([
        ("embedding model", "MiniLM-L6-v2", "semantic"),
        ("dimensions", f"{model_dim}", "semantic"),
        ("max tokens", "256", "sparse"),
        ("vectors cached", f"{engine.stats()['chunks']}", "graph"),
    ])

    left, right = ui.demo_panel()
    with left:
        query = ui.query_input("sem", PRESETS, PRESETS[0])
        controls = st.columns([1, 1, 1])
        k = controls[0].slider("Top-k", 3, 12, 6, key="sem_k")
        category = controls[1].selectbox(
            "Category filter", ["(none)"] + sorted(engine.products["category"].unique()),
            key="sem_cat")
        max_price = controls[2].select_slider(
            "Max price", options=[0, 10000, 25000, 50000, 100000, 200000], value=0,
            format_func=lambda value: "no filter" if value == 0 else f"₹{value:,}",
            key="sem_price")

        frame, elapsed = _run(engine, query, k, "" if category == "(none)" else category,
                              max_price, None)
        ui.latency_chips({"semantic": elapsed})
        ui.results_table(frame, score_column="score", caption="Cosine similarity (normalised, so 1.0 is identical).")
        ui.passage(frame)
        ui.chips([
            ("no keyword overlap required", "lab-chip-good"),
            ("numbers are invisible", "lab-chip-bad"),
        ])
    with right:
        ui.note_card("cosine")

    ui.divider("Structured filters: fixing what embeddings cannot see")

    filter_left, filter_right = ui.demo_panel()
    with filter_left:
        st.markdown("##### The same query, with and without metadata filters")
        budget_query = "laptop for editing 4K video under 70000 rupees"
        raw, _ = _run(engine, budget_query, 4, "", 0, None)
        filtered, _ = _run(engine, budget_query, 4, "Laptops", 70000, "desc")

        st.markdown(f"**Query:** `{budget_query}`")
        st.markdown("**Unfiltered semantic result**")
        if len(raw):
            top_price = int(raw.iloc[0]["price_inr"])
            st.markdown(
                f"Top-1 is **{raw.iloc[0]['title']}** at **₹{top_price:,}** "
                f"→ over budget: **{top_price > 70000}**"
            )
        ui.results_table(raw, score_column="score", max_rows=4)

        st.markdown("**With `category = Laptops` and `price_inr <= 70000`**")
        ui.results_table(filtered, score_column="score", max_rows=4)
        if len(filtered):
            prices = filtered["price_inr"].tolist()
            st.success(f"All results inside budget: {prices}")
    with filter_right:
        ui.note_card("filter")

    ui.divider("The failure mode: numbers are not meaning")

    fail_left, fail_right = ui.demo_panel()
    with fail_left:
        st.markdown("##### Ask for an exact GPU, then check the ground truth")
        fail_query = "RTX 4060 laptop"
        result, elapsed = _run(engine, fail_query, 6, "", 0, "desc")
        ui.results_table(result, score_column="score", caption="Dense top-6 for an exact specification.")
        truth = sorted(engine.products.loc[
            engine.products["specs"].apply(lambda s: "RTX 4060" in str(s.get("gpu", ""))), "product_id"
        ].tolist())
        top1 = result.iloc[0] if len(result) else None
        if top1 is not None:
            gpu = engine.products.set_index("product_id").loc[top1["product_id"], "specs"].get("gpu", "-")
            st.markdown(
                f"Dense top-1: **{top1['product_id']}** · GPU **{gpu}** → "
                f"{'CORRECT' if top1['product_id'] in truth else 'WRONG PRODUCT'}"
            )
        st.caption(f"Products that truly have an RTX 4060: {truth}")
    with fail_right:
        ui.note_card("failure")

    ui.divider("Inside the geometry")

    tab_cosine, tab_pca = st.tabs(["Cosine similarity between sentences", "The whole embedding space"])
    with tab_cosine:
        cosine_left, cosine_right = st.columns([1.5, 1], gap="large")
        with cosine_left:
            fig, sim = charts.cosine_figure(HEATMAP_SENTENCES)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with cosine_right:
            st.markdown("**Read the matrix**")
            st.markdown(
                f"""
                - S1 vs S2 (no shared words, same meaning): **{sim[0, 1]:.2f}**
                - S3 vs S4 (no shared words, same meaning): **{sim[2, 3]:.2f}**
                - S1 vs S5 (different topics): **{sim[0, 4]:.2f}**

                High similarity with no lexical overlap is the entire value of a
                bi-encoder. It is also why the model cannot tell `XM4` from `XM5`:
                the difference is a token, not a meaning.
                """
            )
            for index, sentence in enumerate(HEATMAP_SENTENCES, start=1):
                st.caption(f"S{index}: {sentence}")
    with tab_pca:
        fig = charts.pca_figure(engine, dark=ui.is_dark())
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Stars are query embeddings. Clusters are categories; only clusters are meaningful at 14% variance.")
