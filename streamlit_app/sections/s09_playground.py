import time

import pandas as pd
import streamlit as st

import rag_core as rc
import ui


def render() -> None:
    engine = ui.get_engine()

    ui.page_header(
        "Playground: predict, then check",
        "Bring your own query. Choose retrievers and filters, then compare quality, "
        "latency and provenance. The goal is to predict the winner before you run it.",
        tag="Playground",
    )

    left, right = ui.demo_panel()
    with left:
        query = st.text_input("Your query", value="noise cancelling headphones for a long flight",
                              key="pg_query")
        controls = st.columns([1.3, 1, 1])
        methods = controls[0].multiselect(
            "Retrievers", ["semantic", "sparse", "graph", "hybrid"],
            default=["semantic", "sparse", "hybrid"], key="pg_methods",
            format_func=lambda name: rc.RETRIEVER_LABELS[name])
        k = controls[1].slider("Top-k", 3, 12, 5, key="pg_k")
        category = controls[2].selectbox(
            "Category", ["(all)"] + sorted(engine.products["category"].unique()), key="pg_cat")
        max_price = st.select_slider(
            "Max price", options=[0, 10000, 25000, 50000, 100000, 200000], value=0,
            format_func=lambda value: "no filter" if value == 0 else f"₹{value:,}",
            key="pg_price")

        if not methods:
            st.warning("Pick at least one retriever.")
            return

        results = {}
        timings = {}
        for method in methods:
            started = time.perf_counter()
            frame = rc.search(engine, method, query, k=k,
                              category="" if category == "(all)" else category,
                              max_price=max_price if max_price else None)
            timings[method] = (time.perf_counter() - started) * 1000
            results[method] = frame

        ui.latency_chips(timings)

        tabs = st.tabs([rc.RETRIEVER_LABELS[method] for method in methods])
        for tab, method in zip(tabs, methods):
            with tab:
                frame = results[method]
                score_column = "score" if "score" in frame.columns else "rrf_score"
                score_max = None if method in ("semantic", "hybrid") else (
                    float(frame[score_column].max()) if len(frame) else 1.0)
                ui.results_table(frame, score_column=score_column, score_max=score_max,
                                 max_rows=k)
                ui.passage(frame)

        export = results[methods[0]].copy()
        if len(export):
            columns = [column for column in ("doc_id", "title", "category", "price_inr", "chunk")
                       if column in export.columns]
            export = export[columns]
        st.download_button(
            "Download first retriever's results (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="rag_lab_results.csv",
            mime="text/csv",
            key="pg_download",
        )
    with right:
        ui.note_card("playground")

    ui.divider("Prediction card")

    prediction_left, prediction_right = st.columns([1.65, 1], gap="large")
    with prediction_left:
        st.markdown(
            """
            Ask the room to commit to an answer before pressing run:

            | If the query… | predict | because |
            |---|---|---|
            | contains a SKU or model number | **sparse** | exact token, huge IDF, no synonym drift |
            | describes a feeling or use case | **dense** | meaning match with zero shared words |
            | asks "which X works with Y" | **graph** | it is a set-membership question over edges |
            | mixes constraints | **hybrid + filters** | no single retriever covers all constraints |

            Then validate any hunch on the **Evaluation** page before changing production.
            One query is an anecdote; twelve labelled queries are evidence.
            """
        )
        ui.callout(
            "The transferable skill",
            "Retrieval choice is not a matter of taste. Look at the query: is the signal a "
            "token, a meaning, or a relation? Pick the structure that represents that signal.",
            tone="info",
        )
    with prediction_right:
        ui.note_card("compare")
