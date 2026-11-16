import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import rag_core as rc
import ui


def _compute(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    frame = rc.evaluate_all(engine, k=5)
    frame.attrs["seconds"] = time.perf_counter() - started

    latency = {}
    queries = list(engine.qrels)
    for name, function in (
        ("semantic", lambda q: rc.dense_rank_ids(engine, q, 10)),
        ("sparse", lambda q: rc.sparse_rank_ids(engine, q, 10)),
        ("graph", lambda q: rc.graph_rank_ids(engine, q, 10)),
        ("hybrid", lambda q: rc.hybrid_search(engine, q, k=10)["doc_id"].tolist()),
    ):
        function(queries[0])
        started = time.perf_counter()
        for query in queries:
            function(query)
        latency[name] = (time.perf_counter() - started) / len(queries) * 1000
    return frame, pd.Series(latency)


def render() -> None:
    engine = ui.get_engine()

    if "eval_frame" not in st.session_state:
        with st.spinner("Scoring 4 retrievers against 12 labelled queries…"):
            st.session_state["eval_frame"], st.session_state["eval_latency"] = _compute(engine)

    frame = st.session_state["eval_frame"]
    latency = st.session_state["eval_latency"]
    metrics = ["hit@5", "precision@5", "recall@5", "mrr", "ndcg@5"]
    overall = (frame.groupby("retriever")[metrics].mean()
               .reindex(["semantic", "sparse", "graph", "hybrid"]).round(3))

    ui.page_header(
        "Evaluation: prove it, do not vibe it",
        "Twelve queries with labels derived from the data, five metrics, sliced by intent. "
        "This is where opinions about retrievers become numbers.",
        tag="Evaluation",
    )

    hybrid = overall.loc["hybrid"]
    ui.kpi_row([
        ("hybrid nDCG@5", f"{hybrid['ndcg@5']:.3f}", "hybrid"),
        ("hybrid Hit@5", f"{hybrid['hit@5']:.3f}", "semantic"),
        ("hybrid MRR", f"{hybrid['mrr']:.3f}", "sparse"),
        ("queries labelled", f"{len(engine.qrels)}", "graph"),
    ])

    left, right = ui.demo_panel()
    with left:
        st.markdown("#### Aggregate metrics")
        styled = overall.reset_index().rename(columns={"retriever": "retriever"})
        st.dataframe(styled, hide_index=True, use_container_width=True)
        st.caption(
            "Hit@5 asks whether anything relevant appeared. MRR asks how high the first "
            "relevant hit was. nDCG discounts each relevant item by its position."
        )

        st.markdown("#### Latency (mean per query, CPU)")
        ui.latency_chips({name: float(value) for name, value in latency.items()})
        fig = charts.latency_figure(latency, dark=ui.is_dark())
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        ui.note_card("metrics")

    ui.divider("The slice that matters: intent")

    intent_left, intent_right = ui.demo_panel()
    with intent_left:
        by_intent = (frame.groupby(["retriever", "intent"])["ndcg@5"].mean()
                     .unstack("intent")
                     .reindex(["semantic", "sparse", "graph", "hybrid"]).round(3))
        fig = charts.heatmap_figure(by_intent, dark=ui.is_dark())
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(by_intent.reset_index(), hide_index=True, use_container_width=True)
        st.caption(
            "Graph scores 0.000 on exact-id and conceptual by design: the router never runs "
            "it there. When it does run, on relational queries, it is the only retriever that works."
        )
    with intent_right:
        ui.note_card("honesty")

    ui.divider("Per-query detail")

    per_query = (frame.pivot_table(index=["intent", "query"], columns="retriever", values="ndcg@5")
                 .reindex(columns=["semantic", "sparse", "graph", "hybrid"]).round(3).reset_index())
    st.dataframe(per_query, hide_index=True, use_container_width=True, height=380)

    wins = per_query.melt(id_vars=["intent", "query"], var_name="retriever", value_name="ndcg")
    best = wins.loc[wins.groupby("query")["ndcg"].idxmax()]
    summary = best["retriever"].value_counts().rename_axis("best retriever").reset_index(name="queries won")
    st.markdown("**Who wins each query?**")
    summary_left, summary_right = st.columns([1, 1.6], gap="large")
    with summary_left:
        st.dataframe(summary, hide_index=True, use_container_width=True)
    with summary_right:
        st.markdown(
            """
            Nobody wins everything. Sparse owns exact identifiers, dense owns the
            paraphrase gap, graph owns relations, and the hybrid wins the aggregate
            because it never collapses on a query type.
            """
        )

    with st.expander("How the labels were built (no hand-labelling)", expanded=False):
        rows = []
        for query, (intent, relevant) in engine.qrels.items():
            rows.append({"query": query, "intent": intent, "n_relevant": len(relevant),
                         "sample ids": ", ".join(sorted(relevant)[:6])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Labels are derived from catalogue attributes and graph edges, so every "
                   "label is auditable.")
