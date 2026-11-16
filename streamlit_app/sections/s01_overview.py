import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import ui


def render() -> None:
    engine = ui.get_engine()
    stats = engine.stats()

    ui.page_header(
        "Comparative retrieval, end to end",
        "One catalogue, three retrieval families, one grounded answering layer. "
        "Every panel shows the experiment on the left and the reasoning on the right.",
        tag="Overview",
    )

    ui.kpi_row([
        ("products", f"{stats['products']}", "semantic"),
        ("chunks indexed", f"{stats['chunks']}", "sparse"),
        ("graph edges", f"{stats['graph_edges']:,}", "graph"),
        ("labelled queries", f"{len(engine.qrels)}", "hybrid"),
    ])

    left, right = ui.demo_panel()
    with left:
        fig = charts.pipeline_figure()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        ui.callout(
            "The one sentence to remember",
            "The generator never sees what the retriever did not fetch. "
            "Every hallucination you will see today is a retrieval failure wearing a costume.",
            tone="info",
        )
    with right:
        ui.note_card("pipeline")

    st.markdown("### The three families at a glance")
    comparison = pd.DataFrame([
        {"Family": "Sparse (BM25)", "Indexes": "Exact terms + IDF statistics",
         "Wins when": "SKUs, model numbers, error codes, 65W, IP68",
         "Blind spot": "Synonyms and paraphrase", "Latency": "~2.5 ms"},
        {"Family": "Semantic (dense)", "Indexes": "Learned 384-d meaning vectors",
         "Wins when": "Vague intent, paraphrase, cross-lingual",
         "Blind spot": "Numbers and identifiers blur", "Latency": "~16 ms"},
        {"Family": "Graph", "Indexes": "Entities and typed relations",
         "Wins when": "Compatibility, succession, policy paths, multi-hop",
         "Blind spot": "Coverage gaps; a missing edge is invisible", "Latency": "~0.1 ms"},
        {"Family": "Hybrid (RRF)", "Indexes": "Ranks from all of the above",
         "Wins when": "You do not know the query type in advance",
         "Blind spot": "Costs the sum of its parts", "Latency": "~30 ms"},
    ])
    ui.chips([
        ("blue = dense semantics", "lab-badge-semantic"),
        ("orange = lexical exactness", "lab-badge-sparse"),
        ("green = relations", "lab-badge-graph"),
        ("purple = fusion", "lab-badge-hybrid"),
    ])
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    st.markdown("### Suggested route through the lab")
    route_left, route_right = st.columns([1.65, 1], gap="large")
    with route_left:
        st.markdown(
            """
            1. **Dataset** — meet the corpus and see why chunks, not rows, are indexed.
            2. **Semantic** — embeddings and cosine, then watch identifiers break it.
            3. **Sparse** — BM25 from scratch, with a live term-contribution table.
            4. **Graph** — compatibility, succession and policy paths over edges.
            5. **Hybrid & routing** — the same query through four retrievers, fused.
            6. **Evaluation** — twelve labelled queries, five metrics, honest slices.
            7. **Grounded answers** — citations, claim verification, abstention.
            8. **Playground** — your own queries, your own weights.
            """
        )
        st.info(
            "Each page follows the same layout: the experiment on the left, the explanation "
            "rail on the right. Presenter mode in the sidebar enlarges both.",
            icon="🎤",
        )
    with route_right:
        ui.note_card("fusion")
