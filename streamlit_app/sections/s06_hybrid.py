import time

import pandas as pd
import streamlit as st

import rag_core as rc
import ui

PRESETS = [
    "Sony WH-1000XM4",
    "RTX 4060 laptop",
    "headphones that silence aeroplane engine noise",
    "accessories compatible with the Dell XPS 15 9530",
    "return policy for earbuds",
]


def render() -> None:
    engine = ui.get_engine()

    ui.page_header(
        "Hybrid retrieval: fuse ranks, then route",
        "Run the same query through every retriever, fuse with Reciprocal Rank Fusion, "
        "and let intent decide which lists are even consulted.",
        tag="Hybrid",
    )

    left, right = ui.demo_panel()
    with left:
        query = ui.query_input("hyb", PRESETS, PRESETS[3])
        plan = rc.route_plan(query)
        badges = "".join(
            f'<span class="lab-badge lab-badge-{name}">{name} ×{weight:.1f}</span>'
            for name, weight in plan["weights"].items()
        )
        st.markdown(f'<div style="margin-bottom:.4rem">{badges}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lab-callout lab-callout-info"><div class="lab-callout-title">'
            "Routing decision</div><div class=\"lab-callout-body\">"
            + " · ".join(plan["reasons"])
            + "</div></div>",
            unsafe_allow_html=True,
        )

        weights = None
        with st.expander("Override the RRF weights", expanded=False):
            override = st.toggle("Use manual weights", value=False, key="hyb_override")
            weight_cols = st.columns(4)
            manual = {
                "dense": weight_cols[0].slider("dense", 0.0, 2.0, float(plan["weights"]["dense"]), 0.1,
                                               key="w_dense"),
                "sparse": weight_cols[1].slider("sparse", 0.0, 2.0, float(plan["weights"]["sparse"]), 0.1,
                                                key="w_sparse"),
                "graph": weight_cols[2].slider("graph", 0.0, 2.0, float(plan["weights"]["graph"]), 0.1,
                                               key="w_graph"),
                "policy": weight_cols[3].slider("policy", 0.0, 2.0, float(plan["weights"]["policy"]), 0.1,
                                                key="w_policy"),
            }
            if override:
                weights = manual

        started = time.perf_counter()
        fused = rc.hybrid_search(engine, query, k=8, weights=weights)
        elapsed = (time.perf_counter() - started) * 1000
        ui.latency_chips({"hybrid": elapsed})

        if len(fused):
            display = fused.copy()
            rank_columns = [column for column in display.columns if column.startswith("rank_")]
            display["found by"] = display.apply(
                lambda row: " · ".join(
                    f"{column.replace('rank_', '')}@{int(row[column])}"
                    for column in rank_columns if pd.notna(row[column])
                ), axis=1)
            ui.results_table(display, score_column="rrf_score",
                             caption="RRF score = Σ weight / (60 + rank). The right column shows which list found each hit.")
            st.dataframe(
                display[["doc_id", "found by", "rank_dense", "rank_sparse"]
                        + (["rank_graph"] if "rank_graph" in display else [])
                        + (["rank_policy"] if "rank_policy" in display else [])],
                hide_index=True, use_container_width=True,
            )
        else:
            st.warning("Nothing retrieved.")
    with right:
        ui.note_card("fusion")

    ui.divider("Same query, four retrievers")

    comparison_rows = []
    per_method = {}
    for method in ("semantic", "sparse", "graph", "hybrid"):
        started = time.perf_counter()
        frame = rc.search(engine, method, query, k=3)
        ms = (time.perf_counter() - started) * 1000
        per_method[method] = frame
        titles = " | ".join(frame["title"].astype(str).str.slice(0, 38).tolist()) if len(frame) else "—"
        comparison_rows.append({
            "retriever": rc.RETRIEVER_LABELS[method],
            "top 3": titles,
            "latency_ms": round(ms, 1),
        })

    compare_left, compare_right = ui.demo_panel()
    with compare_left:
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True,
                     column_config={"latency_ms": st.column_config.NumberColumn("ms", format="%.1f")})
        ui.latency_chips({method: row["latency_ms"] for method, row in
                          zip(["semantic", "sparse", "graph", "hybrid"], comparison_rows)})

        selected_method = st.radio(
            "Inspect one retriever's full result",
            ["semantic", "sparse", "graph", "hybrid"],
            horizontal=True, key="hyb_method",
        )
        frame = per_method[selected_method]
        score_column = "score" if "score" in frame.columns else "rrf_score"
        score_max = None if selected_method in ("semantic", "hybrid") else (
            float(frame[score_column].max()) if len(frame) else 1.0)
        ui.results_table(frame, score_column=score_column, score_max=score_max)
        if len(frame) == 0:
            ui.callout(
                "Empty result is informative",
                "The graph is only consulted for relational intents. Ask a compatibility or "
                "succession question and this column fills up.",
                tone="warn",
            )
    with compare_right:
        ui.note_card("routing")

    ui.divider("Read the disagreement")

    disagree_left, disagree_right = ui.demo_panel()
    with disagree_left:
        probe = "Sony WH-1000XM4 headphones"
        rows = []
        for method in ("semantic", "sparse", "hybrid"):
            frame = rc.search(engine, method, probe, k=1)
            rows.append({
                "retriever": rc.RETRIEVER_LABELS[method],
                "top-1": frame.iloc[0]["title"] if len(frame) else "—",
                "id": frame.iloc[0]["doc_id"] if len(frame) else "—",
            })
        st.markdown(f"**Query:** `{probe}`")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "When two independent retrievers agree on the same document, confidence rises. "
            "When they disagree, the rank columns tell you which one to investigate."
        )
    with disagree_right:
        ui.note_card("compare")
