import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import rag_core as rc
import ui


def _product_id_by_model(engine, model: str) -> str:
    matches = engine.products.loc[engine.products["subcategory"] == model, "product_id"]
    return matches.iloc[0]


def _compatibility(engine) -> None:
    product_id = _product_id_by_model(engine, "XPS 15 9530")
    graph = engine.graph
    neighbours = rc.neighbours(graph, product_id, "COMPATIBLE_WITH", "both")
    frame = pd.DataFrame([
        {"product_id": node, "title": graph.nodes[node]["label"],
         "category": graph.nodes[node]["category"],
         "price_inr": graph.nodes[node]["price_inr"],
         "warranty_months": graph.nodes[node]["warranty_months"],
         "path": f"{rc.node_label(graph, product_id)} --COMPATIBLE_WITH--> {graph.nodes[node]['label']}"}
        for node in neighbours
    ]).sort_values("price_inr")
    st.markdown(f"**Seed:** {graph.nodes[product_id]['label']} · {len(frame)} compatible products")
    st.dataframe(
        frame[["product_id", "title", "category", "price_inr", "warranty_months"]],
        hide_index=True, use_container_width=True,
        column_config={"price_inr": st.column_config.NumberColumn("price", format="₹%d")},
    )
    with st.expander("Show provenance paths", expanded=False):
        for path in frame["path"].head(6):
            st.code(path, language="text")


def _succession(engine) -> None:
    graph = engine.graph
    start = _product_id_by_model(engine, "WH-1000XM4")
    chain, current = [start], start
    for _ in range(6):
        following = rc.neighbours(graph, current, "SUCCESSOR_OF", "out")
        if not following:
            break
        current = following[0]
        chain.append(current)

    rows = []
    for position, node in enumerate(chain):
        data = graph.nodes[node]
        rows.append({
            "step": position,
            "product_id": node,
            "title": data["label"],
            "price_inr": data["price_inr"],
            "warranty_months": data["warranty_months"],
            "role": "newest" if position == len(chain) - 1 else f"step {position}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"price_inr": st.column_config.NumberColumn("price", format="₹%d")})
    st.markdown("**Provenance**")
    for older, newer in zip(chain, chain[1:]):
        st.code(f"{rc.node_label(graph, older)} --SUCCESSOR_OF--> {rc.node_label(graph, newer)}",
                language="text")


def _aggregation(engine, min_warranty: int) -> None:
    graph = engine.graph
    rows = []
    for laptop in engine.products.loc[engine.products["category"] == "Laptops", "product_id"]:
        for accessory in rc.neighbours(graph, laptop, "COMPATIBLE_WITH", "both"):
            data = graph.nodes[accessory]
            if data["kind"] != "product" or data["category"] == "Laptops":
                continue
            if data["warranty_months"] >= min_warranty:
                rows.append({"brand": data["brand"], "accessory": data["label"],
                             "category": data["category"], "price_inr": data["price_inr"],
                             "warranty_months": data["warranty_months"],
                             "for_laptop": graph.nodes[laptop]["label"]})
    if not rows:
        st.info("No edges matched — a coverage gap, which is the graph's failure mode.")
        return
    frame = pd.DataFrame(rows)
    ranking = (frame.groupby("brand")
               .agg(accessories=("accessory", "nunique"),
                    avg_price=("price_inr", "mean"),
                    max_warranty=("warranty_months", "max"))
               .sort_values("accessories", ascending=False).reset_index())
    ranking["avg_price"] = ranking["avg_price"].round(0).astype(int)
    st.dataframe(ranking, hide_index=True, use_container_width=True,
                 column_config={"avg_price": st.column_config.NumberColumn("avg price", format="₹%d")})
    top_brand = ranking.iloc[0]["brand"]
    st.markdown(f"**Top brand:** {top_brand} — {int(ranking.iloc[0]['accessories'])} qualifying accessories")
    st.dataframe(
        frame[frame["brand"] == top_brand][["accessory", "category", "price_inr", "warranty_months", "for_laptop"]],
        hide_index=True, use_container_width=True,
        column_config={"price_inr": st.column_config.NumberColumn("price", format="₹%d")},
    )


def _policy(engine) -> None:
    graph = engine.graph
    category = "CAT:Earbuds"
    policies = rc.neighbours(graph, category, "GOVERNED_BY", "out")
    st.markdown(f"**Seed:** {rc.node_label(graph, category)}")
    for policy_id in policies:
        row = engine.policies.set_index("policy_id").loc[policy_id]
        st.markdown(f"**{policy_id} — {row['title']}**")
        st.write(row["text"])
        st.code(f"Earbuds --GOVERNED_BY--> {row['title']}", language="text")


def render() -> None:
    engine = ui.get_engine()
    graph = engine.graph
    kinds = pd.Series([data["kind"] for _, data in graph.nodes(data=True)]).value_counts()

    ui.page_header(
        "Graph retrieval: facts that are edges, not sentences",
        "Compatibility, succession and policy paths. Every answer arrives with the "
        "traversal that produced it, which is the point.",
        tag="Graph",
    )

    ui.kpi_row([
        ("nodes", f"{graph.number_of_nodes():,}", "graph"),
        ("edges", f"{graph.number_of_edges():,}", "graph"),
        ("products", f"{kinds.get('product', 0)}", "semantic"),
        ("policies", f"{kinds.get('policy', 0)}", "hybrid"),
    ])

    left, right = ui.demo_panel()
    with left:
        product_ids = engine.products.sort_values("title")["product_id"].tolist()
        chosen = st.selectbox(
            "Neighbourhood of a product",
            product_ids,
            index=product_ids.index(_product_id_by_model(engine, "XPS 15 9530")),
            format_func=lambda pid: engine.products.set_index("product_id").loc[pid, "title"],
            key="graph_product",
        )
        with st.spinner("Laying out the 2-hop subgraph…"):
            fig, node_count, edge_count = charts.subgraph_figure(engine, chosen, radius=2,
                                                                 dark=ui.is_dark())
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(f"2-hop neighbourhood: {node_count} nodes, {edge_count} edges. "
                   "Blue products, orange brands, green categories, purple policies.")

        st.markdown("### Entity linking")
        link_query = st.text_input(
            "Free text → seed nodes",
            value="I am looking at the Dell XPS 15 9530 — what works with it?",
            key="graph_link_query",
        )
        links = rc.link_entities(engine, link_query)
        if links:
            st.dataframe(
                pd.DataFrame(links, columns=["node", "kind", "match length"]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.warning("No entity matched. The graph cannot answer what it cannot link.")
    with right:
        ui.note_card("kg")

    ui.divider("Four traversal demos")

    demo = st.segmented_control(
        "Choose a traversal",
        ["Compatibility", "Succession", "Aggregation", "Policy path"],
        default="Compatibility",
        key="graph_demo",
    )

    demo_left, demo_right = ui.demo_panel()
    with demo_left:
        if demo == "Succession":
            _succession(engine)
        elif demo == "Aggregation":
            threshold = st.slider("Minimum warranty (months)", 6, 36, 12, step=6,
                                  key="graph_warranty")
            _aggregation(engine, threshold)
        elif demo == "Policy path":
            _policy(engine)
        else:
            _compatibility(engine)
    with demo_right:
        key = {"Compatibility": "compat", "Succession": "successor",
               "Aggregation": "aggregation", "Policy path": "policy"}.get(demo, "compat")
        ui.note_card(key)

    ui.divider("Graph coverage")
    coverage = pd.Series({
        "products with a compatibility edge": int(engine.products["product_id"].isin(
            {source for source, _, data in graph.edges(data=True) if data["relation"] == "COMPATIBLE_WITH"}
            | {target for _, target, data in graph.edges(data=True) if data["relation"] == "COMPATIBLE_WITH"}
        ).sum()),
        "products with a successor edge": int(engine.products["product_id"].isin(
            {source for source, _, data in graph.edges(data=True) if data["relation"] == "SUCCESSOR_OF"}
            | {target for _, target, data in graph.edges(data=True) if data["relation"] == "SUCCESSOR_OF"}
        ).sum()),
        "categories mapped to a policy": len({source for source, _, data in graph.edges(data=True)
                                                     if data["relation"] == "GOVERNED_BY"
                                                     and source.startswith("CAT:")}),
    })
    fig = charts.bar_figure(coverage, horizontal=True, color=rc.PALETTE["graph"],
                            title="How much of the catalogue is connected?")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "Coverage is the graph's honest weakness: a missing edge is a silent failure. "
        "Vectors can improvise; a graph cannot."
    )
