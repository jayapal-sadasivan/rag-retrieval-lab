import pandas as pd
import streamlit as st

import charts
import rag_core as rc
import ui


def _product_card(product: pd.Series, product_id: str) -> None:
    specs = product["specs"]
    st.markdown(
        f"""
        <div class="lab-card" style="border:1px solid var(--lab-border);border-radius:14px;
             padding:1rem 1.1rem;background:var(--lab-panel)">
          <div style="font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#64748b">
            {product_id} · {product['sku']}
          </div>
          <div style="font-size:1.12rem;font-weight:750;margin:.2rem 0 .35rem 0">{product['title']}</div>
          <div style="display:flex;gap:1.4rem;flex-wrap:wrap;font-size:.85rem;color:#334155">
            <span><b>₹{int(product['price_inr']):,}</b> <span style="color:#94a3b8">(MRP ₹{int(product['mrp_inr']):,})</span></span>
            <span>★ {product['rating']} · {int(product['rating_count']):,} ratings</span>
            <span>warranty {int(product['warranty_months'])} mo</span>
            <span>{product['category']} · {product['brand']}</span>
          </div>
          <p style="margin:.7rem 0 .2rem 0;font-size:.87rem;line-height:1.5;color:#1f2937">
            {product['description']}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Specifications", expanded=False):
        spec_frame = pd.DataFrame(
            [{"attribute": key.replace("_", " "), "value": ("Yes" if value else "No") if isinstance(value, bool) else value}
             for key, value in specs.items()]
        )
        st.dataframe(spec_frame, hide_index=True, use_container_width=True)


def render() -> None:
    engine = ui.get_engine()
    stats = engine.stats()
    products = engine.products

    ui.page_header(
        "A catalogue built to break retrievers",
        "Realistic product copy, near-duplicate model names, exact identifiers, and "
        "prose policies. The traps here are the ones that decide which retriever wins.",
        tag="Dataset",
    )

    ui.kpi_row([
        ("products", f"{stats['products']}", "semantic"),
        ("reviews", f"{stats['reviews']}", "sparse"),
        ("policy documents", f"{stats['policies']}", "graph"),
        ("typed relations", f"{stats['relations']:,}", "hybrid"),
    ])

    left, right = ui.demo_panel()
    with left:
        st.markdown("#### Filter the catalogue")
        filter_a, filter_b, filter_c = st.columns([1, 1, 1.2])
        categories = sorted(products["category"].unique())
        selected_categories = filter_a.multiselect("Category", categories, default=categories,
                                                   key="ds_categories")
        brands = sorted(products.loc[products["category"].isin(selected_categories), "brand"].unique())
        selected_brands = filter_b.multiselect("Brand", brands, default=[], key="ds_brands",
                                               placeholder="All brands")
        max_price = filter_c.slider("Max price (₹)", 500, 250000, 250000, step=500, key="ds_price")

        view = products[products["category"].isin(selected_categories) & (products["price_inr"] <= max_price)]
        if selected_brands:
            view = view[view["brand"].isin(selected_brands)]

        table = view[["product_id", "title", "brand", "category", "price_inr", "rating", "warranty_months"]]
        st.dataframe(
            table, hide_index=True, use_container_width=True, height=260,
            column_config={
                "product_id": st.column_config.TextColumn("id", width="small"),
                "title": st.column_config.TextColumn("title", width="large"),
                "price_inr": st.column_config.NumberColumn("price", format="₹%d"),
                "rating": st.column_config.NumberColumn("rating", format="%.1f"),
                "warranty_months": st.column_config.NumberColumn("warranty", width="small"),
            },
        )
        st.caption(f"{len(view)} products match. Select one to inspect its chunks.")

        chosen = st.selectbox(
            "Inspect a product",
            view["product_id"].tolist(),
            format_func=lambda pid: products.set_index("product_id").loc[pid, "title"],
            key="ds_product",
        )
        product = products.set_index("product_id").loc[chosen]
        _product_card(product, chosen)

        st.markdown("#### Chunk inspector — what the retriever actually sees")
        product_chunks = engine.chunks[engine.chunks["doc_id"].str.startswith(f"{chosen}:")]
        tabs = st.tabs([f"{row['chunk_type']} · {len(row['text'].split())} words"
                        for _, row in product_chunks.iterrows()])
        for tab, (_, row) in zip(tabs, product_chunks.iterrows()):
            with tab:
                st.code(row["text"], language="text")
                st.caption(f"`{row['doc_id']}` · metadata: "
                           f"{ {k: v for k, v in list(row['metadata'].items())[:4]} }")

        with st.expander("Read the reviews behind the review chunk", expanded=False):
            reviews = engine.reviews[engine.reviews["product_id"] == chosen]
            st.dataframe(reviews[["rating", "verified_purchase", "review_date", "review_text"]],
                         hide_index=True, use_container_width=True)
    with right:
        ui.note_card("chunking")

    ui.divider("Why this dataset is hard")

    traps_left, traps_right = st.columns([1.65, 1], gap="large")
    with traps_left:
        traps = pd.DataFrame([
            {"Trap": "Near-duplicate model names", "Example": "WH-1000XM4 / XM5 / XM6",
             "Hurts": "Dense (vectors almost identical)"},
            {"Trap": "Exact identifiers", "Example": "SKU SONY-WH1000XM5-BLK",
             "Hurts": "Dense (sub-word tokenisation)"},
            {"Trap": "Numeric constraints", "Example": "65W, IP68, RTX 4060, under ₹70,000",
             "Hurts": "Dense (numbers weakly represented)"},
            {"Trap": "Synonyms and intent", "Example": "block out noise vs noise cancellation",
             "Hurts": "Sparse (zero lexical overlap)"},
            {"Trap": "Relational facts", "Example": "which charger fits this laptop",
             "Hurts": "Both (the fact lives in an edge)"},
            {"Trap": "Abstract prose", "Example": "refund, EMI, installation policies",
             "Hurts": "Sparse (formal, repetitive vocabulary)"},
        ])
        st.dataframe(traps, hide_index=True, use_container_width=True)
        ui.callout(
            "How the corpus is stored on disk",
            "products.csv · reviews.csv · policies.csv · relations.csv · products_specs.json · "
            "qrels.csv, plus cached chunk embeddings keyed by a SHA-1 of the text. "
            "The app recomputes that hash at startup and rebuilds only if the corpus changed.",
            tone="info",
        )
    with traps_right:
        ui.note_card("catalogue")

    ui.divider("Corpus shape")
    shape_a, shape_b = st.columns(2, gap="large")
    with shape_a:
        category_counts = products["category"].value_counts()
        fig = charts.bar_figure(category_counts, horizontal=True, color=rc.PALETTE["hybrid"],
                                title="Products per category")
        st.pyplot(fig, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close(fig)
    with shape_b:
        price_by_category = products.groupby("category")["price_inr"].median().sort_values()
        fig = charts.bar_figure(price_by_category, horizontal=True, color=rc.PALETTE["semantic"],
                                title="Median price by category (₹)")
        st.pyplot(fig, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close(fig)
