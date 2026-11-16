"""Matplotlib figures used by the lab pages."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import rag_core as rc

TEXT = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def _theme(dark: bool) -> dict:
    return {
        "text": "#e6edf7" if dark else TEXT,
        "muted": "#94a3b8" if dark else MUTED,
        "grid": "#24324a" if dark else GRID,
        "face": "none",
    }


def _base(dark: bool, width: float, height: float):
    plt.rcParams.update({
        "figure.figsize": (width, height),
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "font.size": 9.5,
        "text.color": _theme(dark)["text"],
        "axes.labelcolor": _theme(dark)["text"],
        "xtick.color": _theme(dark)["muted"],
        "ytick.color": _theme(dark)["muted"],
        "axes.edgecolor": _theme(dark)["grid"],
    })
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    return fig, ax


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    fig.patch.set_alpha(0.0)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.3, 2.5, 1.6, 0.8, "User\nquery", "#e2e8f0"),
        (2.35, 2.5, 1.9, 0.8, "Retriever\n(top-k)", "#dbeafe"),
        (4.7, 2.5, 1.9, 0.8, "Context\npacker", "#fef3c7"),
        (7.05, 2.5, 1.9, 0.8, "Generator\n(LLM)", "#ede9fe"),
        (9.4, 2.5, 1.8, 0.8, "Grounded\nanswer", "#dcfce7"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    linewidth=1.3, edgecolor="#334155", facecolor=color))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)

    for x0, x1 in [(1.93, 2.35), (4.28, 4.7), (6.63, 7.05), (8.98, 9.4)]:
        ax.add_patch(FancyArrowPatch((x0, 2.9), (x1, 2.9), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.4, color="#475569"))

    index_boxes = [
        (2.25, 0.55, 1.8, 0.9, "Sparse\nBM25", rc.PALETTE["sparse"]),
        (4.3, 0.55, 1.8, 0.9, "Dense\nvectors", rc.PALETTE["semantic"]),
        (6.35, 0.55, 1.8, 0.9, "Graph\nentities", rc.PALETTE["graph"]),
    ]
    for x, y, w, h, label, color in index_boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    linewidth=1.3, edgecolor=color, facecolor="none"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color=color)
    for x in (3.15, 5.2, 7.25):
        ax.add_patch(FancyArrowPatch((x, 1.5), (3.3, 2.45), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.2, color="#94a3b8", linestyle="--"))
    ax.text(5.7, 1.75, "hybrid fusion (RRF)", ha="center", fontsize=8, style="italic", color="#64748b")
    ax.text(0.3, 3.75, "Where accuracy is won or lost", fontsize=11, fontweight="bold", color=TEXT)
    ax.text(0.3, 3.42, "the generator never sees what the retriever did not fetch",
            fontsize=8.5, color="#64748b")
    plt.tight_layout()
    return fig


def cosine_figure(sentences: list[str]):
    vectors = rc.embed_texts(sentences)
    sim = vectors @ vectors.T
    fig, ax = _base(False, 6.4, 4.6)
    im = ax.imshow(sim, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = [f"S{i + 1}" for i in range(len(sentences))]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if sim[i, j] > 0.62 else TEXT)
    plt.colorbar(im, fraction=0.04)
    plt.tight_layout()
    return fig, sim


def pca_figure(engine: rc.Engine, dark: bool = False):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=rc.SEED)
    coords = pca.fit_transform(engine.chunk_vecs)
    frame = engine.chunks.assign(x=coords[:, 0], y=coords[:, 1])

    queries = [
        "noise cancelling headphones for flights",
        "budget 65W laptop charger",
        "4K monitor for video editing",
        "can I return opened earbuds",
    ]
    query_coords = pca.transform(rc.embed_texts(queries))

    colors = _theme(dark)
    fig, ax = _base(dark, 9.5, 5.2)
    cmap = plt.get_cmap("tab20")
    for index, (category, group) in enumerate(frame.groupby("category")):
        ax.scatter(group["x"], group["y"], s=12, alpha=0.65, color=cmap(index % 20), label=category)
    for (qx, qy), label in zip(query_coords, queries):
        ax.scatter(qx, qy, marker="*", s=260, color="#111827", edgecolor="white", linewidth=0.7, zorder=5)
        ax.annotate(label, (qx, qy), textcoords="offset points", xytext=(7, 5), fontsize=7.5,
                    fontweight="bold", color=colors["text"],
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=colors["grid"], alpha=0.9))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.85)
    plt.tight_layout()
    return fig


def subgraph_figure(engine: rc.Engine, product_id: str, radius: int = 2, dark: bool = False):
    graph = engine.graph
    ego = nx.ego_graph(graph.to_undirected(as_view=True), product_id, radius=radius)
    colors = {
        "product": rc.PALETTE["semantic"], "brand": rc.PALETTE["sparse"],
        "category": rc.PALETTE["graph"], "policy": rc.PALETTE["hybrid"], "supplier": "#a16207",
    }
    sizes = {"product": 340, "brand": 720, "category": 720, "policy": 720, "supplier": 720}

    pos = nx.spring_layout(ego, seed=rc.SEED, k=0.45, iterations=160)
    fig, ax = _base(dark, 9.5, 5.6)
    ax.grid(False)
    ax.axis("off")
    for kind, color in colors.items():
        nodes = [node for node, data in ego.nodes(data=True) if data.get("kind") == kind]
        if nodes:
            nx.draw_networkx_nodes(ego, pos, nodelist=nodes, ax=ax, node_color=color,
                                   node_size=sizes[kind], alpha=0.92,
                                   edgecolors="white", linewidths=1.0)
    nx.draw_networkx_edges(ego, pos, ax=ax, alpha=0.3, arrows=False)
    labels = {}
    for node, data in ego.nodes(data=True):
        label = data.get("label", str(node))
        labels[node] = label[:24] + "…" if len(label) > 27 else label
    text_color = _theme(dark)["text"]
    nx.draw_networkx_labels(ego, pos, labels=labels, font_size=6, font_color=text_color, ax=ax)
    for kind, color in colors.items():
        ax.scatter([], [], c=color, s=70, label=kind)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    return fig, ego.number_of_nodes(), ego.number_of_edges()


def heatmap_figure(matrix: pd.DataFrame, dark: bool = False, value_format: str = "{:.2f}"):
    fig, ax = _base(dark, 6.6, 3.4)
    data = matrix.to_numpy()
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, value_format.format(data[i, j]), ha="center", va="center",
                    fontsize=9, color="white" if data[i, j] > 0.55 else TEXT)
    plt.colorbar(im, fraction=0.04)
    ax.grid(False)
    plt.tight_layout()
    return fig


def bar_figure(series: pd.Series, dark: bool = False, color: str = "#2563eb",
               horizontal: bool = False, title: str = ""):
    fig, ax = _base(dark, 6.2, 3.0)
    if horizontal:
        ax.barh(series.index.astype(str), series.values, color=color, alpha=0.9)
        ax.invert_yaxis()
    else:
        ax.bar(series.index.astype(str), series.values, color=color, alpha=0.9)
        ax.tick_params(axis="x", rotation=20)
    if title:
        ax.set_title(title)
    ax.grid(axis="y" if horizontal else "x", alpha=0.2)
    plt.tight_layout()
    return fig


def latency_figure(series: pd.Series, dark: bool = False):
    colors = [rc.PALETTE.get(name, "#64748b") for name in series.index]
    fig, ax = _base(dark, 6.2, 2.8)
    ax.barh(series.index.astype(str), series.values, color=colors, alpha=0.92)
    for i, value in enumerate(series.values):
        ax.text(value + max(series.values) * 0.02, i, f"{value:.1f} ms", va="center", fontsize=8.5)
    ax.set_xlabel("mean latency per query (ms)")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig


def score_breakdown(terms: pd.DataFrame, dark: bool = False):
    fig, ax = _base(dark, 6.2, 2.6)
    terms = terms.head(8)
    ax.barh(terms["term"], terms["contribution"], color=rc.PALETTE["sparse"], alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("score contribution")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig
