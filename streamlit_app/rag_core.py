"""Shared retrieval engine for the RAG Retrieval Lab.

Mirrors the logic of RAG_Retrieval_Comparative_Tutorial.ipynb, but returns
values instead of printing, so both the notebook and the Streamlit app stay
in agreement. The notebook remains the teaching artefact; this module is the
application layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"
EMB_PATH = DATA_DIR / "chunk_embeddings.npy"
HASH_PATH = DATA_DIR / "chunk_embeddings.sha1"

SEED = 20261114
ABSTAIN_THRESHOLD = 0.40
CHUNK_PRIORITY = {"desc": 0, "policy": 0, "spec": 1, "rev": 2}

PALETTE = {
    "semantic": "#2563eb",
    "sparse": "#ea580c",
    "graph": "#059669",
    "hybrid": "#7c3aed",
    "muted": "#64748b",
}

RETRIEVER_LABELS = {
    "semantic": "Semantic (dense)",
    "sparse": "Sparse (BM25)",
    "graph": "Graph (multi-hop)",
    "hybrid": "Hybrid (RRF)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class Corpus:
    products: pd.DataFrame
    reviews: pd.DataFrame
    policies: pd.DataFrame
    relations: pd.DataFrame
    chunks: pd.DataFrame


def _render_specs(specs: dict) -> str:
    parts = []
    for key, value in specs.items():
        label = key.replace("_", " ").title()
        if isinstance(value, bool):
            value = "Yes" if value else "No"
        parts.append(f"{label}: {value}")
    return " | ".join(parts)


def load_frames(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(data_dir / "products.csv")
    specs = json.loads((data_dir / "products_specs.json").read_text(encoding="utf-8"))
    products["specs"] = products["product_id"].map(specs)
    reviews = pd.read_csv(data_dir / "reviews.csv")
    policies = pd.read_csv(data_dir / "policies.csv")
    relations = pd.read_csv(data_dir / "relations.csv")
    return products, reviews, policies, relations


def build_chunks(products: pd.DataFrame, reviews: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    reviews_by_product = reviews.groupby("product_id")["review_text"].apply(list).to_dict()
    rows = []

    for _, p in products.iterrows():
        meta = {
            "product_id": p["product_id"], "sku": p["sku"], "brand": p["brand"],
            "category": p["category"], "price_inr": int(p["price_inr"]),
            "rating": float(p["rating"]), "warranty_months": int(p["warranty_months"]),
            "stock_units": int(p["stock_units"]), "launch_year": int(p["launch_year"]),
        }
        rows.append({
            "doc_id": f"{p['product_id']}:desc", "chunk_type": "desc",
            "text": (
                f"{p['title']}. Brand: {p['brand']}. Category: {p['category']}. "
                f"{p['description']} Price: Rs {p['price_inr']}. "
                f"Warranty: {p['warranty_months']} months."
            ),
            "metadata": meta,
        })
        rows.append({
            "doc_id": f"{p['product_id']}:spec", "chunk_type": "spec",
            "text": (
                f"{p['title']} ({p['sku']}). Specifications -> {_render_specs(p['specs'])}. "
                f"Price: Rs {p['price_inr']}. Warranty: {p['warranty_months']} months. "
                f"Rating: {p['rating']} out of 5 from {p['rating_count']} buyers."
            ),
            "metadata": meta,
        })
        product_reviews = reviews_by_product.get(p["product_id"], [])[:3]
        if product_reviews:
            rows.append({
                "doc_id": f"{p['product_id']}:rev", "chunk_type": "rev",
                "text": f"Customer reviews for {p['title']}: " + " ".join(product_reviews),
                "metadata": meta,
            })

    for _, policy in policies.iterrows():
        rows.append({
            "doc_id": f"{policy['policy_id']}:policy", "chunk_type": "policy",
            "text": f"{policy['title']} [{policy['topic']}]. {policy['text']}",
            "metadata": {
                "product_id": None, "policy_id": policy["policy_id"],
                "category": "Policy", "brand": "Zephyr", "price_inr": 0,
                "rating": 0.0, "warranty_months": 0, "stock_units": 0,
                "launch_year": 2026, "sku": policy["policy_id"],
            },
        })

    chunks = pd.DataFrame(rows)
    for field in ("category", "brand", "price_inr", "warranty_months"):
        chunks[field] = chunks["metadata"].apply(lambda meta: meta.get(field))
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
_MODEL = None


def get_embedder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _MODEL


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    model = get_embedder()
    return np.asarray(
        model.encode(list(texts), batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def corpus_hash(chunks: pd.DataFrame) -> str:
    return hashlib.sha1("\n".join(chunks["text"].tolist()).encode("utf-8")).hexdigest()


def load_or_build_embeddings(chunks: pd.DataFrame) -> tuple[np.ndarray, dict]:
    digest = corpus_hash(chunks)
    info = {"hash": digest, "source": "cache"}
    if EMB_PATH.exists() and HASH_PATH.exists() and HASH_PATH.read_text().strip() == digest:
        return np.load(EMB_PATH), info

    vectors = embed_texts(chunks["text"].tolist())
    np.save(EMB_PATH, vectors)
    HASH_PATH.write_text(digest)
    info["source"] = "rebuilt"
    return vectors, info


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------
def _metadata_array(chunks: pd.DataFrame, field: str) -> np.ndarray:
    return chunks["metadata"].apply(lambda meta: meta.get(field)).to_numpy()


def semantic_search(state: "Engine", query: str, k: int = 8, chunk_type: str | None = None,
                    category: str | None = None, min_price: float | None = None,
                    max_price: float | None = None, min_warranty: float | None = None) -> pd.DataFrame:
    query_vec = embed_texts([query])[0]
    scores = state.chunk_vecs @ query_vec

    mask = np.ones(len(state.chunks), dtype=bool)
    if chunk_type:
        mask &= (state.chunks["chunk_type"] == chunk_type).to_numpy()
    if category:
        mask &= (state.chunks["category"] == category).to_numpy()
    if max_price is not None:
        mask &= (_metadata_array(state.chunks, "price_inr") <= max_price)
    if min_price is not None:
        mask &= (_metadata_array(state.chunks, "price_inr") >= min_price)
    if min_warranty is not None:
        mask &= (_metadata_array(state.chunks, "warranty_months") >= min_warranty)

    scores = np.where(mask, scores, -np.inf)
    top = np.argsort(-scores)[:k]

    out = state.chunks.iloc[top][["doc_id", "chunk_type", "text", "category"]].copy()
    out.insert(1, "score", np.round(scores[top], 4))
    out["product_id"] = out["doc_id"].str.split(":").str[0]
    titles = state.products.set_index("product_id")["title"]
    out["title"] = out["product_id"].map(titles).fillna("(policy document)")
    prices = state.products.set_index("product_id")["price_inr"]
    out["price_inr"] = out["product_id"].map(prices).fillna(0).astype(int)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sparse retrieval
# ---------------------------------------------------------------------------
STOPWORDS = set(
    """a an and are as at be been but by for from had has have he her his i if in
into is it its me my not of on or our out she that the their them then there these they this
those to us was we were what when where which who will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    raw_tokens = re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", text)
    tokens = []
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        tokens.append(token)
        if any(ch.isdigit() for ch in token) and re.search(r"[-./]", token):
            tokens.append(re.sub(r"[-./]", "", token))
    return [token for token in tokens if len(token) > 1]


def light_stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ss"):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def preprocess(text: str) -> list[str]:
    return [light_stem(token) for token in tokenize(text)]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [preprocess(text) for text in texts]
        self.doc_len = np.array([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean())
        self.N = len(texts)

        self.tf = []
        df_counter: Counter = Counter()
        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            self.tf.append(counts)
            df_counter.update(counts.keys())
        self.df = df_counter
        self.idf = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in df_counter.items()
        }
        self.vocab = set(df_counter)

    def term_scores(self, query: str) -> dict:
        query_terms = preprocess(query)
        scores = np.zeros(self.N, dtype=np.float32)
        contributions: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.N, dtype=np.float32))
        for term in query_terms:
            if term not in self.vocab:
                continue
            idf = self.idf[term]
            for i in range(self.N):
                freq = self.tf[i].get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                value = idf * (freq * (self.k1 + 1)) / denom
                scores[i] += value
                contributions[term][i] += value
        return {"scores": scores, "terms": dict(contributions), "query_terms": query_terms}

    def explain(self, query: str, doc_index: int) -> pd.DataFrame:
        result = self.term_scores(query)
        rows = [
            {"term": term, "idf": round(self.idf[term], 3),
             "tf_in_doc": self.tf[doc_index].get(term, 0),
             "contribution": round(float(values[doc_index]), 3)}
            for term, values in result["terms"].items()
        ]
        return pd.DataFrame(rows).sort_values("contribution", ascending=False)


def bm25_search(state: "Engine", query: str, k: int = 8, chunk_type: str | None = None) -> pd.DataFrame:
    index = state.bm25
    scores = index.term_scores(query)["scores"]
    order = np.argsort(-scores)

    rows = []
    for idx in order:
        score = float(scores[idx])
        if score <= 0:
            break
        row = state.chunks.iloc[int(idx)]
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
    return pd.DataFrame(rows, columns=["doc_id", "product_id", "chunk_type", "score",
                                       "title", "text", "category", "price_inr"])


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph(products: pd.DataFrame, policies: pd.DataFrame, relations: pd.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for _, p in products.iterrows():
        graph.add_node(p["product_id"], kind="product", label=p["title"], brand=p["brand"],
                       category=p["category"], price_inr=int(p["price_inr"]),
                       warranty_months=int(p["warranty_months"]), sku=p["sku"])
    for brand in sorted(products["brand"].unique()):
        graph.add_node(f"BRAND:{brand}", kind="brand", label=brand)
    for category in sorted(products["category"].unique()):
        graph.add_node(f"CAT:{category}", kind="category", label=category)
    for _, policy in policies.iterrows():
        graph.add_node(policy["policy_id"], kind="policy", label=policy["title"])
    for supplier in sorted(relations.loc[relations["relation"] == "SUPPLIED_BY", "target"].unique()):
        graph.add_node(supplier, kind="supplier", label=supplier.replace("SUP:", ""))

    for _, rel in relations.iterrows():
        if rel["source"] in graph and rel["target"] in graph:
            graph.add_edge(rel["source"], rel["target"], key=rel["relation"],
                           relation=rel["relation"], weight=float(rel["weight"]))
    return graph


def neighbours(graph: nx.MultiDiGraph, node: str, relation: str | None = None,
               direction: str = "out") -> list[str]:
    found = []
    if direction in ("out", "both"):
        for _, target, data in graph.out_edges(node, data=True):
            if relation is None or data["relation"] == relation:
                found.append(target)
    if direction in ("in", "both"):
        for source, _, data in graph.in_edges(node, data=True):
            if relation is None or data["relation"] == relation:
                found.append(source)
    return sorted(set(found))


def node_label(graph: nx.MultiDiGraph, node: str) -> str:
    return graph.nodes[node].get("label", node)


def describe_path(graph: nx.MultiDiGraph, path: list[str]) -> str:
    parts = []
    for i, node in enumerate(path):
        parts.append(node_label(graph, node))
        if i < len(path) - 1:
            edge = graph.get_edge_data(path[i], path[i + 1]) or graph.get_edge_data(path[i + 1], path[i])
            relation = next(iter(edge.values()))["relation"] if edge else "RELATED_TO"
            parts.append(f"--{relation}-->")
    return " ".join(parts)


def _squeeze(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def link_entities(state: "Engine", query: str) -> list[tuple[str, str, int]]:
    squeezed = _squeeze(query)
    hits: dict[str, int] = {}
    for product_id, model_squeezed, sku_squeezed in state.product_lookup:
        for candidate in (model_squeezed, sku_squeezed):
            if len(candidate) >= 5 and candidate in squeezed:
                hits[product_id] = max(hits.get(product_id, 0), len(candidate))
    for node, value in state.brand_lookup:
        if len(value) >= 3 and value in squeezed:
            hits[node] = max(hits.get(node, 0), len(value))
    for node, value in state.category_lookup:
        if len(value) >= 4 and value in squeezed:
            hits[node] = max(hits.get(node, 0), len(value))
    ordered = sorted(hits.items(), key=lambda item: -item[1])
    return [(node, state.graph.nodes[node]["kind"], score) for node, score in ordered]


# ---------------------------------------------------------------------------
# Routing and hybrid retrieval
# ---------------------------------------------------------------------------
RELATIONAL_PATTERNS = {
    "successor": ["replac", "successor", "newer", "newest", "next model", "later model"],
    "compatibility": ["compatib", "work with", "works with", "accessor", "what works",
                      "fit for", "charger for", "which charger"],
    "policy": ["policy", "policies", "applies", "governed", "terms and conditions"],
}
POLICY_PATTERNS = ["return", "refund", "warranty", "policy", "emi", "shipping", "delivery",
                   "installation", "cancel", "exchange", "gst", "invoice", "cod",
                   "loyalty", "gift card", "privacy", "international", "bulk", "claim"]


def detect_graph_intent(query: str) -> str | None:
    lowered = query.lower()
    for intent, patterns in RELATIONAL_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return intent
    return None


def is_policy_query(query: str) -> bool:
    lowered = query.lower()
    return any(pattern in lowered for pattern in POLICY_PATTERNS)


def dense_rank_ids(state: "Engine", query: str, k: int = 50, product_only: bool | None = None) -> list[str]:
    if product_only is None:
        product_only = not is_policy_query(query)
    frame = semantic_search(state, query, k=k * 3 if product_only else k)
    if product_only:
        frame = frame[frame["doc_id"].str.split(":").str[1] != "policy"]
    return frame["doc_id"].tolist()[:k]


def sparse_rank_ids(state: "Engine", query: str, k: int = 50, product_only: bool | None = None) -> list[str]:
    if product_only is None:
        product_only = not is_policy_query(query)
    frame = bm25_search(state, query, k=k * 3 if product_only else k)
    if product_only:
        frame = frame[frame["chunk_type"] != "policy"]
    return frame["doc_id"].tolist()[:k]


def graph_rank_ids(state: "Engine", query: str, k: int = 25) -> list[str]:
    intent = detect_graph_intent(query)
    if intent is None:
        return []

    graph = state.graph
    seeds = link_entities(state, query)
    product_seeds = [node for node, kind, _ in seeds if kind == "product"][:2]
    category_seeds = [node for node, kind, _ in seeds if kind == "category"][:2]

    doc_ids: list[str] = []
    if intent == "successor":
        for product_id in product_seeds:
            frontier, seen = [product_id], {product_id}
            while frontier:
                node = frontier.pop(0)
                for neighbour in neighbours(graph, node, "SUCCESSOR_OF", "out"):
                    if neighbour not in seen and graph.nodes[neighbour]["kind"] == "product":
                        seen.add(neighbour)
                        doc_ids.append(f"{neighbour}:desc")
                        frontier.append(neighbour)
    elif intent == "compatibility":
        for product_id in product_seeds:
            for neighbour in neighbours(graph, product_id, "COMPATIBLE_WITH", "both"):
                if (graph.nodes[neighbour]["kind"] == "product"
                        and graph.nodes[neighbour]["category"] != graph.nodes[product_id]["category"]):
                    doc_ids.append(f"{neighbour}:desc")
    elif intent == "policy":
        for category in category_seeds:
            for policy in neighbours(graph, category, "GOVERNED_BY", "out"):
                doc_ids.append(f"{policy}:policy")

    ordered, seen_docs = [], set()
    for doc_id in doc_ids:
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            ordered.append(doc_id)
    return ordered[:k]


def policy_rank_ids(state: "Engine", query: str, k: int = 15) -> list[str]:
    if not is_policy_query(query):
        return []
    frame = bm25_search(state, query, k=50, chunk_type="policy")
    return frame["doc_id"].tolist()[:k]


def rrf_fuse(rankings: dict[str, list[str]], k: int = 60,
             weights: dict[str, float] | None = None, top_k: int = 10) -> pd.DataFrame:
    weights = weights or {name: 1.0 for name in rankings}
    fused: dict[str, float] = defaultdict(float)
    detail: dict[str, dict[str, int]] = defaultdict(dict)

    for name, ids in rankings.items():
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] += weights.get(name, 1.0) / (k + rank)
            detail[doc_id][name] = rank

    rows = []
    for doc_id, score in sorted(fused.items(), key=lambda item: -item[1])[:top_k]:
        rows.append({
            "doc_id": doc_id,
            "rrf_score": round(score, 5),
            **{f"rank_{name}": detail[doc_id].get(name) for name in rankings},
        })
    return pd.DataFrame(rows)


def intent_weights(query: str) -> dict[str, float]:
    weights = {"dense": 1.0, "sparse": 1.0, "graph": 0.7, "policy": 1.0}
    if any(re.search(r"[a-z]*\d", token) for token in preprocess(query)):
        weights["sparse"] = 1.7
        weights["dense"] = 0.8
    graph_intent = detect_graph_intent(query)
    if graph_intent in ("compatibility", "successor"):
        weights["graph"] = 1.8
        weights["dense"] = 0.8
        weights["sparse"] = 0.8
    if graph_intent == "policy" or is_policy_query(query):
        weights["policy"] = 1.8
        weights["graph"] = 1.2
    return weights


def route_plan(query: str) -> dict:
    graph_intent = detect_graph_intent(query)
    policy_intent = is_policy_query(query)
    weights = intent_weights(query)
    specs = [token for token in preprocess(query) if re.search(r"[a-z]*\d", token)]
    reasons = []
    if specs:
        reasons.append(f"spec token(s) {specs[:3]} present")
    if graph_intent:
        reasons.append(f"relational intent '{graph_intent}' -> graph enabled")
    if policy_intent:
        reasons.append("policy vocabulary -> policy ranker enabled")
    if not reasons:
        reasons.append("no routing signal -> balanced RRF")
    return {
        "graph_intent": graph_intent,
        "policy_intent": policy_intent,
        "spec_tokens": specs,
        "weights": weights,
        "reasons": reasons,
    }


def hybrid_search(state: "Engine", query: str, k: int = 8,
                  weights: dict[str, float] | None = None,
                  methods: Iterable[str] = ("dense", "sparse", "graph", "policy")) -> pd.DataFrame:
    methods = set(methods)
    rankings: dict[str, list[str]] = {}
    if "dense" in methods:
        rankings["dense"] = dense_rank_ids(state, query, 50)
    if "sparse" in methods:
        rankings["sparse"] = sparse_rank_ids(state, query, 50)
    if "graph" in methods:
        graph_ids = graph_rank_ids(state, query, 25)
        if graph_ids:
            rankings["graph"] = graph_ids
    if "policy" in methods:
        policy_ids = policy_rank_ids(state, query, 15)
        if policy_ids:
            rankings["policy"] = policy_ids

    applied = intent_weights(query)
    if weights:
        applied.update(weights)

    fused = rrf_fuse(rankings, weights=applied, top_k=k)
    return enrich(state, fused, source_ranks=True)


# ---------------------------------------------------------------------------
# Result enrichment
# ---------------------------------------------------------------------------
def enrich(state: "Engine", frame: pd.DataFrame, source_ranks: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["product_id"] = out["doc_id"].str.split(":").str[0]
    if "text" not in out.columns:
        text_map = state.chunks.set_index("doc_id")["text"]
        out["text"] = out["product_id"].map(lambda _: None)
        out["text"] = out["doc_id"].map(text_map).fillna("")
    if "title" not in out.columns:
        out["title"] = None
    product_rows = state.products.set_index("product_id")
    policy_rows = state.policies.set_index("policy_id")

    def resolve_title(row):
        pid = row["product_id"]
        if pid in product_rows.index:
            return product_rows.loc[pid, "title"]
        if pid in policy_rows.index:
            return policy_rows.loc[pid, "title"]
        return row.get("title") or pid

    def resolve_category(row):
        pid = row["product_id"]
        if pid in product_rows.index:
            return product_rows.loc[pid, "category"]
        if pid in policy_rows.index:
            return "Policy"
        return row.get("category", "-")

    def resolve_price(row):
        pid = row["product_id"]
        if pid in product_rows.index:
            return int(product_rows.loc[pid, "price_inr"])
        return 0

    out["title"] = out.apply(resolve_title, axis=1)
    out["category"] = out.apply(resolve_category, axis=1)
    out["price_inr"] = out.apply(resolve_price, axis=1)
    out["chunk"] = out["doc_id"].str.split(":").str[1].fillna("-")
    return out


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
FILLER_PREFIX = re.compile(r"^(brand|category)\s*:", re.IGNORECASE)
FILLER_SENTENCES = [
    "backed by a manufacturer warranty and easy replacement",
    "available in multiple finishes",
    "free delivery in 2-4 days",
    "a bestseller in its price band",
    "rated highly by verified buyers",
]
ATTRIBUTE_HINTS = {
    "price": ["inr", "cost", "rupee"],
    "warranty": ["warranty", "guarantee", "months"],
    "battery": ["battery", "mah", "hours", "charging"],
    "return": ["return", "refund", "policy", "claim", "exchange"],
    "installation": ["install"],
    "shipping": ["shipping", "delivery", "deliver"],
    "emi": ["emi", "no-cost"],
    "compatibility": ["compatible", "works with", "accessory"],
    "weight": ["weight", " kg", " gram"],
}
PRICE_PATTERN = re.compile(r"\brs\.?\s*\d|\binr\s*\d|\u20b9")


def _matches_attribute(sentence_lower: str, attribute: str) -> bool:
    if attribute == "price":
        return bool(PRICE_PATTERN.search(sentence_lower)) or "price" in sentence_lower
    return any(hint in sentence_lower for hint in ATTRIBUTE_HINTS[attribute])


def query_attributes(query: str) -> list[str]:
    lowered = query.lower()
    found = []
    for name in ATTRIBUTE_HINTS:
        if name == "price":
            if PRICE_PATTERN.search(lowered) or any(
                    word in lowered for word in ("price", "cost", "rupee", "how much")):
                found.append(name)
        elif any(hint in lowered for hint in ATTRIBUTE_HINTS[name]):
            found.append(name)
    return found


def attributes_present(sentence: str) -> set[str]:
    lowered = sentence.lower()
    return {name for name in ATTRIBUTE_HINTS if _matches_attribute(lowered, name)}


def split_sentences(text: str, min_len: int = 15, max_len: int = 320) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT.split(text)]
    out = []
    for part in parts:
        if len(part) < min_len or len(part) > max_len:
            continue
        if FILLER_PREFIX.match(part):
            continue
        if re.search(r"\[[A-Za-z ]+\]\.$", part):
            continue
        lowered = part.lower()
        if any(filler in lowered for filler in FILLER_SENTENCES):
            continue
        carries_attribute = any(_matches_attribute(lowered, name) for name in ATTRIBUTE_HINTS)
        if len(part.split()) < 4 and not carries_attribute:
            continue
        out.append(part)
    return out


def score_sentences(query: str, context: list[dict]) -> pd.DataFrame:
    rows = []
    for doc in context:
        for position, sentence in enumerate(split_sentences(doc["text"])):
            rows.append({"doc_id": doc["doc_id"], "pos": position, "sentence": sentence})
    if not rows:
        return pd.DataFrame(columns=["doc_id", "pos", "sentence", "semantic",
                                     "lexical", "attribute", "score"])

    frame = pd.DataFrame(rows)
    sentence_vectors = embed_texts(frame["sentence"].tolist())
    query_vector = embed_texts([query])[0]
    frame["semantic"] = (sentence_vectors @ query_vector).round(4)

    query_terms = set(preprocess(query))
    frame["lexical"] = frame["sentence"].apply(
        lambda sentence: len(query_terms & set(preprocess(sentence))) / max(1, len(query_terms))
    )

    attrs = query_attributes(query)
    if attrs:
        denom = min(len(attrs), 2)

        def attribute_hit(sentence: str) -> float:
            lowered = sentence.lower()
            hits = sum(1 for name in attrs if _matches_attribute(lowered, name))
            return min(1.0, hits / denom)

        frame["attribute"] = frame["sentence"].apply(attribute_hit)
        frame["score"] = (0.45 * frame["semantic"] + 0.15 * frame["lexical"]
                          + 0.40 * frame["attribute"]).round(4)
    else:
        frame["attribute"] = 0.0
        frame["score"] = (0.75 * frame["semantic"] + 0.25 * frame["lexical"]).round(4)

    frame["chunk_rank"] = (frame["doc_id"].str.split(":").str[1]
                           .map(CHUNK_PRIORITY).fillna(3))
    return (frame.sort_values(["score", "chunk_rank"], ascending=[False, True])
            .drop(columns=["chunk_rank"]).reset_index(drop=True))


def build_context(state: "Engine", results: pd.DataFrame, max_docs: int = 4,
                  max_chars: int = 700, dedupe: str = "product") -> list[dict]:
    text_by_doc = state.chunks.set_index("doc_id")["text"]
    seen, context = set(), []
    for _, row in results.iterrows():
        key = row["doc_id"].split(":")[0] if dedupe == "product" else row["doc_id"]
        if key in seen:
            continue
        seen.add(key)
        text = row["text"] if "text" in results.columns and isinstance(row["text"], str) else text_by_doc.get(row["doc_id"], "")
        context.append({"doc_id": row["doc_id"], "text": (text or "")[:max_chars],
                        "title": row.get("title", "")})
        if len(context) >= max_docs:
            break
    return context


def scope_to_named_product(state: "Engine", query: str, results: pd.DataFrame):
    seeds = link_entities(state, query)
    product_seeds = [node for node, kind, _ in seeds if kind == "product"]
    if not product_seeds:
        return results, None

    product_id = product_seeds[0]
    wanted = [f"{product_id}:desc", f"{product_id}:spec", f"{product_id}:rev"]
    available = [doc_id for doc_id in wanted if doc_id in set(state.chunks["doc_id"])]
    text_by_doc = state.chunks.set_index("doc_id")["text"]
    title = state.products.set_index("product_id").loc[product_id, "title"]
    scoped = pd.DataFrame({
        "doc_id": available,
        "title": [title] * len(available),
        "text": [text_by_doc.loc[doc_id] for doc_id in available],
        "rrf_score": 1.0,
    })
    return scoped, product_id


def choose_sentences(query: str, scored: pd.DataFrame, threshold: float = ABSTAIN_THRESHOLD,
                     max_sentences: int = 3, per_doc_limit: int = 2) -> list:
    if scored.empty:
        return []

    attribute_list = query_attributes(query)
    attribute_set = set(attribute_list)
    primary = attribute_list[0] if attribute_list else None

    def primary_row():
        if not primary:
            return None
        for _, row in scored.head(20).iterrows():
            if row["score"] < 0.30:
                break
            if _matches_attribute(row["sentence"].lower(), primary):
                return row
        return None

    top_doc = scored.iloc[0]["doc_id"]
    picked, per_doc, seen_sentences = [], defaultdict(int), set()
    covered: set[str] = set()

    forced = primary_row()
    if forced is not None:
        picked.append(forced)
        per_doc[forced["doc_id"]] += 1
        seen_sentences.add(forced["sentence"])
        covered |= attributes_present(forced["sentence"])

    for _, row in scored.iterrows():
        if row["score"] < threshold * 0.9:
            break
        if len(picked) >= max_sentences:
            break
        if row["sentence"] in seen_sentences:
            continue
        if per_doc[row["doc_id"]] >= per_doc_limit:
            continue
        if row["doc_id"] != top_doc:
            adds_value = attributes_present(row["sentence"]) & (attribute_set - covered)
            if not adds_value:
                continue
        picked.append(row)
        per_doc[row["doc_id"]] += 1
        seen_sentences.add(row["sentence"])
        covered |= attributes_present(row["sentence"])

    return sorted(picked, key=lambda row: (CHUNK_PRIORITY.get(row["doc_id"].split(":")[1], 3),
                                           row["doc_id"], row["pos"]))


def extractive_answer(state: "Engine", query: str, k: int = 6, max_sentences: int = 3,
                      threshold: float = ABSTAIN_THRESHOLD) -> dict:
    results = hybrid_search(state, query, k=k)
    retrieved = results
    results, scoped_product = scope_to_named_product(state, query, results)
    context = build_context(state, results, max_docs=4,
                            dedupe="chunk" if scoped_product else "product")
    scored = score_sentences(query, context)

    base = {
        "query": query, "scoped_to": scoped_product, "evidence": scored,
        "results": retrieved, "context": context, "threshold": threshold,
    }

    if scored.empty or scored["score"].iloc[0] < threshold:
        return {**base, "abstained": True, "answer": None, "citations": [],
                "confidence": float(scored["score"].iloc[0]) if len(scored) else 0.0,
                "reason": f"best sentence score {scored['score'].iloc[0]:.2f} < threshold {threshold}"
                if len(scored) else "no candidate sentences"}

    attrs = query_attributes(query)
    if attrs:
        primary = attrs[0]
        candidates = scored.head(3)["sentence"].str.lower()
        if not candidates.apply(lambda sentence: _matches_attribute(sentence, primary)).any():
            return {**base, "abstained": True, "answer": None, "citations": [],
                    "confidence": float(scored["score"].iloc[0]),
                    "reason": f"no candidate contains the asked attribute: {primary}"}

    picked = choose_sentences(query, scored, threshold, max_sentences)
    if not picked:
        return {**base, "abstained": True, "answer": None, "citations": [],
                "confidence": float(scored["score"].iloc[0]),
                "reason": "no sentence passed the grounding rules"}

    answer = " ".join(row["sentence"] for row in picked)
    confidence = float(np.mean([row["score"] for row in picked]))
    return {**base, "abstained": False, "answer": answer,
            "citations": list(dict.fromkeys(row["doc_id"] for row in picked)),
            "confidence": round(confidence, 4), "picked": picked}


CLAIM_PATTERN = re.compile(
    r"(?:rs\.?\s?\d[\d,]*|\b\d[\d,.]*\s?(?:w|gb|mb|mah|months?|hours?|hrs?|inch|mp|%)\b"
    r"|\b\d[\d,]*\b|[a-z]{2,}-\d[\w-]*)",
    flags=re.IGNORECASE,
)


def extract_claims(text: str) -> list[str]:
    claims = [match.group(0).strip().lower() for match in CLAIM_PATTERN.finditer(text)]
    return [claim for claim in claims if len(claim) > 1]


def verify_answer(state: "Engine", answer: str, citations: list[str]) -> pd.DataFrame:
    text_by_doc = state.chunks.set_index("doc_id")["text"]
    blob = " ".join(text_by_doc.get(doc_id, "") for doc_id in citations).lower()
    blob = re.sub(r"\s+", " ", blob)
    rows = []
    for claim in extract_claims(answer or ""):
        normalized = re.sub(r"\s+", " ", claim)
        rows.append({"claim": claim, "supported": normalized in blob})
    return pd.DataFrame(rows, columns=["claim", "supported"])


def simulated_naive_generator(query: str) -> str:
    seed_value = int(hashlib.sha1(query.encode()).hexdigest()[:8], 16)
    fake_prices = [21499, 24799, 26299, 18999, 23999]
    fake_price = fake_prices[seed_value % len(fake_prices)]
    fake_warranty = [36, 24, 48][seed_value % 3]
    fake_months = 1 + seed_value % 5
    return (
        f"Great question. Based on the product listing, this model is available at "
        f"Rs {fake_price:,} and comes with a {fake_warranty}-month warranty. "
        f"It also ships with a free hard case and supports {fake_months}-month no-cost EMI. "
        f"Customers typically report 40 hours of battery life."
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _rel_by_rule(products: pd.DataFrame, rule) -> set[str]:
    return set(products.loc[rule(products), "product_id"])


def build_qrels(state: "Engine") -> dict[str, tuple[str, set[str]]]:
    products = state.products
    graph = state.graph
    qrels: dict[str, tuple[str, set[str]]] = {}

    qrels["Sony WH-1000XM4"] = ("exact-id", _rel_by_rule(
        products, lambda d: (d["brand"] == "Sony") & (d["subcategory"] == "WH-1000XM4")))
    qrels["65W USB-C charger"] = ("exact-id", _rel_by_rule(
        products, lambda d: (d["category"] == "Chargers")
        & (d["specs"].apply(lambda s: s.get("watts") == 65))))
    qrels["IP68 earbuds"] = ("exact-id", _rel_by_rule(
        products, lambda d: (d["category"] == "Earbuds")
        & (d["specs"].apply(lambda s: s.get("ip_rating") == "IP68"))))
    qrels["RTX 4060 laptop"] = ("exact-id", _rel_by_rule(
        products, lambda d: (d["category"] == "Laptops")
        & (d["specs"].apply(lambda s: "RTX 4060" in str(s.get("gpu", ""))))))

    qrels["headphones that silence aeroplane engine noise"] = ("conceptual", _rel_by_rule(
        products, lambda d: (d["category"] == "Headphones")
        & (d["specs"].apply(lambda s: s.get("anc") is True))))
    qrels["something to keep my phone alive on a long trek"] = ("conceptual", _rel_by_rule(
        products, lambda d: (d["category"] == "Power Banks")
        & (d["specs"].apply(lambda s: s.get("capacity_mah", 0) >= 20000))))
    qrels["a fast machine for editing 4K video"] = ("conceptual", _rel_by_rule(
        products, lambda d: (d["category"] == "Laptops")
        & (d["specs"].apply(lambda s: s.get("ram_gb", 0) >= 16))
        & (d["specs"].apply(lambda s: "RTX" in str(s.get("gpu", ""))))))
    qrels["earbuds that will not fall out while running"] = ("conceptual", _rel_by_rule(
        products, lambda d: (d["category"] == "Earbuds")
        & (d["specs"].apply(lambda s: s.get("ip_rating") in {"IPX5", "IP55", "IP57", "IP68"}))))

    xm4 = products.loc[(products["brand"] == "Sony")
                       & (products["subcategory"] == "WH-1000XM4"), "product_id"].iloc[0]
    xm5 = set(neighbours(graph, xm4, "SUCCESSOR_OF", "out"))
    xm6 = set()
    for node in xm5:
        xm6 |= set(neighbours(graph, node, "SUCCESSOR_OF", "out"))
    qrels["what replaced the Sony WH-1000XM4"] = ("relational", xm5 | xm6)

    xps = products.loc[products["subcategory"] == "XPS 15 9530", "product_id"].iloc[0]
    qrels["accessories compatible with the Dell XPS 15 9530"] = ("relational", {
        node for node in neighbours(graph, xps, "COMPATIBLE_WITH", "both")
        if graph.nodes[node]["kind"] == "product" and graph.nodes[node]["category"] != "Laptops"})

    macbook = products.loc[products["subcategory"] == "MacBook Air M3 13", "product_id"].iloc[0]
    qrels["which charger works with a MacBook Air M3 13"] = ("relational", {
        node for node in neighbours(graph, macbook, "COMPATIBLE_WITH", "both")
        if graph.nodes[node]["category"] == "Chargers"})

    qrels["return policy for earbuds"] = ("relational", {"POL002"})
    return qrels


def doc_key(doc_id: str) -> str:
    return doc_id.split(":")[0]


def _dedupe_ordered(keys: list[str]) -> list[str]:
    seen, out = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _hit_at_k(ranked, relevant, k):
    return float(any(item in relevant for item in ranked[:k]))


def _precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


def _recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def _mrr(ranked, relevant):
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def _ndcg_at_k(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(i + 1)
              for i, item in enumerate(ranked[:k], start=1) if item in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(state: "Engine", retriever: str, k: int = 5) -> pd.DataFrame:
    qrels = state.qrels
    rows = []
    for query, (intent, relevant) in qrels.items():
        if retriever == "semantic":
            ids = dense_rank_ids(state, query, 50)
        elif retriever == "sparse":
            ids = sparse_rank_ids(state, query, 50)
        elif retriever == "graph":
            ids = graph_rank_ids(state, query, 25)
        else:
            ids = hybrid_search(state, query, k=8)["doc_id"].tolist()
        ranked = _dedupe_ordered([doc_key(doc_id) for doc_id in ids])
        rows.append({
            "retriever": retriever, "intent": intent, "query": query,
            f"hit@{k}": _hit_at_k(ranked, relevant, k),
            f"precision@{k}": round(_precision_at_k(ranked, relevant, k), 3),
            f"recall@{k}": round(_recall_at_k(ranked, relevant, k), 3),
            "mrr": round(_mrr(ranked, relevant), 3),
            f"ndcg@{k}": round(_ndcg_at_k(ranked, relevant, k), 3),
        })
    return pd.DataFrame(rows)


def evaluate_all(state: "Engine", k: int = 5) -> pd.DataFrame:
    frames = [evaluate(state, name, k) for name in ("semantic", "sparse", "graph", "hybrid")]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Engine facade
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.products, self.reviews, self.policies, self.relations = load_frames()
        self.chunks = build_chunks(self.products, self.reviews, self.policies)
        self.chunk_vecs, self.embedding_info = load_or_build_embeddings(self.chunks)
        self.bm25 = BM25Index(self.chunks["text"].tolist())
        self.graph = build_graph(self.products, self.policies, self.relations)
        self.product_lookup = [
            (row["product_id"], _squeeze(row["subcategory"]), _squeeze(row["sku"]))
            for _, row in self.products.iterrows()
        ]
        self.brand_lookup = [(f"BRAND:{brand}", _squeeze(brand))
                             for brand in self.products["brand"].unique()]
        self.category_lookup = [(f"CAT:{category}", _squeeze(category))
                                for category in self.products["category"].unique()]
        self.qrels = build_qrels(self)

    def stats(self) -> dict:
        return {
            "products": len(self.products),
            "chunks": len(self.chunks),
            "reviews": len(self.reviews),
            "policies": len(self.policies),
            "relations": len(self.relations),
            "brands": int(self.products["brand"].nunique()),
            "categories": int(self.products["category"].nunique()),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "vocab": len(self.bm25.vocab),
            "avgdl": round(self.bm25.avgdl, 1),
        }


def search(state: Engine, method: str, query: str, k: int = 8,
           chunk_type: str | None = None, category: str | None = None,
           max_price: float | None = None) -> pd.DataFrame:
    if method == "semantic":
        return semantic_search(state, query, k=k, chunk_type=chunk_type,
                               category=category, max_price=max_price)
    if method == "sparse":
        frame = bm25_search(state, query, k=k * 3 if category or max_price else k,
                            chunk_type=chunk_type)
        return enrich(state, frame).head(k)
    if method == "graph":
        ids = graph_rank_ids(state, query, k=k)
        if not ids:
            return pd.DataFrame(columns=["doc_id", "title", "category", "price_inr", "chunk", "text"])
        frame = pd.DataFrame({"doc_id": ids, "rrf_score": 1.0})
        return enrich(state, frame)
    return hybrid_search(state, query, k=k)
