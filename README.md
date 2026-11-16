# RAG Retrieval Lab

**Experimental project.** This repository is purely for educational and experimental purposes.
All product, review, policy and relation data is **synthetic** and does not resemble any real
company, catalogue or industry dataset. Nothing here should be used for production or commercial
decisions.

A hands-on comparison of three retrieval families — semantic (dense), sparse (BM25) and graph —
built on one synthetic e-commerce catalogue, plus the answering layer that keeps the output
grounded.

## What is inside

| Path | What it is |
|---|---|
| `RAG_Retrieval_Comparative_Tutorial.ipynb` | Self-contained tutorial, executed end to end. Explains and implements all three retrievers, evaluates them, and builds a citation-backed answering layer. |
| `streamlit_app/` | Nine-page interactive lab that mirrors the notebook. Every demo shows the experiment on the left and a "why / when it breaks" note on the right. |
| `Seminar_Script_14Nov2026.docx` | Presenter script: 45-minute walkthrough plus 15-minute Q&A preparation. |
| `data/` | The synthetic catalogue: products, reviews, policies, typed relations and evaluation labels. |
| `run_app.bat` | One-click launcher for the Streamlit app. |
| `requirements.txt` | Pinned versions, validated on Python 3.14 / Windows 11. |

## The dataset

207 products across 12 categories, 716 reviews, 25 policy documents and 1,544 typed relations.
It is generated deterministically from a fixed seed, and it is deliberately hard:

- near-duplicate model names (`WH-1000XM4` / `XM5` / `XM6`)
- exact identifiers (SKUs, part numbers)
- numeric specs (`65W`, `IP68`, `RTX 4060`)
- synonyms and paraphrases that share no words with the catalogue
- relational facts that exist as edges, not sentences
- abstract policy prose (returns, warranty, EMI, installation)

Every failure demo in the notebook and the app comes from one of these rows.

## Quick start

```bat
python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name rag-seminar --display-name "RAG Seminar (Python 3.14)"
run_app.bat
```

The app opens at <http://localhost:8501>. Select the **RAG Seminar (Python 3.14)** kernel for the
notebook.

First run downloads `all-MiniLM-L6-v2` (~90 MB) once and caches it. The embedding cache and the
Chroma index are not committed; both are rebuilt automatically on first use (about 40 seconds).

## What the demos show

| Query | Winner | Why |
|---|---|---|
| `RTX 4060 laptop` | sparse | exact token; embeddings blur 4060 and 4070 |
| "block out engine noise on a flight" | dense | meaning, with zero shared words |
| "accessories compatible with the Dell XPS 15" | graph | set membership over edges |
| "laptop under 70,000" | hybrid + filters | numbers belong in metadata filters |

Measured on 12 labelled queries (`k = 5`):

| Retriever | Hit@5 | nDCG@5 | Mean latency (CPU) |
|---|---|---|---|
| Semantic | 0.750 | 0.463 | ~16 ms |
| Sparse (BM25) | 0.833 | 0.672 | ~2.5 ms |
| Graph | 0.333 | 0.333 | <1 ms |
| Hybrid (RRF) | **1.000** | **0.838** | ~30 ms |

Graph scores low overall by design: the router only runs it on relational queries, where it is
the only method that works.

## Design notes

- **Field-aware chunking.** One product becomes three chunks (description, specifications, review
  digest) so each chunk stays coherent and inside the model's 256-token window.
- **Routing before fusion.** Digit-bearing queries trust BM25, relational queries enable the graph,
  policy vocabulary enables a policy ranker. Fusion is Reciprocal Rank Fusion on ranks, not scores.
- **Grounded answering.** Answers are assembled from verbatim sentences, every numeric claim is
  verified against the cited source, and the system abstains when the evidence is missing.

## Reproducibility

The notebook and the app share the same seed, corpus hash and evaluation labels. The app asserts
that the cached embeddings match a SHA-1 of the chunk text and rebuilds them if they do not, so a
fresh clone produces the same numbers.

## Acknowledgements

Built with `sentence-transformers`, `ChromaDB`, `rank_bm25`, `NetworkX`, `scikit-learn` and
`Streamlit`. Model: `sentence-transformers/all-MiniLM-L6-v2`.
