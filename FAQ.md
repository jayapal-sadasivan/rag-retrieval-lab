# Frequently Asked Questions

Twenty questions that come up after the seminar, with short answers - except Q16, which
gets the detail it deserves.

> **Note on the data.** This project is for educational and experimental purposes. Every
> product, review, policy and relation in the catalogue is **synthetic** and does not
> resemble any real company or industry dataset.

**Contents:** [1](#1-what-is-rag-and-how-is-it-different-from-search) | [2](#2-why-not-just-fine-tune-the-model-on-our-catalogue) | [3](#3-what-actually-is-an-embedding) | [4](#4-why-cosine-similarity-rather-than-euclidean-distance) | [5](#5-why-does-semantic-search-fail-on-rtx-4060) | [6](#6-what-is-bm25-and-how-does-it-differ-from-tf-idf) | [7](#7-what-do-k1-and-b-actually-control) | [8](#8-is-bm25-obsolete-in-2026) | [9](#9-what-is-a-chunk-and-how-do-you-choose-chunk-size) | [10](#10-what-is-hnsw-and-what-does-approximate-cost-us-at-10-million-documents) | [11](#11-how-does-reciprocal-rank-fusion-work-and-why-fuse-ranks-instead-of-scores) | [12](#12-how-does-the-router-decide-which-retriever-runs) | [13](#13-what-is-a-knowledge-graph-and-when-is-it-better-than-text) | [14](#14-how-do-you-answer-what-replaced-the-xm4---walk-the-hops) | [15](#15-what-is-graphrag-local-versus-global-search) | [16](#16-how-is-the-metadata-filter-of-rs-70000-when-we-say-under-70000-rendered-from-the-vector-space-and-filtered) | [17](#17-how-do-you-evaluate-a-retriever---hitk-mrr-ndcg) | [18](#18-why-does-bm25-beat-dense-retrieval-on-conceptual-queries-here) | [19](#19-how-do-you-stop-the-model-from-hallucinating) | [20](#20-when-should-the-system-refuse-to-answer-and-how-do-you-set-the-threshold)

---

## 1. What is RAG, and how is it different from search?

Retrieval-augmented generation fetches relevant passages and places them in the model's
context before it answers. Search returns a list of documents; RAG returns a written answer
grounded in them. The model never sees what the retriever did not fetch, so retrieval quality
- not prompt wording - is the biggest lever on accuracy.

## 2. Why not just fine-tune the model on our catalogue?

Fine-tuning changes behaviour and style; it does not give the model a reliable, current copy
of your catalogue. A fine-tuned model still cannot cite sources, still drifts when data
changes, and still invents specifics. Retrieval keeps knowledge external, attributable and
updatable. Use fine-tuning for tone and format, and retrieval for facts.

## 3. What actually is an embedding?

A fixed-length vector of numbers - 384 here - that represents meaning. A transformer is
trained on millions of sentence pairs so that paraphrases land close together and unrelated
text lands far apart. You encode the corpus once, encode the query at request time, and rank
by vector similarity.

## 4. Why cosine similarity rather than Euclidean distance?

Cosine compares direction and ignores magnitude, which is what "similar meaning" looks like in
these spaces. If every vector is L2-normalised at encode time, cosine similarity becomes a
plain dot product, which is fast. Euclidean distance on unnormalised vectors is dominated by
length rather than meaning.

## 5. Why does semantic search fail on `RTX 4060`?

Model numbers are tokenised into sub-words, and `4060` / `4070` appear in almost identical
surrounding text. The embedding captures the topic ("gaming laptop") but not the exact
identifier. In the demo, the top hit for "RTX 4060 laptop" was a 4070 machine. Exact tokens
belong in a lexical index.

## 6. What is BM25, and how does it differ from TF-IDF?

BM25 is a bag-of-words ranker. TF-IDF multiplies raw term frequency by inverse document
frequency; BM25 adds two corrections: term-frequency saturation and document-length
normalisation. Those corrections stop a keyword-stuffed or unusually long document from
winning simply by volume. It needs no training and no GPU.

## 7. What do `k1` and `b` actually control?

`k1` controls saturation: how quickly repeated occurrences of a term stop adding score.
`b` controls length normalisation: `0` ignores document length, `1` applies it fully. The
defaults used here are `k1 = 1.5` and `b = 0.75`, which behave well across domains.

## 8. Is BM25 obsolete in 2026?

No. On this catalogue it scores nDCG@5 = 1.000 on exact-identifier queries versus 0.555 for
dense retrieval, runs in about 2.5 ms on CPU, requires no model download, and can explain
every point of its score. Most production search is hybrid, not pure vector.

## 9. What is a chunk, and how do you choose chunk size?

A chunk is the unit that is indexed and later pasted into the context window. This project
uses field-aware chunks: description, specifications and review digest per product, averaging
about 55 words and staying well inside the model's 256-token window. Too large dilutes
similarity; too small loses the context that makes a sentence interpretable. Test two or three
sizes against labelled queries rather than guessing.

## 10. What is HNSW, and what does "approximate" cost us at 10 million documents?

HNSW is a multi-layer proximity graph searched by greedy descent, giving roughly
`O(log N)` lookups instead of a full scan. "Approximate" means recall below 100%, traded
against latency and memory through `M`, `ef_construction` and `ef_search`. At 10 million
documents you shard the index, quantise vectors, cache query embeddings, and expect filtered
searches to need larger candidate lists.

## 11. How does Reciprocal Rank Fusion work, and why fuse ranks instead of scores?

For each document, add `weight / (60 + rank)` across every ranked list, then sort. BM25 scores
are unbounded while cosine sits between -1 and 1, so adding raw scores lets one retriever
dominate arbitrarily. Ranks are comparable by construction. RRF needs almost no tuning, which
is why Elasticsearch, Vespa and Azure AI Search ship it.

## 12. How does the router decide which retriever runs?

Cheap rules on the query text. A token containing digits (`4060`, `65W`, `XM4`) raises the
sparse weight; compatibility or succession phrasing enables the graph; returns, warranty, EMI
or shipping vocabulary enables the policy ranker. When no signal fires, all lists run with
equal weight. It is a deliberate first pass that a trained classifier can replace later.

## 13. What is a knowledge graph, and when is it better than text?

A set of typed triples - head, relation, tail - such as `(ZP0064, COMPATIBLE_WITH, ZP0143)`.
It wins when the fact is an edge rather than a sentence: compatibility, succession, policy
applicability, multi-hop questions. It loses when the edge does not exist, because a graph
cannot improvise the way an embedding can.

## 14. How do you answer "what replaced the XM4?" - walk the hops

Entity-link the query to the XM4 node, then follow `SUCCESSOR_OF` outwards: XM5, then XM6.
Each step returns the product, its price and warranty, and the path itself is the provenance.
That path is why graphs are attractive in audited settings.

## 15. What is GraphRAG, local versus global search?

Local search links entities, expands a neighbourhood and reranks the passages it reaches -
what this app does. Global search pre-computes summaries of graph communities and answers
corpus-level questions ("what are the main themes?") from those summaries. Local for specific
multi-hop facts, global for thematic questions.

## 16. How is the metadata filter of Rs 70,000 when we say "under 70000" rendered from the vector space and filtered?

**Short answer: it is not rendered in the vector space at all.** Price is a structured field,
and `price_inr <= 70000` is a boolean mask applied to the candidate set. The vectors only rank
what survives the filter.

### 1. From sentence to constraint

The user writes "under 70000". The system needs `price_inr <= 70000`, a typed comparison. Here
a small rule turns the phrase into `max_price = 70000`. In production an LLM tool-call or a
query parser emits the same structure:

```json
{"query": "laptop for editing 4K video", "filters": {"category": "Laptops", "price_inr": {"$lte": 70000}}}
```

### 2. Why the embedding cannot do this

An embedding is trained on semantic similarity, not arithmetic. Three consequences:

- `4060` and `4070` are near-identical sub-word tokens, so numbers are weakly represented.
- Cosine measures closeness of meaning; it has no concept of "less than".
- Two products at Rs 69,999 and Rs 129,999 can be equally similar to the query phrase, because
  similarity reflects topic, not magnitude.

You can sometimes teach a model that "budget" correlates with low price, but you cannot make
it honour an exact threshold. Inequalities belong to databases.

### 3. Where the filter lives

Every chunk carries metadata alongside its text and vector:

```text
product_id, sku, brand, category, price_inr, rating, warranty_months, stock_units, launch_year
```

`price_inr` is an integer. Filtering is a comparison over that column, and vector search runs
inside the surviving subset.

### 4. Our implementation

`semantic_search()` scores all 646 chunks, then keeps only eligible ones:

```python
scores = chunk_vecs @ query_vec          # cosine over the whole corpus
mask = np.ones(len(chunks_df), dtype=bool)

if category:
    mask &= (chunks_df["category"] == category).to_numpy()
if max_price is not None:
    mask &= (metadata_array("price_inr") <= max_price)

scores = np.where(mask, scores, -np.inf)  # ineligible items can never rank
top = np.argsort(-scores)[:k]             # top-k drawn from eligible items only
```

Setting excluded scores to `-inf` is the key detail: the top-k is selected *after* masking, so
filtering cannot silently remove results from an already-chosen list.

### 5. The vector-database equivalent

In ChromaDB the same constraint is pushed into the query:

```python
collection.query(
    query_texts=["machine for editing 4K video on the go"],
    n_results=5,
    where={"$and": [{"category": "Laptops"}, {"price_inr": {"$lte": 70000}}]},
)
```

Where the filter is applied matters at scale:

| Strategy | When it runs | Trade-off |
|---|---|---|
| Pre-filter | before graph traversal | exact, but very selective filters can degrade HNSW recall and speed |
| Post-filter | after top-k retrieval | simple, but may return fewer than k results; you must over-fetch |
| Application mask | after full scoring | fine for hundreds of chunks, wrong for millions |

Our numpy version is the third row. That is appropriate at 646 chunks and a poor choice at
10 million, where the filter must be pushed into the index.

### 6. Worked example from the demo

| Step | Result |
|---|---|
| Unfiltered semantic top-1 | Samsung Galaxy Book4 Pro at **Rs 129,990** |
| With `category = Laptops` and `price_inr <= 70,000` | Rs 69,990 | Rs 62,990 | Rs 59,990 | Rs 54,990 | Rs 46,990 |
| Constraint satisfied by every row | yes |

The ranking inside the filtered set is still semantic: Nitro V 15, RedmiBook Pro 15 and
Inspiron 14 come back in cosine order. The filter changed *who was allowed to compete*, not how
they were scored.

### 7. Engineering details that bite

- **Units.** Store integer rupees or paise consistently; mixing them silently invalidates bounds.
- **Bounds.** `$lte` versus `$lt` is a business decision, and the boundary is a real product.
- **Missing values.** Null prices must be excluded or defaulted explicitly; a null compares
  unpredictably in some engines.
- **Currency.** Multi-currency catalogues need a normalised field, not a string.
- **Combinators.** Real queries need `$and` across category, price, warranty and stock.
- **Schema discipline.** You cannot filter on a number you never extracted. If price lives only
  inside marketing prose, the filter is impossible.

### 8. Takeaway

Vectors rank **meaning**; the database filters **facts**. A RAG system needs both, and the
order - filter first, rank second - is what keeps the answer correct.

## 17. How do you evaluate a retriever - Hit@k, MRR, nDCG?

Hit@k asks whether anything relevant appears in the top k. MRR is one divided by the rank of
the first relevant hit, so position matters. nDCG discounts each relevant item by its position
and normalises by the ideal ordering. Always slice by query type: an average that hides a zero
on exact-identifier queries is worse than no average.

## 18. Why does BM25 beat dense retrieval on "conceptual" queries here?

Because e-commerce copy is keyword-dense. It literally says `IPX5`, `mAh`, `noise cancellation`
and `video editing`. When the user's words appear verbatim in the corpus, BM25 finds them
exactly. Dense retrieval wins where the phrasing diverges from catalogue language - "a fast
machine for editing 4K" - which is the long tail you cannot cover with keyword rules.

## 19. How do you stop the model from hallucinating?

With an architecture, not a prompt. Every sentence must come from a retrieved span, every
answer carries resolvable citations, and a mechanical check verifies each numeric claim against
the cited text. In production you keep the verifier, add a regenerate-once step, and refuse
when verification fails.

## 20. When should the system refuse to answer, and how do you set the threshold?

When retrieval confidence is low, or when no candidate sentence contains the attribute the
question asked for. This project gates on a score threshold of 0.40 plus that attribute check,
and the threshold is tuned on labelled data by sweeping it and plotting answered-versus-wrong.
A confident "I don't have that information" is better than an invented price, warranty or
safety claim.

---

*Companion files: `RAG_Retrieval_Comparative_Tutorial.ipynb` for the full code,
`streamlit_app/` for the interactive lab, and `Seminar_Script_14Nov2026.docx` for the
presenter walkthrough.*