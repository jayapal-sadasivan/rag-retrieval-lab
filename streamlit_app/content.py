"""All explanatory copy for the app, kept in one place.

Each entry maps to a demo and renders in the right-hand note panel:
    see    - what the viewer is looking at
    why    - the mechanism behind it
    breaks - the failure mode to watch for
    try    - a next action / question for the room
"""

NOTES: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- overview
    "pipeline": {
        "see": "Four boxes: retriever, context packer, generator, answer. The "
               "retriever is the only one that touches your data.",
        "why": "The generator can only reason over what the retriever puts in "
               "front of it. If the right passage never arrives, the model either "
               "says it does not know (rare) or invents something plausible (common).",
        "breaks": "Teams spend weeks on prompts and swap only the LLM. Quality "
                  "moves far more when the retriever changes.",
        "try": "Ask the room: which box would you fix first for a product-search bot?",
    },

    # ----------------------------------------------------------------- dataset
    "catalogue": {
        "see": "207 products across 12 categories, 716 reviews, 25 policy documents "
               "and 1,544 typed relations.",
        "why": "Real catalogues are messy: casing noise, near-duplicate model names, "
               "missing fields, Hinglish reviews. We reproduce that on purpose so the "
               "retrieval failures you see later are realistic, not contrived.",
        "breaks": "A clean benchmark hides the failures that matter in production.",
        "try": "Filter to Headphones and compare 'Sony WH-1000XM4', XM5 and XM6 titles.",
    },
    "chunking": {
        "see": "Each product becomes three chunks: description, specifications, and a "
               "review digest. Policy documents become one chunk each.",
        "why": "Retrieval works on chunks, not rows. Field-aware chunking keeps each "
               "chunk coherent (~55 words) and inside the 256-token window, while the "
               "spec chunk concentrates the exact tokens BM25 needs.",
        "breaks": "One giant chunk per product dilutes similarity and buries the "
                  "answer. Too-small chunks lose the context that makes a sentence "
                  "interpretable.",
        "try": "Compare the desc and spec chunk of the same product in the inspector.",
    },

    # ---------------------------------------------------------------- semantic
    "cosine": {
        "see": "A 5x5 cosine-similarity matrix between sentences that barely share "
               "words.",
        "why": "A bi-encoder maps text to a 384-dimensional vector. Meaning becomes "
               "geometry: 'block engine noise on a plane' lands near 'noise cancelling' "
               "even with zero word overlap.",
        "breaks": "Similarity is not truth. Two products can be near-identical in "
                  "vector space while differing in the one detail the user asked for.",
        "try": "Find the pair with high similarity but no shared words (S1 vs S2).",
    },
    "pca": {
        "see": "All 646 chunk embeddings projected to two dimensions, coloured by "
               "category, with four queries as stars.",
        "why": "Related products cluster: laptops with laptops, chargers with chargers. "
               "Queries land inside the cluster they mean, which is why top-k retrieval "
               "works at all.",
        "breaks": "PCA keeps only ~14% of the variance here. Clusters are meaningful; "
                  "exact distances are not.",
        "try": "Point at where the 'can I return opened earbuds' star lands - the "
               "policy cluster, not the earbuds cluster.",
    },
    "filter": {
        "see": "The same query run twice: once pure semantic, once with price and "
               "category filters applied before the vector search.",
        "why": "Embeddings encode meaning, not arithmetic. 'under 70,000' is invisible "
               "to them; a structured filter is the correct tool. Every production "
               "vector store exposes query(vector, where={...}).",
        "breaks": "Filtering on price alone lets a 12,999 power bank through. Every "
                  "hard constraint in the query needs its own filter.",
        "try": "Remove the category filter and watch the power bank reappear.",
    },
    "failure": {
        "see": "Dense retrieval asked for 'RTX 4060 laptop'. The top hit is a 4070 "
               "machine; a monitor sneaks into the top few.",
        "why": "Model numbers are tokenised into sub-words. 4060 and 4070 are nearly "
               "the same vector, and the surrounding copy is identical. The one signal "
               "that disambiguates is the literal token.",
        "breaks": "Any query that hinges on an exact identifier: SKUs, part numbers, "
                  "wattages, error codes, IP ratings.",
        "try": "Ask which retriever you would trust for a support ticket containing "
               "an error code.",
    },
    "chroma": {
        "see": "The same corpus in ChromaDB, queried with a metadata filter and "
               "persisted to disk.",
        "why": "A vector database gives you persistence, metadata filtering and an "
               "approximate nearest-neighbour index (HNSW). HNSW trades a little "
               "recall for O(log N) queries instead of O(N).",
        "breaks": "ANN is approximate: tune ef_search for the SLA. And the index does "
                  "not fix bad chunks - it only finds them faster.",
        "try": "Compare cosine similarities here with the numpy search above.",
    },

    # ------------------------------------------------------------------ sparse
    "tokenize": {
        "see": "Raw text on the left, tokens after lowercasing, stopword removal, "
               "model-number preservation and light stemming on the right.",
        "why": "Tokenisation is the single most important choice in sparse retrieval. "
               "We emit both 'wh-1000xm5' and 'wh1000xm5' so a user typing either form "
               "still matches the catalogue.",
        "breaks": "Aggressive stemming merges distinct model names. No stemming misses "
                  "plurals. There is no universally right answer - test on your queries.",
        "try": "Type a query with a hyphenated model number and watch the tokens.",
    },
    "bm25": {
        "see": "BM25 scores for a query, with the k1 and b parameters exposed as sliders.",
        "why": "Two corrections to TF-IDF: term frequency saturates (k1) so a keyword "
               "stuffed document cannot dominate, and document length is normalised (b) "
               "so long documents do not win by volume.",
        "breaks": "Set b=0 and long spec sheets climb. Set k1 very low and a single "
                  "occurrence is all that counts.",
        "try": "Set b to 0, then to 1, on the same query and watch the ranking change.",
    },
    "explain": {
        "see": "Per-term contributions for the top document: IDF, term frequency, and "
               "the resulting score share.",
        "why": "Sparse retrieval is auditable. You can show a merchandiser exactly which "
               "token produced the ranking. Dense models cannot do this.",
        "breaks": "Explainability does not imply correctness: a rare-but-irrelevant term "
                  "can dominate a score.",
        "try": "Compare the contribution of 'sony' with 'wh-1000xm4'.",
    },
    "sparsefailure": {
        "see": "'Something cheap for the gym that will not fall out of my ears' returns "
               "mid-priced earbuds, chargers and a monitor.",
        "why": "Only 'cheap' and 'ear' survive tokenisation. 'gym', 'fall' and 'workout' "
               "are not in this catalogue's vocabulary, so there is nothing to match. "
               "BM25 has no concept of intent.",
        "breaks": "Every synonym, paraphrase and vague description is invisible to a "
                  "lexical index.",
        "try": "Ask the room to rewrite the query so BM25 succeeds - they will "
               "instinctively use catalogue words.",
    },

    # ------------------------------------------------------------------- graph
    "kg": {
        "see": "A knowledge graph: products, brands, categories, policies and suppliers "
               "as nodes, with typed edges between them.",
        "why": "Some facts are not sentences. 'Which charger works with this laptop' "
               "is a set-membership question over edges. No embedding approximates a join.",
        "breaks": "A missing edge is a silent failure - the graph cannot improvise the "
                  "way an embedding can.",
        "try": "Click a product and inspect its 2-hop neighbourhood.",
    },
    "compat": {
        "see": "One-hop traversal from a laptop to every accessory marked "
               "COMPATIBLE_WITH, with the path printed for each result.",
        "why": "Retrieval becomes traversal. The path is provenance: you can audit why "
               "each item was returned, which is why regulated industries like graphs.",
        "breaks": "Coverage. A product with no compatibility edges returns nothing at all.",
        "try": "Compare this with the dense results for the same question.",
    },
    "successor": {
        "see": "A version chain walked with SUCCESSOR_OF: XM4 to XM5 to XM6, with prices "
               "and warranty for each.",
        "why": "'What replaced X' and 'what is the newest in this line' are multi-hop "
               "questions. The graph answers them in microseconds and shows its work.",
        "breaks": "Cycles or stale edges produce confidently wrong version histories.",
        "try": "Ask for the predecessor of the newest model.",
    },
    "aggregation": {
        "see": "A GROUP BY over edges: which brands supply the most laptop-compatible "
               "accessories with at least 12 months of warranty.",
        "why": "This is a graph-native question. There is no passage that contains the "
               "answer; it is a property of the graph, computed by traversal.",
        "breaks": "Without a graph you would have to precompute and store this aggregate, "
                  "and it would go stale.",
        "try": "Raise the warranty filter to 24 months and watch the ranking reshuffle.",
    },
    "policy": {
        "see": "A two-hop path: category -> GOVERNED_BY -> policy document.",
        "why": "The question 'which policy applies to earbuds' has zero lexical overlap "
               "with the policy text. The edge is what connects them.",
        "breaks": "If the category-to-policy edge is missing, the answer is unreachable.",
        "try": "Ask the same question in the Grounded page and compare.",
    },

    # ------------------------------------------------------------------ hybrid
    "fusion": {
        "see": "Reciprocal Rank Fusion applied to three ranked lists, with the rank "
               "each retriever assigned to every result.",
        "why": "Fusing on ranks instead of scores avoids the scale problem: BM25 is "
               "unbounded, cosine lives in [-1, 1]. RRF only asks how high each list "
               "placed a document, with k=60 damping the curve.",
        "breaks": "If one retriever is consistently noisy, equal weights will drag the "
                  "fused ranking down. That is why we route.",
        "try": "Drag the graph weight to 0 and watch relational results degrade.",
    },
    "routing": {
        "see": "A banner naming which retrievers ran for this query, and why, plus the "
               "weights applied to each list.",
        "why": "Running every retriever on every query is the naive hybrid and it loses. "
               "Digit-bearing queries trust sparse; relational questions enable the graph; "
               "policy vocabulary enables the policy ranker.",
        "breaks": "Rule-based routing misclassifies edge cases. Use it as a cheap first "
                  "pass, and escalate to a classifier when rules disagree.",
        "try": "Type the same question with and without a model number.",
    },
    "compare": {
        "see": "The same query through all four retrievers, side by side, with latency.",
        "why": "Each column exposes a different blind spot. Reading them together shows "
               "why production systems run hybrids instead of picking a favourite.",
        "breaks": "A single query proves nothing. Section 7 aggregates twelve labelled "
                  "queries so the comparison is defensible.",
        "try": "Pick a query where the columns disagree completely.",
    },

    # -------------------------------------------------------------- evaluation
    "metrics": {
        "see": "Hit@5, Precision@5, Recall@5, MRR and nDCG@5 for each retriever, plus "
               "the per-query breakdown.",
        "why": "Hit@5 asks whether anything relevant appeared. MRR asks how high the "
               "first relevant hit was. nDCG discounts relevant items by position, so "
               "ranking quality matters, not just presence.",
        "breaks": "Averages hide intent-specific collapse. Always slice by query type.",
        "try": "Find a query where a retriever scores 0.0 and read why.",
    },
    "honesty": {
        "see": "Sparse beats dense on 'conceptual' queries here (0.692 vs 0.377), which "
               "contradicts the usual story.",
        "why": "E-commerce copy is keyword-dense: it literally contains IPX5, mAh, noise "
               "cancellation, video editing. When the user's words leak into the corpus, "
               "BM25 finds them exactly. 'Semantic' is a property of the query-corpus "
               "gap, not of the query alone.",
        "breaks": "Assuming dense is always better leads teams to delete a strong, cheap "
                  "baseline and lose recall.",
        "try": "Find the query where dense clearly wins; notice the paraphrase gap.",
    },

    # ---------------------------------------------------------------- grounding
    "naive": {
        "see": "A generated answer with three specific, plausible, completely invented "
               "numbers.",
        "why": "This simulates an LLM answering from parametric memory instead of the "
               "context. It is fluent and confident because that is what it was trained "
               "to be - nobody asked it to be grounded.",
        "breaks": "Every ungrounded numeric claim is a liability in support and commerce.",
        "try": "Compare the invented price with the actual catalogue record.",
    },
    "grounded": {
        "see": "An answer assembled from verbatim source sentences, each with a chunk id "
               "citation, plus a confidence score.",
        "why": "Extractive generation makes grounding mechanical: if a sentence is not "
               "in a cited chunk, it cannot appear. That turns trust into a string check.",
        "breaks": "Extractive answers are less fluent. In production you swap in an LLM "
                  "but keep the same contract and verifier.",
        "try": "Open a citation and confirm the sentence appears word for word.",
    },
    "abstain": {
        "see": "A question the corpus cannot answer, and a refusal with a numeric reason.",
        "why": "The system gates on retrieval confidence and on whether the asked "
               "attribute is present in any candidate. If not, it abstains instead of "
               "answering something adjacent.",
        "breaks": "Set the threshold too low and you get confident nonsense; too high and "
                  "you refuse answerable questions. Tune it on labelled data.",
        "try": "Ask about a product that does not exist and watch the reason change.",
    },
    "verify": {
        "see": "Every numeric or identifier claim in the answer, checked verbatim against "
               "the cited sources, with a PASS/FAIL verdict.",
        "why": "This is the guardrail you keep even after adding an LLM: parse claims, "
               "verify each against retrieved text, regenerate once, then abstain.",
        "breaks": "String verification misses semantic entailment. Pair it with a "
                  "faithfulness model for production coverage.",
        "try": "Run the naive answer through the same verifier.",
    },

    # --------------------------------------------------------------- playground
    "playground": {
        "see": "Your query, your retrievers, your filters - with latency and provenance.",
        "why": "The point of the session is prediction: before running, decide which "
               "retriever should win and why. Then check.",
        "breaks": "One query is an anecdote. Validate any hunch against the evaluation "
                  "page before changing production.",
        "try": "Write a query you expect sparse to win, then one you expect dense to win.",
    },
}


def note(key: str) -> dict[str, str]:
    return NOTES.get(key, NOTES["pipeline"])
