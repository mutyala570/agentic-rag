# search_utils.py — Walkthrough

Reference doc for `search_utils.py` — the retrieval engine that takes a user query and returns the top-k relevant chunks for the LLM to answer from. Implements Slide 3 (the two indexes) + Slide 4 (BM25, rerank) of the Day 1 Agentic RAG lecture.

## Contents

1. [30-second overview](#1-30-second-overview)
2. [Class 1 — `ChromaVectorStore`](#2-class-1--chromavectorstore)
3. [Class 2 — `BM25Store`](#3-class-2--bm25store)
4. [Class 3 — `SearchEngine`](#4-class-3--searchengine)
5. [The retrieval pipeline end-to-end](#5-the-retrieval-pipeline-end-to-end)
6. [Things to flag](#6-things-to-flag)
7. [Cross-reference back to the lecture](#7-cross-reference-back-to-the-lecture)

---

## 1. 30-second overview

Three classes do all the work:

| Class | Job |
|---|---|
| `ChromaVectorStore` | Wraps Chroma DB — semantic search via embeddings |
| `BM25Store` | Wraps the pickled BM25 index — keyword search |
| `SearchEngine` | Orchestrator — runs both in parallel, then reranks the union |

---

## 2. Class 1 — `ChromaVectorStore`

Thin wrapper around `chromadb.PersistentClient` — disk-backed vector DB.

### `load()` (line 29)

```python
self.client = chromadb.PersistentClient(path=str(self.chroma_path))
self.collection = self.client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
```

- **`PersistentClient`** reads/writes the on-disk Chroma store at `vectorstores/chroma/`.
- **`get_or_create_collection`** is idempotent — first run creates the `documents` collection; later runs reuse it.
- **`hnsw:space=cosine`** — Chroma uses HNSW (Hierarchical Navigable Small World) approximate-NN index with **cosine distance**. Standard fast ANN algorithm.

### `create_from_embeddings()` (line 42)

Called by `ingest.py` at offline-pipeline time. Splits the input dataframe into three parallel lists (`ids`, `documents`, `metadatas`), then `collection.add(...)`. Chroma persists everything to disk.

### `search()` (line 56)

Called at query time. Returns top-k nearest neighbours by cosine distance. The line:

```python
record = {"text": doc, "score": float(1 - dist), **meta}
```

converts cosine **distance** → cosine **similarity** (`1 - dist`). Higher score = more similar.

---

## 3. Class 2 — `BM25Store`

Wraps `rank_bm25.BM25Okapi` — classical keyword retrieval, no neural model.

### `load()` (line 83)

```python
with open(self.bm25_index_path, "rb") as f:
    self.bm25 = pickle.load(f)
```

The whole BM25 index lives in memory as one Python object, pickled to `bm25.pkl`. Same for `bm25_metadata.pkl`. **This is the Slide 3 "pkl" file** you saw in the screenshot.

### `create_from_texts()` (line 101)

```python
tokenized_corpus = [text.lower().split() for text in texts]
self.bm25 = BM25Okapi(tokenized_corpus)
```

**Naive tokenization** — lowercase + split-on-whitespace. No stemming, no stop-words. Good enough for technical docs like PostgreSQL (terms like `shared_buffers` stay intact). For multilingual or grammar-heavy corpora you'd want a proper tokenizer (spaCy, etc.).

### `search()` (line 107)

```python
tokenized_query = query.lower().split()
scores = self.bm25.get_scores(tokenized_query)
top_indices = np.argsort(scores)[::-1][:top_k]
```

Scores **every** document against the query, then `argsort` for top-k. Full-scan, not indexed — fine for ≤ 10k chunks; gets slow at 100k+.

---

## 4. Class 3 — `SearchEngine`

The orchestrator. Has four collaborators:

1. `vector_store` — `ChromaVectorStore`
2. `bm25_store` — `BM25Store`
3. `embedding_generator` — converts query string to vector (from `embed.py`)
4. `reranker` — `CrossEncoder` model for reranking

### `hybrid_search()` — the main method (line 164)

Four steps:

**Step 1: Embed the query (line 168)**
```python
query_vector = self.embedding_generator.embed_query([query])
```
Run the query string through the embedding model → single 384-dim vector for MiniLM.

**Step 2: Concurrent BM25 + Vector search (lines 170–175)**
```python
with ThreadPoolExecutor() as executor:
    bm25_future = executor.submit(self.bm25_store.search, query, top_k)
    vector_future = executor.submit(self.vector_store.search, query_vector, top_k)

    bm25_results = bm25_future.result()
    vector_results = vector_future.result()
```
Both searches run in parallel threads. BM25 doesn't wait for Chroma, and vice versa. Halves wall-clock latency. Threads work here because both are I/O-bound (BM25 = in-memory numpy; Chroma = disk-backed sqlite — both release the GIL).

**Step 3: Union + dedup (line 179)**
```python
unique_texts = list(set(bm25_texts + vector_texts))
```
**Note: NOT Reciprocal Rank Fusion** — plain set union. Up to 20 candidate texts (top-10 from each side, minus duplicates). All rank information from the two retrievers is discarded; the reranker re-ranks from scratch.

**Step 4: Cross-encoder rerank (line 180)**
```python
results = self._sentence_transformer_rerank(query, unique_texts, top_k)
```
Quality step — explained below.

Returns the joined text of top-k results.

### `_sentence_transformer_rerank()` (line 145)

```python
pairs = [[query, doc] for doc in documents]
scores = self.reranker.predict(pairs)
ranked_indices = np.argsort(scores)[::-1][:top_k]
```

Default reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Crucial difference between **bi-encoder** (embedding model) and **cross-encoder** (reranker):

| Bi-encoder (embedding model) | Cross-encoder (reranker) |
|---|---|
| Encodes query and doc **separately**, then compares vectors | Encodes `[query, doc]` **together** as one input |
| Fast — can pre-compute doc vectors offline | Slow — must run per `(query, doc)` pair |
| Lower accuracy | Higher accuracy |
| Used for **initial retrieval** over millions of docs | Used for **reranking top-N** (10–50 docs) |

The cross-encoder reads query + doc in one forward pass, so it can model token-level interactions (*"does this query word match that doc word?"*). Embedding models can't — they've already collapsed the doc into a fixed vector.

**Cost:** cross-encoding 20 candidates with a small model takes ~100ms on CPU. Worth the latency for the quality bump. ~80MB model loaded once at `SearchEngine.__init__`.

### `_reciprocal_rank_fusion()` (line 153) — UNUSED

```python
def _reciprocal_rank_fusion(self, result_lists):
    scores, all_results = {}, {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            item_id = item.get("id", str(hash(item["text"])))
            scores[item_id] = scores.get(item_id, 0) + 1.0 / (self.reciprocal_rank_k + rank)
            all_results[item_id] = item
    ...
```

Reciprocal Rank Fusion — the standard way to combine rankings from multiple retrievers. Each doc's score = `Σ 1 / (k + rank_in_list_i)` across all lists, where `k = 60` by default (from `config.yaml`).

- A doc at rank 1 in BM25 gets `1/(60+1) = 0.0164`.
- A doc at rank 1 in vector search also gets `0.0164`.
- A doc that's top-3 in **both** gets `0.0317` — rewarded for consensus.

**But this method is never called.** `hybrid_search` uses set-union + cross-encoder rerank instead. Either dead code or a future toggle. To A/B-test, swap `_sentence_transformer_rerank(query, unique_texts, ...)` for `_reciprocal_rank_fusion([bm25_results, vector_results])`.

### Convenience methods

- `vector_search()` — embedding-only, no BM25, no rerank. For ablation testing.
- `bm25_search()` — BM25-only, no vector, no rerank.

Useful for debugging *"is BM25 or vector responsible for this bad answer?"*. The `AgenticRAG` class only calls `hybrid_search`.

---

## 5. The retrieval pipeline end-to-end

```
User query: "How do I tune shared_buffers?"
        │
        ▼
embed_query([query]) → 384-dim vector
        │
        ├─► ThreadPool ───┐
        │                 │
        ▼                 ▼
BM25.search()       Chroma.search()
  → top 10            → top 10
  texts/scores        texts/scores
        │                 │
        └────────┬────────┘
                 ▼
       set(bm25 + chroma) → ~15-20 unique texts
                 │
                 ▼
       CrossEncoder.predict([query, doc]) × 20
                 │
                 ▼
       argsort → top_k texts
                 │
                 ▼
       "\n\n".join(results)
                 │
                 ▼
       returned to AgenticRAG._perform_search
```

---

## 6. Things to flag

1. **RRF defined but unused** (lines 153–162). Either dead code or a feature flag toggle. The reranker is doing the actual fusion job instead.
2. **No metadata-filtered search exposed.** Chroma supports `collection.query(..., where={...})` but it's not surfaced here. Fine for the single-PDF use case; would matter in multi-doc setups.
3. **BM25 tokenization is whitespace-split-lowercase.** Tokens like `pg_stat_statements` stay intact (good for technical docs). But `connection-pool` and `connection pool` won't match each other. Acceptable tradeoff for English technical content.
4. **`top_k=10` hard-coded** in `agentic_rag.py` line 42 → `hybrid_search(user_query, top_k=10)`. Tunable but not exposed via config.
5. **CrossEncoder loads ~80MB model on instantiation.** Loaded once at `SearchEngine.__init__`; lives in memory for the process lifetime. ~200ms cold-start, fast after.
6. **`embedding_generator` lives in `embed.py`** — not covered here. That's where the embedding model loads — relevant if you swap in the fine-tuned MPNet model.

---

## 7. Cross-reference back to the lecture

| Lecture (Day 1 PDF) | Implementation in this file |
|---|---|
| Slide 3 — `vectorstore.pkl` + `bm25.pkl` | `ChromaVectorStore` (`chroma/` folder) + `BM25Store` (`bm25.pkl`) |
| Slide 4 — BM25 | `BM25Store` |
| Slide 4 — Rerank | `_sentence_transformer_rerank` + CrossEncoder |
| Slide 5 — Embedding model + cosine similarity | `embedding_generator` + Chroma's `hnsw:space=cosine` |
| Slide 4 — Chunking strategies | Not here — that's in `document_processor.py` |
| Slide 4 — Query expansion / rewrite | Not here — that's in `agentic_rag.py` (`_rewrite_question`) |
