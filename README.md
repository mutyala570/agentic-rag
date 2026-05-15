# Agentic RAG Core

A hybrid RAG system combining BM25 keyword search, Chroma vector search, and a LangGraph agentic pipeline that grades retrieved documents and rewrites queries when needed.

## Setup

```bash
uv sync
```

## Usage

### 1. Add your PDFs

Drop PDF files into the `pdfs/` directory:

```
pdfs/
  your_paper.pdf
  another_document.pdf
```

### 2. Process documents

Converts PDFs to markdown and chunks them:

```bash
.venv/bin/python document_processor.py
```

### 3. Ingest into vector stores

Generates embeddings and builds the Chroma + BM25 indexes:

```bash
.venv/bin/python ingest.py
```

### 4. Run a query

**Agentic RAG** (grades docs, rewrites query if needed):
```bash
.venv/bin/python agentic_rag.py
```

**Simple RAG** (single-pass search + generate):
```bash
.venv/bin/python simple_rag.py
```

Edit the `query` variable at the bottom of either file to change the question.

## Configuration

All settings are in `config.yaml` — chunk size, models, top-k, prompts, and file paths.

## Project layout

```
agentic-rag/
├── README.md                          ← this file
├── PROGRESS.md                        ← session-to-session status tracker
├── config.yaml                        ← paths, models, prompts, search/RAG settings
├── pyproject.toml                     ← deps (run `uv sync`)
├── .env                               ← API keys (gitignored)
├── .gitignore
│
├── pdfs/                              ← drop source PDFs here
├── data/                              ← auto-generated: markdown / chunks / processed
├── vectorstores/                      ← auto-generated: chroma/ + bm25.pkl
├── models/                            ← auto-generated: fine-tuned weights
│
├── docs/                              ← reference & walkthrough docs
│   ├── search_utils_explained.md
│   └── finetune_embedding_notes.md
│
│  --- Python source files ---
├── config_loader.py                   ← loads config.yaml
├── document_processor.py              ← PDFs → markdown → chunks
├── embed.py                           ← embedding model wrapper
├── ingest.py                          ← builds Chroma + BM25 indexes
├── search_utils.py                    ← hybrid search (BM25 + vector + rerank)
├── simple_rag.py                      ← baseline single-pass RAG
├── agentic_rag.py                     ← LangGraph agent with grade + rewrite loop
├── finetune_embedding.py              ← triplet-loss fine-tuning script
└── app.py                             ← Gradio UI runner
```

## Documentation

Deeper walkthroughs of specific files live in `docs/`:

- **`docs/search_utils_explained.md`** — full retrieval pipeline (Chroma, BM25, RRF, cross-encoder rerank).
- **`docs/finetune_embedding_notes.md`** — operating notes for `finetune_embedding.py` (CPU vs GPU, what to commit, how to wire the fine-tuned model into the pipeline).
