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
