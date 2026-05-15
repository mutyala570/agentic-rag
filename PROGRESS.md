# Project Progress — Agentic RAG

**Quick read when coming back.** Tracks what's done, what's remaining, and where each piece lives. Update this file at the end of each session.

**Last updated:** 2026-05-15

---

## Project status snapshot

| Item | State |
|---|---|
| Project location | `/Users/mutyalaqwipo/qwipo/agentic-rag/` |
| GitHub repo | `https://github.com/mutyala570/agentic-rag` (private) |
| Initial commit pushed | ✅ |
| `.env` (Groq API key etc.) | ✅ Locally present, gitignored |
| `.gitignore` (protects `.env`, `data/`, `vectorstores/`, `.venv/`) | ✅ Locally present, **not yet committed** |
| PDF for testing | ✅ `pdfs/PostgreSQLPerformanceTuning0424pdf.pdf` (130 pages, text-based) |
| `uv` installed | ❌ Pending — run `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `uv sync` (Python 3.12 + deps) | ❌ Pending |
| `document_processor.py` run (creates `data/`) | ❌ Pending |
| `ingest.py` run (creates `vectorstores/chroma/` + `bm25.pkl`) | ❌ Pending |
| First query via `agentic_rag.py` | ❌ Pending |
| Fine-tuning run via `finetune_embedding.py` | ❌ Pending (optional later) |

---

## Files walked through with Claude

| File | Status | Reference doc |
|---|---|---|
| `agentic_rag.py` | ✅ Walkthrough done | (in chat — not yet in md) |
| `search_utils.py` | ✅ Walkthrough done | `docs/search_utils_explained.md` |
| `finetune_embedding.py` | ✅ Added + walkthrough | `docs/finetune_embedding_notes.md` |
| `embed.py` | ❌ Pending | — |
| `ingest.py` | ❌ Pending | — |
| `document_processor.py` | ❌ Pending | — |
| `simple_rag.py` | ❌ Pending (optional — for compare) | — |
| `app.py` | ❌ Pending (gradio runner) | — |
| `config_loader.py` | ❌ Pending (utility) | — |

---

## What's remaining (priority order)

### 1. Walk through 3 files (resume from here next session)

- [ ] **`embed.py`** — the embedding model wrapper. Short file. Lets us see exactly where you'd plug in the fine-tuned MPNet model.
- [ ] **`ingest.py`** — the offline build. How the two indexes get populated from your PDF.
- [ ] **`document_processor.py`** — the chunking logic (Slide 4: *chunking strategies*).

### 2. Run the pipeline end-to-end

```bash
cd /Users/mutyalaqwipo/qwipo/agentic-rag
curl -LsSf https://astral.sh/uv/install.sh | sh        # install uv
uv sync                                                 # installs Python 3.12 + 20 deps
.venv/bin/python document_processor.py                  # creates data/ folder
.venv/bin/python ingest.py                              # creates vectorstores/ folder
.venv/bin/python agentic_rag.py                         # runs a query
```

### 3. Commit + push the `.gitignore`

After verifying `.env` is excluded:
```bash
git -C /Users/mutyalaqwipo/qwipo/agentic-rag add .gitignore
git -C /Users/mutyalaqwipo/qwipo/agentic-rag commit -m "Add .gitignore"
git -C /Users/mutyalaqwipo/qwipo/agentic-rag push
```

### 4. (Optional) Run fine-tuning + swap models

After base pipeline works, see `docs/finetune_embedding_notes.md` to:
- Run `finetune_embedding.py` (10–20 min on CPU).
- Compare before/after accuracy numbers.
- If gain is meaningful, swap `config.yaml`'s embedding model and re-ingest.

---

## Theory notes (where to look for the "why" behind the code)

| Topic | Lives at |
|---|---|
| Day 1 lecture summary | `/Users/mutyalaqwipo/qwipo/ai-learning-v1/week-five/day-one-agentic-rag.md` |
| LLM Wiki vs Agentic RAG decision | discussed in chat — `ai-learning-v1` = wiki, `agentic-rag/` = RAG learning |
| Fine-tuning operating notes | `docs/finetune_embedding_notes.md` (in this folder) |
| `search_utils.py` walkthrough | `docs/search_utils_explained.md` (in this folder) |

---

## Quick reference — when Claude opens this folder next time

If the user asks *"what's remaining?"* or *"where did we leave off?"*, the answer is:

> *"Three files to walk through (`embed.py`, `ingest.py`, `document_processor.py`), then run the pipeline end-to-end. `uv` is not installed yet, so the pipeline can't run. Suggested order: walk through the three files first (each 10–15 min), then `curl` install uv → `uv sync` → run the four pipeline commands."*
