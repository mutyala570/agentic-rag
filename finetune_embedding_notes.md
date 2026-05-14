# Fine-Tune Embedding — Operating Notes

Reference doc for `finetune_embedding.py`. Read this before running, or when you come back to the file later and forget what to expect.

## Contents

1. [What this file does](#1-what-this-file-does)
2. [How to run](#2-how-to-run)
3. [A few things to know before running](#3-a-few-things-to-know-before-running)
4. [Wiring the fine-tuned model into the RAG pipeline](#4-wiring-the-fine-tuned-model-into-the-rag-pipeline)
5. [What to commit to GitHub](#5-what-to-commit-to-github)

---

## 1. What this file does

Implements **triplet-loss fine-tuning** for an embedding model — the technique from the Week 5 Day 1 Agentic RAG lecture (Slide 6).

It trains `microsoft/mpnet-base` on a subset of the `sentence-transformers/all-nli` triplets dataset, then evaluates **before vs after** accuracy on a held-out test set. The intended takeaway is the score-improvement pattern from the lecture (e.g., 53% → 85%).

| Method | Role |
|---|---|
| `setup_model()` | Loads base model `microsoft/mpnet-base` |
| `load_dataset()` | Pulls `sentence-transformers/all-nli` triplets from HuggingFace |
| `create_*_dataset()` | Slices 1000 train / 200 test / 200 eval |
| `setup_loss()` | `MultipleNegativesRankingLoss` — modern in-batch triplet loss |
| `setup_training_args()` | Epochs, batch size, eval schedule, output dir |
| `evaluate_cosine_accuracy()` | Counts how often `‖A−P‖ < ‖A−N‖` on test set |
| `train_model()` | Runs the trainer |
| `__main__` | eval → train → eval → print before/after accuracy |

---

## 2. How to run

```bash
cd /Users/mutyalaqwipo/qwipo/agentic-rag
uv sync                                  # picks up datasets, scikit-learn, torch, accelerate
.venv/bin/python finetune_embedding.py
```

Expected output (numbers will vary by run):

```
Accuracy before fine-tuning: 0.5300
... training logs ...
Accuracy after fine-tuning:  0.8500
```

---

## 3. A few things to know before running

**CPU vs GPU.** On Mac (CPU / MPS), training **1 epoch on 1000 triplets takes ~10–20 minutes**. On a CUDA GPU it's seconds. If too slow, lower `range(1000)` to `range(200)` in `create_train_dataset()` — quicker run, smaller gain.

**First run downloads ~700MB.** The `microsoft/mpnet-base` model + the all-nli dataset. One-time cost — cached under `~/.cache/huggingface/`. Subsequent runs are instant.

**Output goes to `models/mpnet-base-all-nli-triplet/`.** Folder is auto-created. After training, this directory **is** the fine-tuned model — you can load it with `SentenceTransformer("models/mpnet-base-all-nli-triplet")`.

**Two accuracy numbers get printed.** `Accuracy before fine-tuning` and `Accuracy after fine-tuning`. The lecture's whole point: bigger gap = fine-tuning helped. If the gap is small, your base model already covered the dataset well.

**The `all-nli` dataset is NOT domain-specific** to PostgreSQL or your real use case. It's a general English NLI dataset — used here purely to demonstrate the **mechanics** of triplet-loss fine-tuning. For real production gain on the PostgreSQL corpus, you'd build your own `(anchor, positive, negative)` triplets from the PDF and train on those.

**Fine-tuning is the "last resort" from the lecture.** Slide 4 listed it last in the advanced-RAG sequence. Always try query rewrite, better chunking, hybrid search, and reranking first — they're cheaper. Fine-tune only when the embedding space is genuinely mismatched to your domain.

---

## 4. Wiring the fine-tuned model into the RAG pipeline

After training succeeds, you can swap the production embedding model. Edit `config.yaml`:

```yaml
models:
  embedding:
    name: "models/mpnet-base-all-nli-triplet"   # local path — was "sentence-transformers/all-MiniLM-L6-v2"
```

Then **delete and rebuild** the vector stores so they use the new embeddings:

```bash
rm -rf vectorstores/ data/vectors/
.venv/bin/python ingest.py
```

The Chroma index gets re-embedded with the fine-tuned model. BM25 is unaffected (it's keyword-based, not embedding-based).

**Don't do this swap blindly.** First confirm `Accuracy after > Accuracy before` by a meaningful margin (e.g., +5 percentage points). If the gain is tiny, the swap costs you re-ingestion time for marginal benefit.

---

## 5. What to commit to GitHub

**Commit these:**
- `finetune_embedding.py`
- The new dependencies in `pyproject.toml` (`datasets`, `scikit-learn`, `torch`, `accelerate`)
- This `finetune_embedding_notes.md`

**Do NOT commit these:**
- `models/` — fine-tuned model weights are ~400MB+. Add to `.gitignore`:
  ```
  models/
  ```
- `~/.cache/huggingface/` — already outside the repo, but worth knowing it exists if disk space matters.

If you want to share the fine-tuned model with teammates, push it to the HuggingFace Hub instead of GitHub.
