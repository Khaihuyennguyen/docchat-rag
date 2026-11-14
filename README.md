# DocChat - chat with your documents, with citations

A retrieval-augmented generation (RAG) app. Drop PDFs or Markdown in a folder, ask questions, and get answers that cite the exact file and page.

**Highlights:**

- **Hybrid retrieval.** Vector search (local MiniLM embeddings in Chroma) plus a BM25 keyword index I wrote from scratch, fused with Reciprocal Rank Fusion.
- **Grounded answers.** The model may only use numbered sources, must cite them like `[1]`, says "I don't know" when the answer is missing, and treats documents as data, not instructions (basic prompt-injection defence). Made-up citation numbers are dropped.
- **Measured, not guessed.** `python -m docchat eval` scores retrieval (hit rate and MRR) for dense, keyword and hybrid modes on a labelled question set.
- **Idempotent ingestion.** Chunk ids are content hashes, so re-indexing never creates duplicates.
- **Tested without an API key.** 36 tests use a fake embedder and a fake LLM.

## Results (sample handbook, 12 labelled questions, top-1)

| Mode   | Hit rate | MRR  |
|--------|---------:|-----:|
| dense  | 1.00     | 1.00 |
| sparse | 0.83     | 0.83 |
| hybrid | 1.00     | 1.00 |

Keyword search misses paraphrases ("food budget" vs "meal allowance"); vectors catch them. Hybrid keeps keyword search's strength on rare exact terms such as product names.

## Architecture

```
files -> loaders -> chunking -> embeddings -> Chroma (vectors)
                                           -> BM25 (keywords)
question -> dense top-k + sparse top-k -> RRF fusion -> prompt with numbered sources -> Claude -> cited answer
```

## Run it

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # needed for answers, not for ingest/eval
python -m docchat ingest data
python -m docchat eval evals/questions.jsonl --k 1
python -m docchat ask "How many vacation days do I get?"
uvicorn docchat.api:app --reload        # open http://127.0.0.1:8000
pytest -q
```

## What I would do next

Re-ranking with a cross-encoder, per-user document permissions, streaming answers, and pgvector for production storage.
