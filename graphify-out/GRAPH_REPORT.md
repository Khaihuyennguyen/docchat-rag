# Graph Report - docchat-rag  (2026-11-13)

## Corpus Check
- 23 files · ~4,346 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 1 file(s) not represented in the graph (top: (none) 1)

## Summary
- 157 nodes · 263 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0e246f8c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Chunk
- chunking.py
- llm.py
- answer.py
- embeddings.py
- 4. Top 12 RAG Technical Interview Questions & Model Answers
- BM25
- test_retrieve.py
- Nimbus Coffee Co. - Expense Policy
- Nimbus Coffee Co. - Leave Policy
- Nimbus Coffee Co. - IT Security Policy
- docchat

## God Nodes (most connected - your core abstractions)
1. `Chunk` - 18 edges
2. `VectorStore` - 13 edges
3. `BM25` - 10 edges
4. `HashEmbedder` - 9 edges
5. `HybridRetriever` - 9 edges
6. `answer_question()` - 8 edges
7. `split_text()` - 7 edges
8. `chunk_pages()` - 7 edges
9. `Page` - 7 edges
10. `make_store()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `test_cited_numbers_are_unique_and_sorted()` --calls--> `cited_numbers()`  [EXTRACTED]
  tests/test_answer.py → docchat/answer.py
- `test_tokenize_lowercases_and_drops_punctuation()` --calls--> `tokenize()`  [EXTRACTED]
  tests/test_bm25.py → docchat/bm25.py
- `test_no_match_returns_empty()` --calls--> `BM25`  [EXTRACTED]
  tests/test_bm25.py → docchat/bm25.py
- `test_rare_exact_word_wins()` --calls--> `BM25`  [EXTRACTED]
  tests/test_bm25.py → docchat/bm25.py
- `test_prompt_numbers_every_source()` --calls--> `Chunk`  [EXTRACTED]
  tests/test_answer.py → docchat/chunking.py

## Import Cycles
- None detected.

## Communities (14 total, 3 thin omitted)

### Community 0 - "Chunk"
Cohesion: 0.16
Nodes (8): Chunk, Embedder, HybridRetriever, VectorStore, make_store(), test_search_on_empty_store(), test_search_returns_most_similar_chunk_first(), test_upsert_does_not_duplicate()

### Community 1 - "chunking.py"
Cohesion: 0.19
Nodes (12): chunk_pages(), split_text(), load_file(), load_folder(), Page, test_chunk_ids_are_stable(), test_chunks_overlap(), test_chunks_respect_max_size() (+4 more)

### Community 2 - "llm.py"
Cohesion: 0.12
Nodes (4): Settings, ClaudeLLM, FakeLLM, MissingAPIKey

### Community 3 - "answer.py"
Cohesion: 0.19
Nodes (7): Answer, answer_question(), build_prompt(), cited_numbers(), LLM, test_cited_numbers_are_unique_and_sorted(), test_prompt_numbers_every_source()

### Community 4 - "embeddings.py"
Cohesion: 0.18
Nodes (5): HashEmbedder, LocalEmbedder, test_same_text_same_vector(), test_shared_words_mean_higher_similarity(), test_vectors_have_unit_length()

### Community 5 - "4. Top 12 RAG Technical Interview Questions & Model Answers"
Cohesion: 0.14
Nodes (13): 1. End-to-End System Architecture, 2. The 3 Levels of RAG (Industry Classification), 3. Deep Dive: RRF vs. Cross-Encoder Reranker, 4. Top 12 RAG Technical Interview Questions & Model Answers, DocChat RAG: Complete Architecture, Visual Flow & Interview Guide, Embeddings & Vector Stores, Grounding, Debugging & Evals, Hybrid Retrieval & Fusion (+5 more)

### Community 6 - "BM25"
Cohesion: 0.26
Nodes (5): BM25, tokenize(), test_no_match_returns_empty(), test_rare_exact_word_wins(), test_tokenize_lowercases_and_drops_punctuation()

### Community 7 - "test_retrieve.py"
Cohesion: 0.28
Nodes (4): rrf(), test_every_mode_returns_at_most_k(), test_rrf_rewards_items_ranked_high_in_both_lists(), test_unknown_mode_raises()

### Community 8 - "Nimbus Coffee Co. - Expense Policy"
Cohesion: 0.33
Nodes (5): Home office, Hotels and flights, Meals while travelling, Nimbus Coffee Co. - Expense Policy, Submitting expenses

### Community 9 - "Nimbus Coffee Co. - Leave Policy"
Cohesion: 0.33
Nodes (5): Bereavement, Nimbus Coffee Co. - Leave Policy, Parental leave, Sick leave, Vacation

### Community 10 - "Nimbus Coffee Co. - IT Security Policy"
Cohesion: 0.33
Nodes (5): Lost devices, Nimbus Coffee Co. - IT Security Policy, Passwords, Phishing, Two-factor authentication

## Knowledge Gaps
- **1 isolated node(s):** `docchat`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 63 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Chunk` connect `Chunk` to `chunking.py`, `answer.py`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Chunk` (e.g. with `Answer` and `build_prompt()`) actually correct?**
  _`Chunk` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `docchat` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `llm.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Why does `HashEmbedder` connect `embeddings.py` to `Chunk`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `VectorStore` (e.g. with `HybridRetriever` and `Chunk`) actually correct?**
  _`VectorStore` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `4. Top 12 RAG Technical Interview Questions & Model Answers` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._