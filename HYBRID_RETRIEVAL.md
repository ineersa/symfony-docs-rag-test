# Hybrid Retrieval System Documentation

## Overview

The Hybrid Retrieval system in `simple-rag` combines the strengths of dense vector retrieval with the structural awareness of the PageIndex tree headers, enhanced by graph-based expansion, Reciprocal Rank Fusion (RRF), and a final LLM-based reranking step.

This approach addresses common RAG failure modes where:

1.  Vector search retrieves relevant snippets but misses the broader context (parent sections).
2.  Keywords are important but semantically "far" in vector space (fixed by lexical scoring).
3.  The "best" answer is adjacent to the vector hit (fixed by neighbor expansion).

## Logic Flow

The retrieval process follows a multi-stage pipeline:

1.  **Vector Shortlist**:
    - The query is embedded and searched against the local ChromaDB vector index.
    - Top `vector_top_n` chunks are retrieved.
    - Chunks are mapped to **PageIndex Nodes** (sections/topics) via line number overlap or breadcrumb matching.
    - Initial scores are assigned based on vector distance and rank.

2.  **Neighbor Expansion**:
    - Top scoring nodes from the vector step act as strict "seeds".
    - The graph is traversed to include related nodes (parents, children, siblings) up to `neighbor_depth`.
    - Scores decay as distance from the seed increases (e.g., Parent: 0.75x, Child: 0.7x, Sibling: 0.6x).
    - Top matching _files_ are also injected to ensure coverage of highly relevant documents even if specific section chunks were missed.

3.  **Reciprocal Rank Fusion (RRF)**:
    - A **Lexical Score** is calculated for all candidates using token overlap (recall-weighted Jaccard) on title + summary + text.
    - It applies light stemming and stopword removal to ensure robust keyword matching.
    * The **Vector-Expanded Score** and **Lexical Score** are combined using RRF (`k=60`).
    * This ensures that a node must ideally be both semantically similar (vector) and keyword-relevant (lexical) to rank highly.

4.  **LLM Reranking (Optional)**:
    - The top `candidate_cap` nodes after RRF are sent to a local LLM.
    - The LLM is prompted to pick the best `top_k` nodes that "most directly answer the question".
    - The final list is returned to the user.

## Logic Diagram

```text
                                  User Query
                                       │
                                       ▼
                             ┌───────────────────┐
                             │  Vector Search    │ (ChromaDB)
                             │   (Top N Chunks)  │
                             └─────────┬─────────┘
                                       │
                                       ▼
                             ┌───────────────────┐
                             │  Map to PageIndex │ (Chunk -> Node)
                             │      Nodes        │
                             └─────────┬─────────┘
                                       │
                                       ▼
    ┌───────────────────────┐    ┌─────┴────────────────┐
    │  Neighbor Expansion   │◄───│  Initial Node Scores │
    │ (Parents, Children)   │    └──────────────────────┘
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────┐    ┌──────────────────────┐
    │  Vector/Graph Score   │    │    Lexical Score     │
    │      (Rank List A)    │    │    (Rank List B)     │
    └──────────┬────────────┘    └──────────┬───────────┘
               │                            │
               └──────────────┬─────────────┘
                              │
                              ▼
                     ┌───────────────────┐
                     │        RRF        │ (Reliprocal Rank Fusion)
                     │    Combination    │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │   Top Candidates  │ (Cap: 30)
                     └─────────┬─────────┘
                               │
                    (If LLM Rerank Enabled)
                               │
                               ▼
                     ┌───────────────────┐
                     │    LLM Rerank     │ (select top K)
                     └─────────┬─────────┘
                               │
                               ▼
                        Final Top K Hits
```

## Configurable Parameters

The `HybridRetriever` can be tuned via several parameters. Here is how they affect the RAG pipeline:

### Search & Expansion

| Parameter              | Default | Description                                               | Impact on RAG                                                                                                                                                              |
| :--------------------- | :------ | :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vector_top_n`         | `40`    | Number of chunks to fetch from ChromaDB.                  | **Recall vs. Noise.** Higher values increase the chance of finding the right "seed" but introduce more irrelevant noise nodes.                                             |
| `neighbor_depth`       | `1`     | Levels of graph traversal (up/down/sideways).             | **Context Discovery.** Setting this >0 allows the system to find "adjacent" answers (e.g., finding the "Configuration" section when the vector hit was on "Installation"). |
| `siblings_per_node`    | `2`     | Max sibling nodes to include.                             | **Width of Context.** Controls how "wide" the expansion is within a single section.                                                                                        |
| `candidate_file_roots` | `5`     | Number of whole files to inject based on aggregate score. | **Document Level Recall.** Ensures that if many chunks from a file match, the file's root node is considered, helpful for "Overview of X" queries.                         |

### Ranking & Filtering

| Parameter             | Default | Description                               | Impact on RAG                                                                                                                                          |
| :-------------------- | :------ | :---------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `candidate_cap`       | `30`    | Max nodes to keep for final RRF/LLM step. | **Performance vs. Quality.** Limits the context window usage for the LLM. Too low = missing good answers; Too high = slower, higher cost.              |
| `use_llm`             | `True`  | Enable/Disable LLM reranking.             | **Accuracy vs. Speed.** LLM reranking significantly improves precision by "reading" the candidates, but adds latency.                                  |
| `final_summary_chars` | `420`   | Length of node summary sent to LLM.       | **Context Window.** Controls how much "summary" metadata the LLM sees.                                                                                 |
| `final_text_chars`    | `1500`  | Length of raw text body sent to LLM.      | **Context Window.** Controls how much actual content the LLM reads. Increasing this allows the LLM to judge relevance better but consumes more tokens. |

### CLI Usage Example

```bash
# High-recall mode: wider vector search, more text for LLM
uv run hybrid_retrieve.py \
  --vector-top-n 60 \
  --candidate-cap 50 \
  --final-text-chars 2000 \
  "how to configure security"
```
