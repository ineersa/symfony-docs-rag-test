# Hybrid Retrieval System Documentation

## Overview

The Hybrid Retrieval system in `simple-rag` combines the strengths of dense vector retrieval with the structural awareness of the PageIndex tree headers, enhanced by graph-based expansion, Reciprocal Rank Fusion (RRF), and a final LLM-based reranking step.

This approach addresses common RAG failure modes where:

1.  Vector search retrieves relevant snippets but misses the broader context (parent sections).
2.  Keywords are important but semantically "far" in vector space (fixed by BM25 scoring).
3.  The "best" answer is adjacent to the vector hit (fixed by neighbor expansion).

## Logic Flow

The retrieval process follows a multi-stage pipeline:

1.  **Vector Shortlist (with optional HyDE)**:
    - Optional HyDE generates 2-3 synthetic Symfony-style summaries for the query.
    - The original query and HyDE variants are embedded and searched against the local ChromaDB vector index.
    - Top `vector_top_n` chunks are retrieved.
    - Chunks are mapped to **PageIndex Nodes** (sections/topics) via line number overlap or breadcrumb matching.
    - Initial scores are assigned based on vector distance and rank.

2.  **Neighbor Expansion**:
    - Top scoring nodes from the vector step act as strict "seeds".
    - The graph is traversed to include related nodes (parents, children, siblings) up to `neighbor_depth`.
    - Scores decay as distance from the seed increases (e.g., Parent: 0.75x, Child: 0.7x, Sibling: 0.6x).
    - Top matching _files_ are also injected to ensure coverage of highly relevant documents even if specific section chunks were missed.

3.  **Reciprocal Rank Fusion (RRF)**:
    - A **BM25 Score** is calculated globally over node text (title + summary + text).
    * The **Vector-Expanded rank list** and **BM25 rank list** are combined using RRF (`k=60`).
    * This ensures that a node must ideally be both semantically similar (vector) and keyword-relevant (BM25) to rank highly.

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
| `hyde_variants`        | `0`     | Number of synthetic HyDE query-doc variants.              | **Dense Recall Boost.** `0` keeps baseline behavior. `2-3` usually helps when user wording differs from docs phrasing.                                                       |
| `hyde_variant_weight`  | `0.7`   | Score weight for each HyDE variant dense hit.             | **Stability vs. Diversity.** Lower than `1.0` keeps original query dominant while still using HyDE for recall.                                                               |
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

HyDE generation controls:

| Parameter          | Default | Description                                 | Impact on RAG                                                                                                      |
| :----------------- | :------ | :------------------------------------------ | :----------------------------------------------------------------------------------------------------------------- |
| `hyde_temperature` | `0.3`   | Sampling temperature for HyDE generation.   | **Variant Diversity.** Slightly higher values create broader hypotheses; too high can make variants noisy.       |
| `hyde_max_chars`   | `420`   | Max length of each generated HyDE document. | **Prompt Budget.** Caps synthetic text before embedding to prevent long noisy variants from dominating retrieval. |

### CLI Usage Example

```bash
# High-recall mode: wider vector search, more text for LLM
uv run hybrid_retrieve.py \
  --vector-top-n 60 \
  --candidate-cap 50 \
  --final-text-chars 2000 \
  "how to configure security"

# HyDE mode: generate 2 synthetic summaries per query
uv run hybrid_retrieve.py \
  --hyde-variants 2 \
  --hyde-variant-weight 0.7 \
  "how to configure security firewalls"
```
