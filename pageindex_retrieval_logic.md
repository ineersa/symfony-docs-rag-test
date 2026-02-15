# PageIndex Retrieval Logic

This document describes the logic used in `pageindex_common.py` (and wrapped by `pageindex_retrieve.py`) to retrieve relevant documentation excerpts using the PageIndex tree structure.

## Overview

The retrieval system operates in two primary modes:

1.  **Lexical-Only**: Fast, keyword-based global search.
2.  **LLM-Guided**: Tree traversal and re-ranking using an LLM, combined with lexical signals.

## Core Component: Lexical Scoring

A strictly deterministic `lexical_score(query, text)` function is used throughout to filter and rank candidates before sending them to the LLM (to save tokens) or as a fallback.

- **Tokenization**: Splits text into tokens, lowercases, removes stopwords, and applies light stemming (e.g., handling plurals).
- **Scoring Formula**:
  ```python
  # weighted mainly for recall (0.8) vs precision (0.2)
  score = (0.8 * recall) + (0.2 * precision)
  ```
  This favors documents that contain most of the query terms, even if the document is long.

## Retrieval Modes

### 1. Lexical-Only Mode

Used when `--no-llm` is passed.

1.  **Flatten**: Considers all content-bearing nodes (`section` and `top` kinds) in the index.
2.  **Score**: Calculates `lexical_score` for every node against the query.
3.  **Rank**: Sorts by score descending.
4.  **Return**: Returns the top K results.

### 2. LLM-Guided Tree Search

Used by default. It combines hierarchical exploration (Beam Search) with a global lexical safety net.

#### Phase A: Tree Traversal (Beam Search)

Starting from the synthetic root (`pageindex_root`):

1.  **Expand**: Get children of current frontier nodes.
2.  **Filter**:
    - **Standard Mode**: The LLM selects the `beam_width` most relevant children given the query `_llm_pick_children`.
    - **LLM-Final-Only Mode** (`--llm-final-only`): Children are selected purely by `lexical_score` to save LLM calls.
3.  **Loop**: Repeat for `max_depth` iterations or until no children remain.
4.  **Collect**: All nodes visited during traversal are added to a `visited` set.

#### Phase B: Candidate Pool Construction

To strictly relying on tree traversal can miss relevant nodes if a high-level parent is misjudged. To mitigate this:

1.  **Global Lexical Search**: The top 80 nodes from the _entire_ index (scored lexically) are retrieved.
2.  **Merge**: The `visited` nodes from the tree traversal are merged with these top 80 global matches.
3.  **Deduplicate**: Creates a unified `candidates` pool.

#### Phase C: Final LLM Reranking

1.  **Pre-rank**: The candidate pool is sorted lexically to pick the top `final_candidate_limit` (default 24) items.
2.  **LLM Rank**: The LLM is asked to rank these final candidates based on relevance to the query (`_llm_rank_candidates`).
    - The LLM is given summaries and text excerpts.
    - It returns a JSON list of IDs in best-first order.
3.  **Fallback/Safety**:
    - If the LLM omits relevant items, high-scoring lexical matches (score >= 0.35) are appended to fill `top_k`.
    - If the LLM fails, it falls back to lexical ranking.

## Logic Diagram

```text
      [ Start ]
          |
          v
    < LLM Enabled? >
    |              |
    | No           | Yes
    |              v
    |      [ Start at Root ]
    |              |
    |              v
    |      [ Expand Children ] <---------------+
    |              |                           |
    |              v                           |
    |      < Pick Children >                   |
    |     (LLM or Lexical only)                |
    |              |                           |
    |              v                           |
    |      [ Update Frontier ]                 |
    |              |                           |
    |              v                           |
    |      < Max Depth? > -------- No ---------+
    |              |
    |              | Yes
    |              v
    |      [ Collect Visited ]
    |              |
    |              v
    |      [ Merge Candidates ] <--- [ Top 80 Global Lexical ]
    |              |
    |              v
    |      [ Pre-rank by Lexical ]
    |              |
    |              v
    |      [ Take Top N Candidates ]
    |              |
    |              v
    |      [ LLM Final Rerank ]
    |              |
    |              v
    |      [ Fill/Fallback (Lexical > 0.35) ]
    |              |
    v              v
    [ Return Top K Hits ]
```
