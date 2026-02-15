# PageIndex-Style RAG Plan (Separate System)

This plan adds a second, independent RAG system inspired by PageIndex's tree retrieval,
without changing the existing vector-based pipeline.

## Goals

- Keep the current pipeline untouched (`chunk.py`, `embed.py`, `retrieve.py`, benchmark flow).
- Add a parallel tree-based retrieval system for hierarchical, reasoning-style navigation.
- Reuse the existing Symfony RST corpus and line-aware metadata for benchmark compatibility.

## New Components

1. `pageindex_build.py`
   - Build a hierarchical index from `symfony-docs/`.
   - Parse document-level structure from RST (`.. toctree::` where possible).
   - Parse section/subsection hierarchy from docutils doctrees.
   - Produce node records with:
     - `node_id`, `parent_id`, `children`
     - `source`, `title`, `breadcrumb`
     - `line_start`, `line_end`
     - `text` and compact `summary`

2. `pageindex_retrieve.py`
   - Query-time tree traversal guided by an LLM.
   - Vectorless retrieval over node summaries/titles.
   - Return top-k leaf-like evidence nodes with citations and confidence-style scores.
   - Support one-shot and REPL usage.

3. `pageindex_api.py`
   - Lightweight HTTP endpoint: `POST /retrieve`.
   - Input: `{"query": "...", "top_k": 5}`
   - Output: `{"hits": [{"id", "source", "line_start", "line_end", "distance"}]}`
   - Compatible with existing `benchmark_run.py --predictor api`.

## Data Layout

- `data/pageindex/tree.json` — full hierarchical tree.
- `data/pageindex/nodes.jsonl` — flattened node index for retrieval.
- `data/pageindex/cache/` — optional query decision cache.

## Retrieval Strategy

1. Start from roots.
2. Ask LLM to pick most relevant children for current frontier.
3. Recurse until depth/node budget is reached.
4. Score final candidates with an additional relevance pass.
5. Return top-k with file + line citations.

## Reliability & Fallbacks

- Strict JSON parsing for LLM responses.
- If LLM output is malformed:
  - retry with tighter prompt,
  - then fallback to lexical scoring over node summaries.
- Cache traversal decisions per query hash to reduce latency/cost.

## Rollout Phases

1. Build indexer + outputs.
2. CLI retriever (one-shot + REPL).
3. API adapter for benchmark compatibility.
4. README docs for separate workflow.
5. Baseline benchmark run and compare with vector retriever.

## Non-Goals

- No changes to existing chunk/embedding/retrieval scripts.
- No dependency on ChromaDB for the tree system.
- No replacement of current benchmark logic.
