# Simple RAG — Symfony Docs

Proof-of-concept retrieval playground over Symfony docs with three retrieval systems:

- vector RAG (`chunk.py` + `embed.py` + `retrieve.py`)
- PageIndex-style tree RAG (`pageindex_build.py` + `pageindex_retrieve.py` + `pageindex_api.py`)
- hybrid retrieval (`hybrid_retrieve.py` + `benchmark_run.py --predictor hybrid`)

## Project Structure

```
chunk.py            — RST-aware chunker for vector pipeline
embed.py            — Embeds chunks into local ChromaDB
retrieve.py         — Interactive semantic retrieval (vector pipeline)

pageindex_build.py   — Builds hierarchical tree index from RST docs
pageindex_retrieve.py — Interactive tree retrieval (LLM-guided or lexical fallback)
pageindex_api.py      — HTTP API for tree retrieval (benchmark-compatible)

benchmark_build.py   — Builds benchmark questions with gold file+line spans
benchmark_run.py     — Runs strict/relaxed hit@1 and hit@5 benchmark
hybrid_retrieve.py   — Hybrid retriever (vector shortlist + tree expansion + LLM rerank)

symfony-docs/       — Symfony documentation source (RST files)
data/               — Generated artifacts (vector + tree outputs)
```

## How to Run

### 1) Install

```bash
uv sync
```

### 2) Vector Pipeline

```bash
# chunk docs -> data/chunks.jsonl
uv run python chunk.py

# embed into ChromaDB (CodeRankEmbed endpoint)
uv run python embed.py

# query vector index
uv run python retrieve.py
```

### 3) PageIndex-Style Tree Pipeline (Separate)

```bash
# build tree index -> data/pageindex/tree.json + nodes.jsonl
uv run python pageindex_build.py

# optional: generate LLM summaries for section/top nodes at build time
uv run python pageindex_build.py --summary-mode llm --summary-base-url http://localhost:8052/v1 --summary-model local-model

# query tree index (LLM-guided)
uv run python pageindex_retrieve.py --base-url http://localhost:8052/v1 --model local-model

# optional: include larger body excerpts in LLM ranking payload
uv run python pageindex_retrieve.py --step-text-chars 1200 --final-text-chars 2000 "your query"

# optional: skip tree-step LLM calls and use LLM only for final selection
uv run python pageindex_retrieve.py --llm-final-only "your query"
```

### 4) Benchmark

```bash
# build questions
uv run python benchmark_build.py --generator heuristic

# run benchmark against vector pipeline (local Chroma predictor)
uv run python benchmark_run.py --mode both --predictor local

# run benchmark against local PageIndex predictor (no API server needed)
uv run python benchmark_run.py --mode both --predictor pageindex --pageindex-base-url http://localhost:8052/v1 --pageindex-model local-model

# run benchmark against local Hybrid predictor (vector + PageIndex rerank)
uv run python benchmark_run.py --mode both --predictor hybrid --hybrid-base-url http://localhost:8059/v1 --hybrid-model local-model

# hybrid ablation: disable final LLM rerank
uv run python benchmark_run.py --mode both --predictor hybrid --hybrid-no-llm

# Note: Hybrid predictor now reuses --pageindex-final-summary-chars (default: 420)
# and --pageindex-final-text-chars (default: 1500) for its LLM reranking step.
# This allows configurable context windows for the reranker.

# run benchmark against PageIndex API predictor
uv run python pageindex_api.py --base-url http://localhost:8052/v1 --model local-model --port 8090
uv run python benchmark_run.py --mode both --predictor api --api-endpoint http://localhost:8090/retrieve

# optional: run API in hybrid mode, then benchmark via API predictor
uv run python pageindex_api.py --mode hybrid --base-url http://localhost:8059/v1 --model local-model --port 8090
uv run python benchmark_run.py --mode both --predictor api --api-endpoint http://localhost:8090/retrieve
```

## Benchmark Quick Notes

- `benchmark_build.py` stores questions in `data/benchmark/questions.jsonl` with `question`, `source_file`, `answer_line_start`, `answer_line_end`, and `answer_quote`
- `benchmark_run.py` supports `--mode strict|relaxed|both` and outputs:
  - `data/benchmark/results.json` (aggregate metrics)
  - `data/benchmark/results.jsonl` (per-question diagnostics)
- `benchmark_run.py` supports predictors `local`, `pageindex`, `hybrid`, and `api`
- For faster experiments, run a random subset with `--sample-size N --sample-seed 42`
- Strict scoring requires line-aware chunk metadata from `chunk.py`, so re-run chunk+embed after chunking changes

## Key Details

- **Embeddings model**: `nomic-ai/CodeRankEmbed` (768-dim, 8192 context) via llama.cpp
- **Vector store**: Local ChromaDB in `data/chroma/`
- **Query handling**: Automatically prefixes queries for the bi-encoder and appends Symfony context
- **Tree artifacts**: `data/pageindex/tree.json` and `data/pageindex/nodes.jsonl`
