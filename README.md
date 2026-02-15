## Simple RAG - Symfony Docs

Proof-of-concept RAG pipeline for benchmarking retrieval over Symfony documentation.

### Setup

```bash
uv sync
```

### Build Index

```bash
# 1) Chunk docs
uv run python chunk.py

# 2) Embed chunks into local ChromaDB (embeddings endpoint on :8059)
uv run python embed.py --base-url http://localhost:8059/v1 --reset
```

### Retrieve Manually

```bash
uv run python retrieve.py --base-url http://localhost:8059/v1
```

### Build Benchmark Questions

```bash
# Fast local heuristic generation
uv run python benchmark_build.py --generator heuristic

# LLM generation (slower, model endpoint on :8052)
uv run python benchmark_build.py --generator llm --base-url http://localhost:8052/v1 --model local-model
```

Useful options:
- `--limit N` to test on a subset
- `--resume` to append while skipping duplicate IDs
- `--shard-index I --shard-count N` for split runs

Output: `data/benchmark/questions.jsonl`

### Run Benchmark

```bash
# Run both strict and relaxed scoring
uv run python benchmark_run.py --mode both --base-url http://localhost:8059/v1

# Faster iteration: run on a random subset (e.g. 200 questions)
uv run python benchmark_run.py --mode both --base-url http://localhost:8059/v1 --sample-size 200 --sample-seed 42
```

Scoring modes:
- `strict`: source file matches and chunk line range overlaps gold line range
- `relaxed`: source file match only
- `both`: computes both

Outputs:
- Summary JSON: `data/benchmark/results.json`
- Per-question JSONL: `data/benchmark/results.jsonl`

Reported metrics include `hit@1` and `hit@5`.

### PageIndex-Style Tree RAG (Separate Pipeline)

This repo also includes an experimental, vectorless tree-retrieval pipeline inspired by PageIndex.
It is separate from the existing chunk+embedding pipeline.

```bash
# 1) Build tree index from RST docs
uv run python pageindex_build.py

# optional: generate LLM summaries during build
uv run python pageindex_build.py --summary-mode llm --summary-base-url http://localhost:8052/v1 --summary-model local-model

# 2) Query tree index (REPL or one-shot)
uv run python pageindex_retrieve.py --base-url http://localhost:8052/v1

# 3) Serve benchmark-compatible API
uv run python pageindex_api.py --base-url http://localhost:8052/v1 --port 8090

# 4) Run benchmark against PageIndex API
uv run python benchmark_run.py \
  --predictor api \
  --api-endpoint http://localhost:8090/retrieve \
  --mode both \
  --results data/benchmark/results_pageindex.json \
  --results-jsonl data/benchmark/results_pageindex.jsonl
```

Notes:
- Index artifacts are written to `data/pageindex/`.
- `pageindex_retrieve.py` and `pageindex_api.py` support `--llm-final-only` to skip tree-step LLM calls and use LLM only for final ranking.
- In LLM mode, retrieval sends both summary and body excerpts to the model (`--step-text-chars` and `--final-text-chars` tune payload size).
- Design/implementation plan is stored in `PAGEINDEX_PLAN.md`.
